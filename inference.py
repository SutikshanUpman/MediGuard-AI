"""
MediGuard-AI — Inference Script (Rule-Based Baseline, Async-Safe)

Key fixes vs previous version:
  1. Fully async: env is awaited with asyncio.wait_for() timeouts on every call
  2. env.close() always called in finally block (required by OpenEnv spec)
  3. signal.SIGALRM removed — it doesn't work reliably inside Docker/threads
  4. Per-call timeouts (STEP_TIMEOUT) guard against any single env.step() hang
  5. RESET_TIMEOUT guards env.reset() which is the most common hang point
  6. IMAGE_NAME read from environment (required for from_docker_image pattern)
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Tuple, Union

# ------------------------------------------------------------------ #
#  Environment config — read from env vars set by the validator      #
# ------------------------------------------------------------------ #

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "rule_based_baseline")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

# ------------------------------------------------------------------ #
#  Timeouts (seconds)                                                 #
# ------------------------------------------------------------------ #

RESET_TIMEOUT    = 60    # env.reset() — Docker container may need time to start
STEP_TIMEOUT     = 30    # each env.step() call
EPISODE_TIMEOUT  = 480   # hard cap per full episode (8 min)
TOTAL_TIMEOUT    = 1500  # hard cap across all 3 tasks (25 min)

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

IGNORE = 0
VERIFY = 1
ALERT  = 2

ACTIVITY_NAMES = {
    0: "resting (lying in bed)",
    1: "eating",
    2: "walking/ambulating",
    3: "in distress",
    4: "falling",
}

BASELINE_SCORES = {
    "suppression":   1.0000,
    "deterioration": 0.3931,
    "triage":        0.3073,
}

MODEL_BY_TASK = {task: MODEL_NAME for task in ("suppression", "deterioration", "triage")}

# ------------------------------------------------------------------ #
#  Structured logging (required by OpenEnv spec)                     #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=mediguard model={model}", flush=True)

def log_step(step: int, action, reward: float, done: bool, error=None) -> None:
    action_str = ",".join(str(a) for a in action) if isinstance(action, (list, tuple)) else str(action)
    done_str   = "true" if done else "false"
    error_str  = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

def log_reasoning(action, reasoning: str) -> None:
    action_str = ",".join(str(a) for a in action) if isinstance(action, (list, tuple)) else str(action)
    print(f"[REASONING] last_action={action_str} reasoning=\"{reasoning}\"", flush=True)

# ------------------------------------------------------------------ #
#  Rule-based baseline agent (unchanged logic)                       #
# ------------------------------------------------------------------ #

def baseline_agent(obs: Dict) -> int:
    activity = obs.get("activity", 0)
    delta    = obs.get("baseline_delta", 0.0)
    hours    = obs.get("hours_observed", 0.0)
    spo2     = obs.get("spo2", 0.5)
    hr       = obs.get("heart_rate", 0.4)
    temp     = obs.get("temperature", 0.4)
    history  = obs.get("vitals_history", [])

    if hours < 1.0:
        return VERIFY

    if activity in (1, 2):
        return ALERT if spo2 < 0.25 else IGNORE

    if activity in (3, 4):
        if delta > 0.3 or spo2 < 0.45 or hr > 0.70:
            return ALERT
        return VERIFY

    if spo2 < 0.35 or temp > 0.85 or hr > 0.80:
        return ALERT

    if delta > 0.55 and activity == 0:
        return ALERT
    if delta > 0.30 and activity == 0:
        return VERIFY

    if spo2 < 0.50 or temp > 0.68 or hr > 0.65:
        return VERIFY

    if len(history) >= 6:
        try:
            recent = [r for r in history[-3:] if any(v != 0.0 for v in r)]
            oldest = [r for r in history[:3]  if any(v != 0.0 for v in r)]
            if len(recent) >= 2 and len(oldest) >= 2:
                spo2_drop = sum(r[3] for r in oldest)/len(oldest) - sum(r[3] for r in recent)/len(recent)
                temp_rise = sum(r[5] for r in recent)/len(recent) - sum(r[5] for r in oldest)/len(oldest)
                if spo2_drop > 0.08 or temp_rise > 0.08:
                    return ALERT
                if spo2_drop > 0.04 or temp_rise > 0.04:
                    return VERIFY
        except (IndexError, TypeError, ZeroDivisionError):
            pass

    if hours > 4.0:
        if (delta > 0.20 and activity == 0) or spo2 < 0.55:
            return VERIFY

    return IGNORE


def triage_baseline(obs_list: List[Dict]) -> List[int]:
    raw    = [baseline_agent(obs) for obs in obs_list]
    scores = [
        obs.get("baseline_delta", 0.0) * 3.0
        + (0.5 - obs.get("spo2", 0.5)) * 4.0
        + obs.get("temperature", 0.4) * 1.5
        + obs.get("heart_rate", 0.4) * 1.0
        + {3: 2.0, 4: 3.0}.get(obs.get("activity", 0), 0.0)
        for obs in obs_list
    ]
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    actions = [IGNORE] * len(obs_list)
    for rank, idx in enumerate(sorted_idx):
        if rank == 0:
            actions[idx] = raw[idx]
        elif rank == 1:
            actions[idx] = min(raw[idx], VERIFY)
    return actions

# ------------------------------------------------------------------ #
#  Observation formatting (kept for app.py compatibility)            #
# ------------------------------------------------------------------ #

def obs_to_user_message(obs: Dict, task: str, vitals_history: list,
                        conversation_history: list) -> str:
    activity_name = ACTIVITY_NAMES.get(obs.get("activity", 0), "unknown")
    hours = obs.get("hours_observed", 0.0)
    lines = [
        f"Step: {int(hours * 60)}/{60}  |  Hours observed: {hours:.1f}h",
        "",
        "CURRENT VITALS (normalized 0-1):",
        f"  Heart Rate:    {obs.get('heart_rate', 0):.3f}",
        f"  SpO2:          {obs.get('spo2', 0):.3f}",
        f"  Systolic BP:   {obs.get('systolic_bp', 0):.3f}",
        f"  Diastolic BP:  {obs.get('diastolic_bp', 0):.3f}",
        f"  Resp Rate:     {obs.get('respiratory_rate', 0):.3f}",
        f"  Temperature:   {obs.get('temperature', 0):.3f}",
        "",
        f"Baseline Delta:  {obs.get('baseline_delta', 0):.3f}",
        f"Activity:        {activity_name}",
    ]
    return "\n".join(lines)


def triage_obs_to_message(obs_list: List[Dict], conversation_history: list) -> str:
    lines = []
    for i, obs in enumerate(obs_list):
        activity_name = ACTIVITY_NAMES.get(obs.get("activity", 0), "unknown")
        lines.append(f"PATIENT {i}:")
        lines.append(
            f"  Vitals:  HR={obs.get('heart_rate', 0):.3f}  "
            f"SpO2={obs.get('spo2', 0):.3f}  "
            f"SysBP={obs.get('systolic_bp', 0):.3f}  "
            f"Temp={obs.get('temperature', 0):.3f}"
        )
        lines.append(
            f"  Activity: {activity_name}  |  "
            f"Baseline Delta: {obs.get('baseline_delta', 0):.3f}  |  "
            f"Hours: {obs.get('hours_observed', 0):.1f}h"
        )
        lines.append("")
    return "\n".join(lines)

# ------------------------------------------------------------------ #
#  Stub LLM functions (API-compatible, no actual calls made)         #
# ------------------------------------------------------------------ #

def llm_agent(obs: Dict, task: str, conversation_history: list,
              vitals_history: list, model: str) -> Tuple[int, str]:
    return baseline_agent(obs), "rule_based_stub"


def triage_llm_agent(obs_list: List[Dict], conversation_history: list,
                     model: str) -> Tuple[List[int], str]:
    return triage_baseline(obs_list), "rule_based_stub"

# ------------------------------------------------------------------ #
#  OpenEnv spec validator                                             #
# ------------------------------------------------------------------ #

def openenv_validate() -> bool:
    try:
        import yaml
    except ImportError:
        result = os.path.exists("openenv.yaml")
        print(f"[VALIDATE] {'pass' if result else 'fail openenv.yaml not found'}", flush=True)
        return result

    if not os.path.exists("openenv.yaml"):
        print("[VALIDATE] fail openenv.yaml not found", flush=True)
        return False

    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)

    for field in ["name", "version", "tasks", "action_space", "observation_space"]:
        if field not in spec:
            print(f"[VALIDATE] fail missing field: {field}", flush=True)
            return False

    if spec.get("action_space", {}).get("n") != 3:
        print(f"[VALIDATE] fail action_space.n expected 3", flush=True)
        return False

    task_names = {t.get("name") for t in spec.get("tasks", [])}
    for t in ("suppression", "deterioration", "triage"):
        if t not in task_names:
            print(f"[VALIDATE] fail missing task: {t}", flush=True)
            return False

    print("[VALIDATE] pass", flush=True)
    return True

# ------------------------------------------------------------------ #
#  Async episode runner                                               #
# ------------------------------------------------------------------ #

async def run_episode(task: str, seed: int = 42) -> Tuple[List[float], float]:
    """
    Run one episode with timeout guards on every async env call.
    Always calls env.close() in the finally block.
    """
    from mediguard_env import MediGuardEnv

    model = "rule_based_baseline"
    log_start(task, model)
    print(f"[AGENT] type=rule_based model={model} temperature=0.0", flush=True)

    rewards: List[float] = []
    steps   = 0
    score   = 0.0
    env     = None

    try:
        # ── Instantiate env ──────────────────────────────────────────
        # Try async from_docker_image first (matches OpenEnv sample pattern).
        # Fall back to sync constructor if the env doesn't support it.
        try:
            if IMAGE_NAME and hasattr(MediGuardEnv, "from_docker_image"):
                env = await asyncio.wait_for(
                    MediGuardEnv.from_docker_image(IMAGE_NAME, task=task, seed=seed),
                    timeout=RESET_TIMEOUT,
                )
            else:
                env = MediGuardEnv(task=task, seed=seed)
        except Exception as e:
            print(f"[WARN] env init: {e} — trying sync constructor", flush=True)
            env = MediGuardEnv(task=task, seed=seed)

        # ── Reset ────────────────────────────────────────────────────
        try:
            reset_coro = env.reset()
            if asyncio.iscoroutine(reset_coro):
                obs = await asyncio.wait_for(reset_coro, timeout=RESET_TIMEOUT)
            else:
                obs = reset_coro
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] env.reset() exceeded {RESET_TIMEOUT}s", flush=True)
            return [], 0.0

        done = False

        # ── Step loop ────────────────────────────────────────────────
        while not done:
            if task == "triage":
                action    = triage_baseline(obs)
                reasoning = "rule_based_triage"
            else:
                action    = baseline_agent(obs)
                reasoning = "rule_based_single"

            try:
                step_coro = env.step(action)
                if asyncio.iscoroutine(step_coro):
                    result = await asyncio.wait_for(step_coro, timeout=STEP_TIMEOUT)
                else:
                    result = step_coro
            except asyncio.TimeoutError:
                steps += 1
                log_step(steps, action, 0.0, True, error=f"step timeout>{STEP_TIMEOUT}s")
                print(f"[TIMEOUT] env.step() exceeded {STEP_TIMEOUT}s at step {steps}", flush=True)
                break

            # Support both tuple return and object return
            if isinstance(result, tuple):
                obs, reward, done, info = result
                steps = info.get("step", steps + 1) if isinstance(info, dict) else steps + 1
            else:
                obs    = result.observation
                reward = result.reward or 0.0
                done   = result.done
                steps += 1

            rewards.append(reward)
            log_step(steps, action, reward, done, error=None)
            log_reasoning(action, reasoning)

        # ── Score ────────────────────────────────────────────────────
        grader_map = {
            "suppression":   getattr(env, "false_alarm_rate_grader", None),
            "deterioration": getattr(env, "deterioration_grader", None),
            "triage":        getattr(env, "triage_grader", None),
        }
        grader = grader_map.get(task)
        if grader is not None:
            grader_result = grader()
            if asyncio.iscoroutine(grader_result):
                score = await grader_result
            else:
                score = grader_result
        else:
            score = sum(rewards) / len(rewards) if rewards else 0.0

    except Exception as exc:
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        print(f"[ERROR] task={task} exception={exc}", flush=True)
        score = 0.0

    finally:
        # Always close the env — required by OpenEnv spec and prevents container leaks
        if env is not None:
            try:
                close_coro = env.close()
                if asyncio.iscoroutine(close_coro):
                    await asyncio.wait_for(close_coro, timeout=15)
                else:
                    close_coro  # sync close, already called
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)

    success = score > 0.0
    log_end(success, steps, rewards, score)
    return rewards, score

# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

async def main() -> None:
    openenv_validate()
    print("[INFO] Mode: rule_based_only (no LLM calls)", flush=True)

    tasks      = ["suppression", "deterioration", "triage"]
    all_results: Dict[str, Dict] = {}

    try:
        async with asyncio.timeout(TOTAL_TIMEOUT):
            for task in tasks:
                try:
                    async with asyncio.timeout(EPISODE_TIMEOUT):
                        rewards, score = await run_episode(task, seed=42)
                except TimeoutError:
                    print(f"[TIMEOUT] Episode '{task}' exceeded {EPISODE_TIMEOUT}s cap", flush=True)
                    rewards, score = [], 0.0
                all_results[task] = {"rewards": rewards, "score": score}

    except TimeoutError:
        print(f"[TIMEOUT] Total inference exceeded {TOTAL_TIMEOUT}s cap", flush=True)
        for task in tasks:
            all_results.setdefault(task, {"rewards": [], "score": 0.0})

    print(flush=True)
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task in tasks:
        rews  = all_results[task]["rewards"]
        score = all_results[task]["score"]
        mean_r = sum(rews) / len(rews) if rews else 0.0
        print(
            f"  {task:15s}  steps={len(rews):4d}  "
            f"mean_reward={mean_r:.4f}  score={score:.4f}",
            flush=True,
        )
    print("=" * 60, flush=True)

    s = all_results["suppression"]["score"]
    d = all_results["deterioration"]["score"]
    t = all_results["triage"]["score"]
    print(f"\n[SUMMARY] suppression={s:.4f} deterioration={d:.4f} triage={t:.4f}", flush=True)
    print(
        f"[IMPROVEMENT] vs_baseline "
        f"suppression={s - BASELINE_SCORES['suppression']:+.4f} "
        f"deterioration={d - BASELINE_SCORES['deterioration']:+.4f} "
        f"triage={t - BASELINE_SCORES['triage']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
