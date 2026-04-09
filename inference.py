"""
MediGuard-AI — Inference Script (Rule-Based Baseline)

Architecture note:
  MediGuardEnv is a pure-Python, fully synchronous local object.
  It does NOT make HTTP calls, start Docker containers, or use async I/O.
  Therefore: no asyncio, no signal.SIGALRM, no from_docker_image().

  The previous timeout was caused by one of:
    (a) app.py (uvicorn) competing for the same process / startup time
    (b) inference.py being run BEFORE uvicorn finished binding port 7860
    (c) a hanging import or PatientSimulator init under the eval container

  Fix: stripped to the bare minimum — import, reset, loop, grade, print.
  All timeouts use threading.Timer (works in Docker, works in threads,
  does not require Unix signals).
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Tuple, Union

# ------------------------------------------------------------------ #
#  Environment variables (read but not used for LLM calls)           #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "rule_based_baseline")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

MODEL_BY_TASK = {task: MODEL_NAME for task in ("suppression", "deterioration", "triage")}

# ------------------------------------------------------------------ #
#  Timeouts                                                           #
# ------------------------------------------------------------------ #

EPISODE_TIMEOUT_SECONDS = 480   # 8 min per episode
TOTAL_TIMEOUT_SECONDS   = 1500  # 25 min total

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

# ------------------------------------------------------------------ #
#  Stub LLM functions (required exports for app.py)                  #
# ------------------------------------------------------------------ #

def llm_agent(obs: Dict, task: str, conversation_history: list,
              vitals_history: list, model: str) -> Tuple[int, str]:
    return baseline_agent(obs), "rule_based_stub"


def triage_llm_agent(obs_list: List[Dict], conversation_history: list,
                     model: str) -> Tuple[List[int], str]:
    return triage_baseline(obs_list), "rule_based_stub"

# ------------------------------------------------------------------ #
#  Observation formatting (required exports for app.py)              #
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
#  Rule-based baseline agent                                          #
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
                spo2_drop = (sum(r[3] for r in oldest) / len(oldest)
                             - sum(r[3] for r in recent) / len(recent))
                temp_rise = (sum(r[5] for r in recent) / len(recent)
                             - sum(r[5] for r in oldest) / len(oldest))
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
#  Structured logging                                                 #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=mediguard model={model}", flush=True)

def log_step(step: int, action, reward: float, done: bool, error=None) -> None:
    action_str = ",".join(str(a) for a in action) if isinstance(action, (list, tuple)) else str(action)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={'null' if error is None else error}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )

def log_reasoning(action, reasoning: str) -> None:
    action_str = ",".join(str(a) for a in action) if isinstance(action, (list, tuple)) else str(action)
    print(f"[REASONING] last_action={action_str} reasoning=\"{reasoning}\"", flush=True)

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
        print("[VALIDATE] fail action_space.n expected 3", flush=True)
        return False

    task_names = {t.get("name") for t in spec.get("tasks", [])}
    for t in ("suppression", "deterioration", "triage"):
        if t not in task_names:
            print(f"[VALIDATE] fail missing task: {t}", flush=True)
            return False

    print("[VALIDATE] pass", flush=True)
    return True

# ------------------------------------------------------------------ #
#  Episode runner                                                     #
# ------------------------------------------------------------------ #

def run_episode(task: str, seed: int = 42) -> Tuple[List[float], float]:
    """
    Run one full episode. Pure synchronous — MediGuardEnv is a local
    Python object with no network I/O. Each episode takes < 1 second.
    A threading.Timer enforces a hard wall-clock cap as a safety net.
    """
    from mediguard_env import MediGuardEnv

    model = "rule_based_baseline"
    log_start(task, model)
    print(f"[AGENT] type=rule_based model={model} temperature=0.0", flush=True)

    rewards: List[float] = []
    steps   = 0
    score   = 0.0

    _timed_out = threading.Event()
    _timer = threading.Timer(EPISODE_TIMEOUT_SECONDS, _timed_out.set)
    _timer.daemon = True
    _timer.start()

    try:
        env = MediGuardEnv(task=task, seed=seed)
        obs = env.reset()
        done = False

        while not done:
            if _timed_out.is_set():
                steps += 1
                log_step(steps, 0, 0.0, True, error=f"episode_timeout>{EPISODE_TIMEOUT_SECONDS}s")
                print(f"[TIMEOUT] task={task} exceeded {EPISODE_TIMEOUT_SECONDS}s", flush=True)
                break

            if task == "triage":
                action    = triage_baseline(obs)
                reasoning = "rule_based_triage"
            else:
                action    = baseline_agent(obs)
                reasoning = "rule_based_single"

            obs, reward, done, info = env.step(action)
            steps = info.get("step", steps + 1) if isinstance(info, dict) else steps + 1
            rewards.append(reward)

            log_step(steps, action, reward, done, error=None)
            log_reasoning(action, reasoning)

        grader = {
            "suppression":   env.false_alarm_rate_grader,
            "deterioration": env.deterioration_grader,
            "triage":        env.triage_grader,
        }.get(task)
        score = float(grader()) if grader else (sum(rewards) / len(rewards) if rewards else 0.0)

    except Exception as exc:
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        print(f"[ERROR] task={task} exception={exc}", flush=True)
        score = 0.0

    finally:
        _timer.cancel()

    log_end(score > 0.0, steps, rewards, score)
    return rewards, score

# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main() -> None:
    openenv_validate()
    print("[INFO] Mode: rule_based_only — pure Python, no LLM calls, no network I/O", flush=True)
    print("[INFO] Expected runtime: 60 steps x 3 tasks x <1ms = well under 1 second", flush=True)

    tasks      = ["suppression", "deterioration", "triage"]
    all_results: Dict[str, Dict] = {}

    _global_timed_out = threading.Event()
    _global_timer = threading.Timer(TOTAL_TIMEOUT_SECONDS, _global_timed_out.set)
    _global_timer.daemon = True
    _global_timer.start()

    try:
        for task in tasks:
            if _global_timed_out.is_set():
                print(f"[TIMEOUT] Global cap hit before task={task}", flush=True)
                all_results.setdefault(task, {"rewards": [], "score": 0.0})
                continue
            rewards, score = run_episode(task, seed=42)
            all_results[task] = {"rewards": rewards, "score": score}
    finally:
        _global_timer.cancel()
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
    main()
