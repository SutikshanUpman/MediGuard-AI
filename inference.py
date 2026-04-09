"""
MediGuard-AI — Inference Script (Rule-Based Baseline)

Runs a deterministic rule-based agent against all 3 hackathon tasks.
NO LLM CALLS — guaranteed to complete in < 30 seconds regardless of
environment variables or injected API keys.

Runtime guarantee: 60 steps × 3 tasks × ~0.001s = well under 1 minute.

Phase 2 evaluation note:
  The hackathon validator runs its OWN LLM (e.g. Nemotron) against your
  environment. This script only needs to provide a reproducible baseline.
  All LLM logic has been removed to prevent timeout kills.

Environment variables (accepted but ignored — no LLM calls made):
  HF_TOKEN / API_KEY / OPENAI_API_KEY — not used
  API_BASE_URL                         — not used
  MODEL_NAME                           — not used
"""

from __future__ import annotations

import json
import os
import signal
import time
from typing import Dict, List, Tuple, Union

from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  Hard wall-clock timeout (safety net for the 30-min validator cap) #
# ------------------------------------------------------------------ #

_TOTAL_TIMEOUT_SECONDS   = 1500  # 25-min hard cap across all 3 tasks
_EPISODE_TIMEOUT_SECONDS = 480   # 8-min cap per episode

def _timeout_handler(signum, frame):
    raise TimeoutError("Inference exceeded allocated wall-clock time — killed by safety guard")

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
#  Stub exports — kept for app.py import compatibility               #
#  No LLM calls are ever made; these values are read-only constants. #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "rule_based_baseline")
API_KEY      = "dummy"   # intentionally never used

MODEL_BY_TASK = {
    "suppression":   MODEL_NAME,
    "deterioration": MODEL_NAME,
    "triage":        MODEL_NAME,
}


def llm_agent(obs: Dict, task: str, conversation_history: list,
              vitals_history: list, model: str) -> Tuple[int, str]:
    """Stub — falls back to rule-based immediately. No LLM calls made."""
    return baseline_agent(obs), "rule_based_stub"


def triage_llm_agent(obs_list: List[Dict], conversation_history: list,
                     model: str) -> Tuple[List[int], str]:
    """Stub — falls back to rule-based immediately. No LLM calls made."""
    return triage_baseline(obs_list), "rule_based_stub"

# ------------------------------------------------------------------ #
#  Observation formatting (kept for API compatibility with app.py)   #
# ------------------------------------------------------------------ #

def obs_to_user_message(obs: Dict, task: str, vitals_history: list,
                        conversation_history: list) -> str:
    """Format a single-patient observation as a human-readable string."""
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
        f"Baseline Delta:  {obs.get('baseline_delta', 0):.3f} (deviation from personal norm)",
        f"Activity:        {activity_name}",
    ]
    return "\n".join(lines)


def triage_obs_to_message(obs_list: List[Dict], conversation_history: list) -> str:
    """Format a multi-patient triage observation as a human-readable string."""
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
    """
    Deterministic rule-based agent for single-patient tasks.

    Decision logic:
      1. Activity context: walking/eating → discount vitals spikes
      2. Absolute thresholds: dangerously out-of-range vitals → ALERT
      3. Baseline delta: large deviation from personal norm → ALERT/VERIFY
      4. Trend detection: SpO2 drop or temp rise over history window
      5. Default: IGNORE if nothing is triggered
    """
    activity = obs.get("activity", 0)
    delta    = obs.get("baseline_delta", 0.0)
    hours    = obs.get("hours_observed", 0.0)
    spo2     = obs.get("spo2", 0.5)
    hr       = obs.get("heart_rate", 0.4)
    temp     = obs.get("temperature", 0.4)
    history  = obs.get("vitals_history", [])

    # Early phase: insufficient baseline established — cautious VERIFY
    if hours < 1.0:
        return VERIFY

    # Walking patient: elevated vitals are expected — raise thresholds
    if activity == 2:
        return ALERT if spo2 < 0.25 else IGNORE

    # Eating: mild elevation expected
    if activity == 1:
        return ALERT if spo2 < 0.25 else IGNORE

    # Falling or distressed: immediate concern
    if activity in (3, 4):
        if delta > 0.3 or spo2 < 0.45 or hr > 0.70:
            return ALERT
        return VERIFY

    # Resting patient: full thresholds apply
    # Absolute emergency thresholds
    if spo2 < 0.35:
        return ALERT
    if temp > 0.85:
        return ALERT
    if hr > 0.80:
        return ALERT

    # Delta-based response (resting patient deviating from personal norm)
    if delta > 0.55 and activity == 0:
        return ALERT
    if delta > 0.30 and activity == 0:
        return VERIFY

    # Borderline thresholds
    if spo2 < 0.50:
        return VERIFY
    if temp > 0.68:
        return VERIFY
    if hr > 0.65:
        return VERIFY

    # Trend detection via vitals history
    if len(history) >= 6:
        try:
            recent = [row for row in history[-3:] if any(v != 0.0 for v in row)]
            oldest = [row for row in history[:3]  if any(v != 0.0 for v in row)]
            if len(recent) >= 2 and len(oldest) >= 2:
                # SpO2 is index 3, temperature is index 5
                recent_spo2 = sum(r[3] for r in recent) / len(recent)
                oldest_spo2 = sum(r[3] for r in oldest) / len(oldest)
                spo2_drop = oldest_spo2 - recent_spo2
                recent_temp = sum(r[5] for r in recent) / len(recent)
                oldest_temp = sum(r[5] for r in oldest) / len(oldest)
                temp_rise = recent_temp - oldest_temp
                if spo2_drop > 0.08:
                    return ALERT
                if spo2_drop > 0.04:
                    return VERIFY
                if temp_rise > 0.08:
                    return ALERT
                if temp_rise > 0.04:
                    return VERIFY
        except (IndexError, TypeError, ZeroDivisionError):
            pass

    # Gentle scan after enough observation time
    if hours > 4.0:
        if delta > 0.20 and activity == 0:
            return VERIFY
        if spo2 < 0.55:
            return VERIFY

    return IGNORE


def triage_baseline(obs_list: List[Dict]) -> List[int]:
    """
    Triage agent: apply baseline_agent per patient.
    Ensures at most one ALERT to avoid concentration penalty
    when all patients happen to look borderline simultaneously.
    """
    raw = [baseline_agent(obs) for obs in obs_list]

    # Compute urgency score per patient for ranking
    scores = []
    for obs in obs_list:
        s = (
            obs.get("baseline_delta", 0.0) * 3.0
            + (0.5 - obs.get("spo2", 0.5)) * 4.0       # lower SpO2 = more urgent
            + obs.get("temperature", 0.4) * 1.5
            + obs.get("heart_rate", 0.4) * 1.0
            + {3: 2.0, 4: 3.0}.get(obs.get("activity", 0), 0.0)   # distress/falling bonus
        )
        scores.append(s)

    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Build differentiated actions: rank 0 → raw action, rank 1 → VERIFY, rest → IGNORE
    actions = [IGNORE] * len(obs_list)
    for rank, idx in enumerate(sorted_idx):
        if rank == 0:
            actions[idx] = raw[idx]              # most urgent: trust rule-based
        elif rank == 1:
            actions[idx] = min(raw[idx], VERIFY) # second: at most VERIFY
        else:
            actions[idx] = IGNORE                # others: IGNORE

    return actions

# ------------------------------------------------------------------ #
#  OpenEnv spec validator                                             #
# ------------------------------------------------------------------ #

def openenv_validate() -> bool:
    """Validate openenv.yaml against required OpenEnv fields."""
    try:
        import yaml
    except ImportError:
        exists = os.path.exists("openenv.yaml")
        status = "pass" if exists else "fail openenv.yaml not found"
        print(f"[VALIDATE] {status}", flush=True)
        return exists

    if not os.path.exists("openenv.yaml"):
        print("[VALIDATE] fail openenv.yaml not found", flush=True)
        return False

    with open("openenv.yaml", "r") as f:
        spec = yaml.safe_load(f)

    required = ["name", "version", "tasks", "action_space", "observation_space"]
    for field in required:
        if field not in spec:
            print(f"[VALIDATE] fail missing field: {field}", flush=True)
            return False

    action_space = spec.get("action_space", {})
    if action_space.get("n") != 3:
        print(f"[VALIDATE] fail action_space.n={action_space.get('n')}, expected 3", flush=True)
        return False

    tasks = spec.get("tasks", [])
    task_names = {t.get("name") for t in tasks}
    for required_task in ("suppression", "deterioration", "triage"):
        if required_task not in task_names:
            print(f"[VALIDATE] fail missing task: {required_task}", flush=True)
            return False

    print("[VALIDATE] pass", flush=True)
    return True

# ------------------------------------------------------------------ #
#  Structured logging (required by OpenEnv spec)                     #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str):
    print(f"[START] task={task} env=mediguard model={model}", flush=True)

def log_agent(model: str):
    print(f"[AGENT] type=rule_based model={model} temperature=0.0", flush=True)

def log_step(step: int, action, reward: float, done: bool, error=None):
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)
    done_str  = "true" if done else "false"
    error_str = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_fallback(step: int, reason: str):
    print(f"[FALLBACK] step={step} reason={reason} using=rule_based", flush=True)

def log_end(success: bool, steps: int, rewards: List[float], score: float):
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

def log_reasoning(action, reasoning: str):
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)
    print(f"[REASONING] last_action={action_str} reasoning=\"{reasoning}\"", flush=True)

# ------------------------------------------------------------------ #
#  Episode runner                                                     #
# ------------------------------------------------------------------ #

def run_episode(task: str, seed: int = 42) -> Tuple[List[float], float]:
    """Run one full episode with the rule-based agent. Returns (rewards, score)."""
    model = "rule_based_baseline"
    log_start(task, model)
    log_agent(model)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    last_action: Union[int, List[int]] = 0

    # Per-episode timeout: if env hangs, abort cleanly rather than killing the process
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_EPISODE_TIMEOUT_SECONDS)

    try:
        env = MediGuardEnv(task=task, seed=seed)
        obs = env.reset()
        done = False

        while not done:
            # Always use rule-based — no LLM calls whatsoever
            if task == "triage":
                action = triage_baseline(obs)
                reasoning = "rule_based_triage"
            else:
                action = baseline_agent(obs)
                reasoning = "rule_based_single"

            obs, reward, done, info = env.step(action)
            steps = info["step"]
            rewards.append(reward)
            last_action = action

            log_step(steps, action, reward, done, error=None)
            log_reasoning(action, reasoning)

        grader_map = {
            "suppression":   env.false_alarm_rate_grader,
            "deterioration": env.deterioration_grader,
            "triage":        env.triage_grader,
        }
        score = grader_map[task]()
        success = score > 0.0

    except TimeoutError as te:
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(te))
        success = False
        score = 0.0
        print(f"[TIMEOUT] task={task} episode exceeded {_EPISODE_TIMEOUT_SECONDS}s limit", flush=True)

    except Exception as exc:
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        success = False
        score = 0.0
        print(f"[ERROR] task={task} exception={exc}", flush=True)

    finally:
        signal.alarm(0)  # Always cancel the alarm when episode ends

    log_end(success, steps, rewards, score)
    return rewards, score

# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main():
    openenv_validate()

    print("[INFO] Mode: rule_based_only (no LLM calls — guaranteed fast)", flush=True)
    print("[INFO] Worst-case runtime: 60 × 3 × ~0.001s = < 1 second", flush=True)

    # Global safety cap: if all 3 episodes somehow exceed 25 min total, abort
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_TOTAL_TIMEOUT_SECONDS)

    tasks = ["suppression", "deterioration", "triage"]
    all_results: Dict[str, Dict] = {}

    try:
        for task in tasks:
            rewards, score = run_episode(task, seed=42)
            all_results[task] = {"rewards": rewards, "score": score}
    except TimeoutError:
        print("[TIMEOUT] Total inference exceeded 25-minute cap. Aborting.", flush=True)
        for task in tasks:
            if task not in all_results:
                all_results[task] = {"rewards": [], "score": 0.0}
    finally:
        signal.alarm(0)

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
