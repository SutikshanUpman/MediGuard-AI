"""
MediGuard-AI — Baseline Inference Script

Runs a rule-based agent against all 3 hackathon tasks and prints
structured logs that the automated scoring pipeline reads.

Environment variables:
  API_BASE_URL — LLM endpoint (default: HuggingFace router)
  MODEL_NAME   — model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN / API_KEY — authentication token

Usage:
  python inference.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Union

from openai import OpenAI
from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  Configuration from environment variables                          #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# OpenAI client — mandatory per hackathon rules.
# The baseline uses rule-based logic, but the client is ready for
# LLM-based agents. Replace baseline_agent() to use this client.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

# Action constants
IGNORE = 0
VERIFY = 1
ALERT  = 2


# ------------------------------------------------------------------ #
#  Baseline rule-based agent                                         #
# ------------------------------------------------------------------ #

def baseline_agent(obs: Dict) -> int:
    """
    Rule-based policy with 3 detection strategies:
      1. Delta-based: sudden deviations from personal baseline
      2. Absolute thresholds: catch slow drifts that delta misses
      3. Trend detection: directional changes in vitals history

    Given a single-patient observation dict, returns an action int:
      0 = Ignore, 1 = Verify, 2 = Alert
    """
    activity = obs["activity"]
    delta    = obs["baseline_delta"]
    hours    = obs["hours_observed"]
    spo2     = obs["spo2"]           # normalized 0-1
    hr       = obs["heart_rate"]     # normalized 0-1
    temp     = obs.get("temperature", 0.4)  # normalized 0-1
    history  = obs.get("vitals_history", [])

    # Still learning baseline — be cautious
    if hours < 1.0:
        return VERIFY

    # Ambulating — elevated HR is expected, don't over-react
    # BUT still check for truly dangerous vitals
    if activity == 2:
        if spo2 < 0.25:  # raw SpO2 < ~77.5% — critical even while walking
            return ALERT
        return IGNORE

    # ── Strategy 1: Delta-based ──
    if delta > 0.6 and activity == 0:
        return ALERT
    if delta > 0.35 and activity == 0:
        return VERIFY

    # ── Strategy 2: Absolute thresholds ──
    # These catch slow drifts that baseline_delta misses
    if spo2 < 0.35:       # raw SpO2 < ~80.5% — dangerously low
        return ALERT
    if spo2 < 0.50:       # raw SpO2 < ~85% — concerning
        return VERIFY

    if temp > 0.80:       # raw temp > ~40.4°C — high fever
        return ALERT
    if temp > 0.65:       # raw temp > ~39.2°C — fever
        return VERIFY

    if hr > 0.75:         # raw HR > ~157 bpm — tachycardia
        return ALERT
    if hr > 0.60:         # raw HR > ~132 bpm — elevated
        return VERIFY

    # ── Strategy 3: Trend detection ──
    # Compare recent vitals vs older vitals in history
    if len(history) >= 6:
        try:
            recent = history[-3:]   # last 3 readings
            oldest = history[:3]    # oldest 3 readings

            # Average SpO2 trend (dropping = bad)
            recent_spo2 = sum(r[3] for r in recent) / 3
            oldest_spo2 = sum(r[3] for r in oldest) / 3
            spo2_drop = oldest_spo2 - recent_spo2

            # Average temp trend (rising = bad)
            recent_temp = sum(r[5] for r in recent) / 3
            oldest_temp = sum(r[5] for r in oldest) / 3
            temp_rise = recent_temp - oldest_temp

            # Significant downward SpO2 trend
            if spo2_drop > 0.08:
                return ALERT
            if spo2_drop > 0.04:
                return VERIFY

            # Significant upward temp trend
            if temp_rise > 0.08:
                return ALERT
            if temp_rise > 0.04:
                return VERIFY
        except (IndexError, TypeError):
            pass

    # ── Time-based escalation ──
    # After 4 hours, lower thresholds
    if hours > 4.0:
        if delta > 0.20 and activity == 0:
            return VERIFY
        if spo2 < 0.55:
            return VERIFY

    return IGNORE


def triage_agent(obs_list: List[Dict]) -> List[int]:
    """Apply baseline_agent independently to each patient in triage mode."""
    return [baseline_agent(obs) for obs in obs_list]


# ------------------------------------------------------------------ #
#  Logging helpers                                                   #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str):
    print(f"[START] task={task} env=mediguard model={model}", flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    """Print a [STEP] line in the mandatory format."""
    # Format action — triage uses comma-separated ints
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)

    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error)

    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], score: float = 0.0):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ------------------------------------------------------------------ #
#  Main episode runner                                               #
# ------------------------------------------------------------------ #

def run_episode(task: str, seed: int = 42):
    """Run a single episode for the given task and print structured logs."""
    log_start(task, MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    success = False

    try:
        env = MediGuardEnv(task=task, seed=seed)
        obs = env.reset()
        done = False

        while not done:
            # Choose action
            if task == "triage":
                action = triage_agent(obs)
            else:
                action = baseline_agent(obs)

            # Step
            obs, reward, done, info = env.step(action)
            steps = info["step"]
            rewards.append(reward)

            log_step(steps, action, reward, done, error=None)

        # Compute grader score
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = mean_reward > 0.4

        # Call task-specific grader
        grader_map = {
            "suppression": env.false_alarm_rate_grader,
            "deterioration": env.deterioration_grader,
            "triage": env.triage_grader,
        }
        score = grader_map[task]()

    except Exception as exc:
        # Log the failing step and then end
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        success = False
        score = 0.0

    log_end(success, steps, rewards, score=score)
    return rewards, score


# ------------------------------------------------------------------ #
#  Entry point                                                       #
# ------------------------------------------------------------------ #

def main():
    tasks = ["suppression", "deterioration", "triage"]
    all_results = {}

    for task in tasks:
        rewards, score = run_episode(task, seed=42)
        all_results[task] = {"rewards": rewards, "score": score}

    # Summary (not part of scored output, but useful for humans)
    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for task in tasks:
        rews = all_results[task]["rewards"]
        score = all_results[task]["score"]
        mean_r = sum(rews) / len(rews) if rews else 0.0
        print(f"  {task:15s}  steps={len(rews):4d}  mean_reward={mean_r:.4f}  score={score:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
