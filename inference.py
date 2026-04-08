"""
MediGuard-AI — LLM-Based Inference Script

Runs an LLM agent (via OpenAI client) against all 3 hackathon tasks.
The LLM reasons about patient vitals, activity context, and trends
to make IGNORE/VERIFY/ALERT decisions.

Falls back to a rule-based baseline agent on API errors.

Environment variables:
  HF_TOKEN / API_KEY / OPENAI_API_KEY — authentication token
  API_BASE_URL — LLM endpoint (default: HuggingFace router)
  MODEL_NAME   — model identifier (default: Qwen/Qwen2.5-72B-Instruct)

Usage:
  python inference.py
"""

from __future__ import annotations

import json 
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI
from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  Configuration from environment variables                          #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "dummy"
)

# Per-task model selection: easy task uses smaller model to save cost
MODEL_BY_TASK = {
    "suppression":   "Qwen/Qwen2.5-7B-Instruct",
    "deterioration": MODEL_NAME,
    "triage":        MODEL_NAME,
}

# OpenAI client — used for all LLM calls
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Action constants
IGNORE = 0
VERIFY = 1
ALERT  = 2

# Activity name mapping: int → human-readable string
ACTIVITY_NAMES = {
    0: "resting (lying in bed)",
    1: "eating",
    2: "walking/ambulating",
    3: "in distress",
    4: "falling",
}

# Baseline scores for improvement delta calculation
BASELINE_SCORES = {
    "suppression":   1.0000,
    "deterioration": 0.3931,
    "triage":        0.3073,
}


# ------------------------------------------------------------------ #
#  System Prompts — task-specific for optimal scoring                 #
# ------------------------------------------------------------------ #

SUPPRESSION_PROMPT = """You are an ICU monitoring AI. Your job: decide if a patient's vitals need attention.

KEY INSIGHT: Vitals are normalized relative to THIS patient's personal baseline (0-1 scale).
A baseline_delta of 0.3 while walking → normal (exertion raises HR).
A baseline_delta of 0.3 while lying still → concerning.

RULES FOR SUPPRESSION TASK:
- This patient is CHRONICALLY HYPERTENSIVE. High BP is NORMAL for them.
- Default to IGNORE unless evidence is compelling.
- Activity context heavily discounts vitals spikes: walking/eating → IGNORE.
- False alarms are MORE costly than missed mild events.
- Only ALERT for baseline_delta > 0.5 WITH resting/distressed/falling context.
- VERIFY is the safe middle ground — use it for uncertain cases.
- After learning the baseline (hours > 1.0), trust that stable readings are fine.

Respond with ONLY JSON: {"action": 0, "reasoning": "max 15 words"}
action: 0=IGNORE, 1=VERIFY, 2=ALERT"""

DETERIORATION_PROMPT = """You are an ICU monitoring AI detecting slow patient deterioration (sepsis).

KEY INSIGHT: Vitals are normalized 0-1. Look for TRENDS, not just current values.
A single reading of SpO2=0.80 might be noise. But SpO2 dropping 0.93→0.88→0.83→0.80 over 4 steps = real deterioration.

RULES FOR DETERIORATION TASK:
- Read the vitals history table carefully — it shows the last 8 readings.
- Look for TRENDS: consistent directional changes across multiple steps.
- SpO2 dropping is MORE urgent than HR rising — weight it higher.
- Temperature rising steadily over multiple steps = infection sign → VERIFY/ALERT.
- A single spike that returns to normal = noise → IGNORE.
- Early detection scores HIGHER than late detection — act on trends, don't wait for crisis.
- If SpO2 is below 0.50 or temp above 0.65, ALERT regardless of trend.

Respond with ONLY JSON: {"action": 0, "reasoning": "max 15 words"}
action: 0=IGNORE, 1=VERIFY, 2=ALERT"""

TRIAGE_PROMPT = """You are an ICU monitoring AI triaging 4 patients simultaneously.

KEY INSIGHT: You must RANK patients by urgency, not just evaluate each independently.
The most critical patient gets ALERT, second-most gets VERIFY, stable ones get IGNORE.

RULES FOR TRIAGE TASK:
- Look at ALL 4 patients before deciding — this is a ranking problem.
- ALERT the most critical patient (highest baseline_delta, worst vitals, dangerous activity).
- VERIFY the second most concerning patient.
- IGNORE patients who look stable.
- If two patients are both critical, ALERT both.
- Activity matters: falling/distressed beats resting which beats walking/eating.
- High baseline_delta + resting = more concerning than high delta + walking.
- Don't ALERT more than 2 patients unless multiple are genuinely critical.

Respond with ONLY JSON: {"actions": [a0,a1,a2,a3], "reasoning": "max 20 words"}
Each action: 0=IGNORE, 1=VERIFY, 2=ALERT. Array position = patient index."""

SYSTEM_PROMPTS = {
    "suppression":   SUPPRESSION_PROMPT,
    "deterioration": DETERIORATION_PROMPT,
    "triage":        TRIAGE_PROMPT,
}


# ------------------------------------------------------------------ #
#  Observation → LLM message formatting                              #
# ------------------------------------------------------------------ #

def obs_to_user_message(obs: Dict, task: str, vitals_history: list,
                        conversation_history: list) -> str:
    """Format an observation dict into a clear LLM prompt."""
    activity_name = ACTIVITY_NAMES.get(obs.get("activity", 0), "unknown")
    hours = obs.get("hours_observed", 0.0)

    lines = [
        f"Step: {int(hours * 60)}/{360}  |  Hours observed: {hours:.1f}h",
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

    # For deterioration: include trend table from last 8 readings
    if task == "deterioration" and len(vitals_history) >= 3:
        lines.append("")
        lines.append("VITALS TREND (last readings, oldest→newest):")
        lines.append("  Step   HR     SpO2   SysBP  Temp")
        n = len(vitals_history)
        for i, reading in enumerate(vitals_history):
            offset = -(n - i)
            if len(reading) >= 6:
                lines.append(
                    f"  {offset:+4d}   {reading[0]:.3f}  {reading[3]:.3f}  "
                    f"{reading[1]:.3f}  {reading[5]:.3f}"
                )

    # Include recent action/reward feedback for pseudo-online-learning
    if conversation_history:
        lines.append("")
        lines.append("YOUR RECENT DECISIONS:")
        for entry in conversation_history[-3:]:
            lines.append(
                f"  Action={entry['action']} → reward={entry['reward']:.2f}"
            )

    return "\n".join(lines)


def triage_obs_to_message(obs_list: List[Dict],
                          conversation_history: list) -> str:
    """Format 4 patient observations into a single triage prompt."""
    lines = []

    for i, obs in enumerate(obs_list):
        activity_name = ACTIVITY_NAMES.get(obs.get("activity", 0), "unknown")
        lines.append(f"PATIENT {i} :")
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

    lines.append("Rank these patients by urgency. Assign actions:")
    lines.append("  Most critical → ALERT(2). Second → VERIFY(1). Stable → IGNORE(0).")

    # Recent feedback
    if conversation_history:
        lines.append("")
        lines.append("YOUR RECENT DECISIONS:")
        for entry in conversation_history[-2:]:
            lines.append(
                f"  Actions={entry['action']} → reward={entry['reward']:.2f}"
            )

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  LLM Agent — single patient                                       #
# ------------------------------------------------------------------ #

def llm_agent(
    obs: Dict,
    task: str,
    conversation_history: list,
    vitals_history: list,
    model: str,
) -> Tuple[int, str]:
    """
    Call the LLM to decide an action for a single patient.

    Returns (action_int, reasoning_string).
    Raises on API errors — caller handles fallback.
    """
    system_prompt = SYSTEM_PROMPTS[task]
    user_message = obs_to_user_message(obs, task, vitals_history, conversation_history)

    # Build messages with conversation context
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent context for pseudo-online learning
    for entry in conversation_history[-2:]:
        messages.append({"role": "user", "content": entry["obs_text"][:500]})
        messages.append({"role": "assistant", "content": entry["response"]})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=120,
        timeout=8.0,
    )

    raw_response = response.choices[0].message.content.strip()

    # Parse JSON defensively
    action, reasoning = _parse_single_response(raw_response)
    return action, reasoning


def _parse_single_response(raw: str) -> Tuple[int, str]:
    """Parse LLM response into (action, reasoning). Raises ValueError on failure."""
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        inner = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        text = inner.strip()

    data = json.loads(text)
    action = int(data.get("action", 1))
    reasoning = str(data.get("reasoning", ""))[:80]

    if action not in (0, 1, 2):
        action = max(0, min(2, action))

    return action, reasoning


# ------------------------------------------------------------------ #
#  LLM Agent — triage (4 patients in ONE call)                      #
# ------------------------------------------------------------------ #

def triage_llm_agent(
    obs_list: List[Dict],
    conversation_history: list,
    model: str,
) -> Tuple[List[int], str]:
    """
    Call the LLM once to decide actions for all 4 patients.

    Returns (list_of_4_actions, reasoning_string).
    Raises on API errors — caller handles fallback.
    """
    system_prompt = SYSTEM_PROMPTS["triage"]
    user_message = triage_obs_to_message(obs_list, conversation_history)

    messages = [{"role": "system", "content": system_prompt}]

    for entry in conversation_history[-2:]:
        messages.append({"role": "user", "content": entry["obs_text"][:500]})
        messages.append({"role": "assistant", "content": entry["response"]})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=120,
        timeout=8.0,
    )

    raw_response = response.choices[0].message.content.strip()

    actions, reasoning = _parse_triage_response(raw_response)
    return actions, reasoning


def _parse_triage_response(raw: str) -> Tuple[List[int], str]:
    """Parse triage LLM response into ([4 actions], reasoning)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        text = inner.strip()

    data = json.loads(text)
    actions = list(data.get("actions", [0, 0, 0, 0]))
    reasoning = str(data.get("reasoning", ""))[:100]

    # Ensure exactly 4 actions
    while len(actions) < 4:
        actions.append(0)
    actions = actions[:4]

    # Clamp each action to [0, 2]
    actions = [max(0, min(2, int(a))) for a in actions]

    return actions, reasoning


# ------------------------------------------------------------------ #
#  Baseline rule-based agent — FALLBACK                              #
# ------------------------------------------------------------------ #

def baseline_agent(obs: Dict) -> int:
    """
    Rule-based fallback policy. Used when LLM API fails.

    3 detection strategies:
      1. Delta-based: sudden deviations from personal baseline
      2. Absolute thresholds: catch slow drifts
      3. Trend detection: directional changes in vitals history
    """
    activity = obs["activity"]
    delta    = obs["baseline_delta"]
    hours    = obs["hours_observed"]
    spo2     = obs["spo2"]
    hr       = obs["heart_rate"]
    temp     = obs.get("temperature", 0.4)
    history  = obs.get("vitals_history", [])

    if hours < 1.0:
        return VERIFY

    if activity == 2:
        if spo2 < 0.25:
            return ALERT
        return IGNORE

    # Strategy 1: Delta-based
    if delta > 0.6 and activity == 0:
        return ALERT
    if delta > 0.35 and activity == 0:
        return VERIFY

    # Strategy 2: Absolute thresholds
    if spo2 < 0.35:
        return ALERT
    if spo2 < 0.50:
        return VERIFY
    if temp > 0.80:
        return ALERT
    if temp > 0.65:
        return VERIFY
    if hr > 0.75:
        return ALERT
    if hr > 0.60:
        return VERIFY

    # Strategy 3: Trend detection
    if len(history) >= 6:
        try:
            recent = history[-3:]
            oldest = history[:3]
            recent_spo2 = sum(r[3] for r in recent) / 3
            oldest_spo2 = sum(r[3] for r in oldest) / 3
            spo2_drop = oldest_spo2 - recent_spo2
            recent_temp = sum(r[5] for r in recent) / 3
            oldest_temp = sum(r[5] for r in oldest) / 3
            temp_rise = recent_temp - oldest_temp

            if spo2_drop > 0.08:
                return ALERT
            if spo2_drop > 0.04:
                return VERIFY
            if temp_rise > 0.08:
                return ALERT
            if temp_rise > 0.04:
                return VERIFY
        except (IndexError, TypeError):
            pass

    if hours > 4.0:
        if delta > 0.20 and activity == 0:
            return VERIFY
        if spo2 < 0.55:
            return VERIFY

    return IGNORE


def triage_baseline(obs_list: List[Dict]) -> List[int]:
    """Apply baseline_agent independently to each patient."""
    return [baseline_agent(obs) for obs in obs_list]


# ------------------------------------------------------------------ #
#  OpenEnv validation                                                #
# ------------------------------------------------------------------ #

def openenv_validate() -> bool:
    """Check openenv.yaml exists and has required fields."""
    try:
        import yaml
    except ImportError:
        # pyyaml not installed — try basic check
        if not os.path.exists("openenv.yaml"):
            print("[VALIDATE] fail openenv.yaml not found", flush=True)
            return False
        print("[VALIDATE] pass (yaml module not available, file exists)", flush=True)
        return True

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

    # Check action space
    action_space = spec.get("action_space", {})
    if action_space.get("n") != 3:
        print(f"[VALIDATE] fail action_space.n={action_space.get('n')}, expected 3", flush=True)
        return False

    # Check tasks
    tasks = spec.get("tasks", [])
    task_names = {t.get("name") for t in tasks}
    for required_task in ("suppression", "deterioration", "triage"):
        if required_task not in task_names:
            print(f"[VALIDATE] fail missing task: {required_task}", flush=True)
            return False

    print("[VALIDATE] pass", flush=True)
    return True


# ------------------------------------------------------------------ #
#  Logging helpers                                                   #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str):
    print(f"[START] task={task} env=mediguard model={model}", flush=True)


def log_agent(model: str):
    print(f"[AGENT] type=llm model={model} temperature=0.0", flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    """Print a [STEP] line in the mandatory format."""
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


def log_fallback(step: int, reason: str):
    print(f"[FALLBACK] step={step} reason={reason} using=rule_based", flush=True)


def log_end(success: bool, steps: int, rewards: List[float], score: float):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.4f} "
        f"rewards={rewards_str}",
        flush=True,
    )


def log_reasoning(action, reasoning: str):
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)
    print(f"[REASONING] last_action={action_str} reasoning=\"{reasoning}\"", flush=True)


# ------------------------------------------------------------------ #
#  Main episode runner                                               #
# ------------------------------------------------------------------ #

def run_episode(task: str, seed: int = 42) -> Tuple[List[float], float]:
    """Run a single episode using LLM agent with rule-based fallback."""

    model = MODEL_BY_TASK.get(task, MODEL_NAME)
    log_start(task, model)
    log_agent(model)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    last_action = 0
    last_reasoning = ""
    fallback_count = 0

    # Conversation history — sliding window of last 4 turns
    conversation_history: List[Dict] = []
    # Vitals history — last 8 readings for trend analysis
    vitals_history: List[list] = []

    try:
        env = MediGuardEnv(task=task, seed=seed)
        obs = env.reset()
        done = False

        while not done:
            action = None
            reasoning = ""
            used_fallback = False

            try:
                if task == "triage":
                    # Triage: ONE API call for all 4 patients
                    action, reasoning = triage_llm_agent(
                        obs, conversation_history, model
                    )
                else:
                    # Single patient
                    action, reasoning = llm_agent(
                        obs, task, conversation_history, vitals_history, model
                    )

            except json.JSONDecodeError:
                used_fallback = True
                reasoning = "parse_error"
            except Exception as e:
                used_fallback = True
                err_type = type(e).__name__
                if "timeout" in err_type.lower() or "Timeout" in str(e):
                    reasoning = "timeout"
                elif "rate" in err_type.lower() or "RateLimit" in str(e):
                    reasoning = "ratelimit"
                    time.sleep(2)  # Wait and retry once
                    try:
                        if task == "triage":
                            action, reasoning = triage_llm_agent(
                                obs, conversation_history, model
                            )
                        else:
                            action, reasoning = llm_agent(
                                obs, task, conversation_history, vitals_history, model
                            )
                        used_fallback = False
                    except Exception:
                        used_fallback = True
                        reasoning = "ratelimit_retry_failed"
                else:
                    reasoning = f"error:{err_type}"

            # Fallback to baseline agent
            if used_fallback or action is None:
                if task == "triage":
                    action = triage_baseline(obs)
                else:
                    action = baseline_agent(obs)
                fallback_count += 1
                log_fallback(steps + 1, reasoning)

            # Step the environment
            obs, reward, done, info = env.step(action)
            steps = info["step"]
            rewards.append(reward)
            last_action = action
            last_reasoning = reasoning

            log_step(steps, action, reward, done, error=None)

            # Update histories
            if task == "triage":
                # For triage, store summary obs text
                obs_text = triage_obs_to_message(obs, conversation_history)
            else:
                obs_text = obs_to_user_message(obs, task, vitals_history, conversation_history)

                # Track vitals history for trend analysis (single patient only)
                history = obs.get("vitals_history", [])
                if history:
                    # Get the latest non-zero entry
                    for entry in reversed(history):
                        if any(v != 0.0 for v in entry):
                            vitals_history.append(entry)
                            break
                    # Keep only last 8
                    vitals_history = vitals_history[-8:]

            # Update conversation history (sliding window of 4)
            response_text = json.dumps({"action": action, "reasoning": reasoning})
            conversation_history.append({
                "obs_text": obs_text,
                "response": response_text + f" → reward: {reward:.2f}",
                "action": action,
                "reward": reward,
            })
            if len(conversation_history) > 4:
                conversation_history = conversation_history[-4:]

        # Compute grader score
        grader_map = {
            "suppression": env.false_alarm_rate_grader,
            "deterioration": env.deterioration_grader,
            "triage": env.triage_grader,
        }
        score = grader_map[task]()
        success = score > 0.50

    except Exception as exc:
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        success = False
        score = 0.0

    log_end(success, steps, rewards, score)
    log_reasoning(last_action, last_reasoning)

    if fallback_count > 0:
        print(
            f"[INFO] fallback_count={fallback_count}/{steps} "
            f"({100*fallback_count/max(steps,1):.1f}% of steps used rule-based)",
            flush=True,
        )

    return rewards, score


# ------------------------------------------------------------------ #
#  Entry point                                                       #
# ------------------------------------------------------------------ #

def main():
    # Validate openenv.yaml
    openenv_validate()

    tasks = ["suppression", "deterioration", "triage"]
    all_results = {}

    for task in tasks:
        rewards, score = run_episode(task, seed=42)
        all_results[task] = {"rewards": rewards, "score": score}

    # ── Summary ──
    print(flush=True)
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task in tasks:
        rews = all_results[task]["rewards"]
        score = all_results[task]["score"]
        mean_r = sum(rews) / len(rews) if rews else 0.0
        print(
            f"  {task:15s}  steps={len(rews):4d}  "
            f"mean_reward={mean_r:.4f}  score={score:.4f}",
            flush=True,
        )
    print("=" * 60, flush=True)

    # ── Improvement vs baseline ──
    print(flush=True)
    s = all_results["suppression"]["score"]
    d = all_results["deterioration"]["score"]
    t = all_results["triage"]["score"]

    print(
        f"[SUMMARY] suppression={s:.4f} deterioration={d:.4f} triage={t:.4f}",
        flush=True,
    )
    print(
        f"[IMPROVEMENT] vs_baseline "
        f"suppression={s - BASELINE_SCORES['suppression']:+.4f} "
        f"deterioration={d - BASELINE_SCORES['deterioration']:+.4f} "
        f"triage={t - BASELINE_SCORES['triage']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()






