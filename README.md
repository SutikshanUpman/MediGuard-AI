---
title: MediGuard-AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# 🏥 MediGuard-AI

**An OpenEnv-compliant RL environment for ICU patient monitoring — built for the Meta PyTorch OpenEnv Hackathon 2026.**

[![Deployed on HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/openenv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

---

## 🧠 What is MediGuard-AI?

An RL environment where an AI agent monitors ICU patients and must decide — every minute — whether to **Ignore**, **Verify**, or **Alert**. The twist: the same vital sign reading means completely different things depending on what the patient is doing.

| Situation | HR 130 bpm | Correct Action |
|-----------|-----------|----------------|
| 🍽️ Patient is eating | Expected | 😴 IGNORE |
| 🚶 Patient is walking | Possible | 🔍 VERIFY |
| 🛏️ Patient is resting | DANGER | 🚨 ALERT |

The environment uses **activity-gated rewards** — the novel mechanic that makes this more than a threshold-checking problem. A naive agent that panics at every spike gets penalized. A smart agent that understands context scores high.

---

## 🎯 Three Tasks (Easy → Hard)

### Task 1 — Suppression (Easy)
A chronically hypertensive patient with baseline BP ~150/95 mmHg. The agent must learn this is *normal for them* and suppress false alarms.

**Grader:** F1 harmonic mean of sensitivity × specificity. Pure-IGNORE → 0.0, Pure-ALERT → 0.0. Only a balanced agent scores high.

### Task 2 — Deterioration (Medium)
A patient slowly developing sepsis over 6 simulated hours. Temperature rises, BP drops, SpO2 falls — all gradually. The agent must catch the trend early.

**Grader:** Onset-delay scoring — `score = 0.3 + 0.7 × max(0, 1 − delay/20)`. Early detection is rewarded exponentially more. VERIFY alone doesn't count — must escalate to ALERT.

### Task 3 — Triage (Hard)
Four concurrent patients: healthy, post-op, deteriorating, healthy. The agent must rank them by urgency and allocate limited attention.

**Grader:** NDCG@4 (50%) + ALERT-F1 (30%) + Responsiveness (20%) − concentration penalty − hesitation penalty. Measures priority *ordering* quality, not just per-patient accuracy.

---

## 🏗️ Architecture

```
mediguard-ai/
├── patient_simulator.py    # Realistic vital sign generator (5 patient types)
├── reward_function.py      # Action × Condition × Activity → reward (with fatigue + personalization)
├── mediguard_env.py        # OpenEnv-compliant RL environment (core)
├── inference.py            # LLM agent (Qwen 7B/72B) + rule-based fallback + structured logging
├── app.py                  # Gradio UI: interactive demo + full inference pipeline + API endpoints
├── task1_suppression.py    # Grader: F1 harmonic mean of sensitivity & specificity
├── task2_deterioration.py  # Grader: onset-delay scoring with false alarm penalty
├── task3_triage.py         # Grader: NDCG@4 + ALERT-F1 + responsiveness − penalties
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container config for HF Spaces
└── requirements.txt        # Python dependencies (numpy, pydantic, openai, gradio, pyyaml)
```

---

## 📋 File-by-File Details

### `patient_simulator.py`
Generates realistic ICU vital signs for 5 patient types: healthy, hypertensive, post-op, deteriorating, and custom. Simulates activity cycles (resting → eating → walking → distressed → falling) and sepsis-like drift with configurable severity. Pure data generator — no decision logic.

### `reward_function.py`
Maps `(action × condition × activity)` → reward signal. Key features:
- **Activity context multipliers** — the core mechanic. Penalties are discounted during high-activity states (walking ×0.50, eating ×0.40) and amplified during dangerous activities (distressed ×1.25, falling ×1.60). Resting is baseline ×1.0.
- **Alarm fatigue modifier** — >5 alerts in last 30 steps → reward reduced to 0.6×
- **Personalization bonus** — correctly ignoring a stable patient after step 200 → +0.2 bonus
- Tracks `action_history`, `condition_history`, and `activity_history` for grader use

**Updated reward table:**
```
             | STABLE | BORDERLINE | EMERGENCY | DRUG_MASKED |
ALERT        |  -0.5  |    +0.2    |   +1.0    |    +1.0     |
VERIFY       |  -0.1  |    +0.7    |   +0.3    |    +0.3     |
IGNORE       |  +0.2  |    -0.2    |   -1.0    |    -1.0     |
```

### `mediguard_env.py`
OpenEnv-compliant environment with `reset() → obs`, `step(action) → (obs, reward, done, info)`, `state() → dict`. Contains:
- **3 Pydantic models**: `ObservationModel`, `ActionModel`, `RewardModel` (required by OpenEnv spec)
- **`_PatientTracker`** — per-patient rolling baseline and vitals history
- **`_classify_condition()`** — maps continuous vitals to discrete `PatientCondition` enum using deterioration severity + vital thresholds
- **Reward normalization** — raw range `[-1.6, 1.6]` mapped to `[0.0, 1.0]`
- **Grader delegation** — `false_alarm_rate_grader()`, `deterioration_grader()`, `triage_grader()` call external grader modules

### `inference.py`
LLM-based agent using OpenAI-compatible API (HuggingFace router). Key features:
- **Per-task model selection**: Suppression → `Qwen/Qwen2.5-7B-Instruct` (cheap), Deterioration/Triage → `Qwen/Qwen2.5-72B-Instruct` (powerful)
- **Task-specific system prompts** — tuned for each grader's scoring criteria
- **Vitals trend table** — last 8 readings formatted for deterioration detection
- **Conversation history** — sliding window of 4 turns with reward feedback for pseudo-online-learning
- **Triage: single API call** — all 4 patients in one prompt, output `{"actions": [2,0,1,0]}`
- **Graceful fallback** — falls back to rule-based `baseline_agent()` on any API error (timeout, rate limit, parse error)
- **Structured logging** — `[START]`, `[STEP]`, `[END]`, `[FALLBACK]`, `[REASONING]`, `[SUMMARY]` lines

**Environment variables:**
```
HF_TOKEN / OPENAI_API_KEY / API_KEY   — authentication (checked in priority order)
API_BASE_URL                           — LLM endpoint (default: HuggingFace router)
MODEL_NAME                             — model ID (default: Qwen/Qwen2.5-72B-Instruct)
```

### `app.py`
Gradio-based UI for HuggingFace Spaces with:
- **Interactive demo tab** — reset env, pick agent mode (LLM / Rule-Based / Manual), step through episodes, live vitals monitor with risk tags
- **How It Works tab** — visual explanation of the insight, scoring formulas, LLM architecture
- **3 agent modes**: LLM Agent (real API calls), Rule-Based (baseline_agent), Manual (user picks actions)
- **Full inference pipeline** — runs `python inference.py` end-to-end and streams output
- **API endpoints** — `reset_env`, `step_env`, `get_state`, `health_check` exposed via Gradio API
- Dark vitals display with raw+normalized values

### `task1_suppression.py`
**F1 harmonic mean** of sensitivity and specificity:
- ALERT on emergency = full true positive
- VERIFY on emergency = 0.5 TP (must commit to ALERT for full credit)
- VERIFY on stable = 0.7 FP (spamming VERIFY is penalized)
- Pure-IGNORE → sensitivity=0 → F1=**0.0** (can't cheat)
- Expected scores: rule-based ~0.50, LLM ~0.80

### `task2_deterioration.py`
**Onset-delay scoring** with strict escalation requirement:
- Finds deterioration episodes (≥5 consecutive non-STABLE steps)
- Score per episode: `0.3 + 0.7 × max(0, 1 − delay/20)`
- VERIFY alone does NOT count — must escalate to ALERT
- False alarm penalty kicks in at 8% FAR, capped at 0.40
- Expected scores: rule-based ~0.39, LLM ~0.75

### `task3_triage.py`
**NDCG@4 + F1 + Responsiveness − penalties:**
- NDCG@4 (50%) — measures priority ordering quality across 4 patients
- ALERT-F1 (30%) — only ALERT counts as true positive for emergencies
- Responsiveness (20%) — how quickly agent adapts to condition changes
- Concentration penalty — spamming same action to all patients is penalized
- Hesitation penalty — VERIFY on EMERGENCY patients is explicitly penalized
- Expected scores: rule-based ~0.22, LLM ~0.65

### `openenv.yaml`
OpenEnv spec metadata pointing to:
- `env_module: mediguard_env`, `env_class: MediGuardEnv`
- Pydantic models: `ObservationModel`, `ActionModel`, `RewardModel`
- `inference_script: inference.py`, `app: app.py`
- 3 tasks with grader references matching env methods

### `Dockerfile`
`python:3.11-slim` based container. Copies all source files and launches `app.py` on port 7860.

### `requirements.txt`
Minimal dependencies — only what the code actually imports:
```
numpy       — vital sign simulation + normalization
pydantic    — OpenEnv schema validation (ObservationModel, ActionModel, RewardModel)
openai      — LLM client (HuggingFace router compatible)
gradio      — UI + API server
pyyaml      — openenv.yaml validation in inference.py
```

---

## 🔬 Environment Spec

**Observation Space** — 10 fields per patient:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `heart_rate` | float | 0–1 | Normalized HR (30–200 bpm) |
| `systolic_bp` | float | 0–1 | Normalized systolic BP (60–220 mmHg) |
| `diastolic_bp` | float | 0–1 | Normalized diastolic BP (30–140 mmHg) |
| `spo2` | float | 0–1 | Normalized SpO2 (70–100%) |
| `respiratory_rate` | float | 0–1 | Normalized RR (5–40 bpm) |
| `temperature` | float | 0–1 | Normalized temp (34–42°C) |
| `baseline_delta` | float | 0–1 | Rolling deviation from personal baseline |
| `hours_observed` | float | ≥0 | Elapsed time (step / 60) |
| `activity` | int | 0–4 | 0=resting, 1=eating, 2=ambulating, 3=distressed, 4=falling |
| `vitals_history` | list | [10][6] | Last 10 timesteps of normalized vitals |

**Action Space:** `Discrete(3)` — 0=Ignore, 1=Verify, 2=Alert
**Episode Length:** 360 steps (6 simulated hours)
**Reward:** Normalized to [0.0, 1.0]; shaped by activity context + alarm fatigue + personalization
**Seed:** 42 (reproducible)

---

## 🚀 Quick Start

### Run locally

```bash
git clone <your-repo>
cd mediguard-ai
pip install -r requirements.txt

# Set your HuggingFace token for LLM agent
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Launch Gradio UI
python app.py

# Run LLM inference across all 3 tasks
python inference.py
```

### Use the environment in Python

```python
from mediguard_env import MediGuardEnv

# Single patient task
env = MediGuardEnv(task="suppression", seed=42)
obs = env.reset()

while True:
    action = 1  # Verify
    obs, reward, done, info = env.step(action)
    if done:
        break

score = env.false_alarm_rate_grader()
print(f"Suppression score: {score:.4f}")

# Triage (4 patients)
env = MediGuardEnv(task="triage", seed=42)
obs = env.reset()  # returns list of 4 observation dicts

obs, reward, done, info = env.step([1, 0, 2, 0])  # per-patient actions
score = env.triage_grader()
```

### API Endpoints (Gradio)

| Endpoint | Input | Description |
|----------|-------|-------------|
| `/api/reset_env` | `task` (str), `seed` (int) | Reset environment |
| `/api/step_env` | `action` (str or `"1,0,2,0"`) | Take one step |
| `/api/get_state` | — | Current environment state |
| `/api/health_check` | — | Liveness check |

---

## 📊 Expected Scores

| Task | Random Agent | Rule-Based | LLM Agent | Perfect |
|------|:-----------:|:----------:|:---------:|:-------:|
| Suppression | ~0.33 | ~0.50 | ~0.80 | 1.0 |
| Deterioration | ~0.33 | ~0.39 | ~0.75 | 1.0 |
| Triage | ~0.33 | ~0.22 | ~0.65 | 1.0 |

These are honest scores — the graders are specifically designed so that:
1. Pure-IGNORE and Pure-ALERT both score ~0.0 (no cheating)
2. Rule-based agents score moderately (can't game the graders)
3. LLM agents show clear improvement (activity context reasoning)
4. Trained RL agents have room to improve further

---

## 🐳 Docker / HF Spaces

The app auto-detects its environment:
- HuggingFace Spaces or Docker → binds to `0.0.0.0:7860`
- Local development → binds to `127.0.0.1:7860`

```dockerfile
FROM python:3.11-slim
# Copies all source files and launches app.py on port 7860
```

---

## 🏆 Hackathon Context

Built for the **Meta PyTorch OpenEnv Hackathon 2026** (India), organized by Scaler School of Technology in collaboration with Meta, Hugging Face, and PyTorch.

- **Theme:** Real RL infrastructure, not just demos
- **Framework:** Meta's OpenEnv
- **Finale:** 48-hour in-person hackathon, Bangalore, April 25–26 2026
- **Prize pool:** $30,000 + interview opportunities at Meta & Hugging Face

---

*MediGuard-AI — because missing a deteriorating patient is not an option.*