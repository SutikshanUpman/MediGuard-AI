---
title: MediGuard AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏥 MediGuard-AI

**Context-aware ICU patient monitoring — an OpenEnv-compliant RL environment where agents learn that the same vital sign means something completely different depending on what the patient is doing.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces/SutikshanUpman/MediGuard-AI)

---

## What MediGuard-AI Does

MediGuard-AI is an RL environment that simulates an ICU nurse's triage decision-making. At every timestep, an agent observes a patient's vitals and must choose one of three actions:

| Action | Code | Meaning |
|--------|:----:|---------|
| Ignore | `0` | Vitals look normal for this patient — no action needed |
| Verify | `1` | Something is off — alert the nurse to check |
| Alert  | `2` | Genuine emergency — call the doctor immediately |

**The core insight:** vitals are scored as deviations from *this patient's personal 3-hour rolling baseline*, not population norms. Heart rate 130 bpm while a patient is eating lunch is completely expected. Heart rate 130 bpm while a patient is lying still at rest is an emergency. A naive agent that fires on every spike drowns staff in false alarms. A good agent — like a good nurse — reads context.

This environment rewards agents that:
- Suppress false alarms for patients whose elevated vitals are simply their normal
- Detect slow deterioration trends before they become crises
- Correctly rank multiple patients by urgency when attention is limited

Activity context is tracked via pose/motion signals (compatible with MediaPipe-style activity detection), enabling realistic discounting of vitals during expected-high-activity states like walking or eating.

---

## Why OpenEnv

OpenEnv provides a standardised interface (`reset()` → `step()` → `state()`) that makes RL environments composable, reproducible, and directly comparable across agents. By building on OpenEnv, MediGuard-AI can be evaluated by any compliant agent — from simple rule-based baselines to frontier LLMs like Nemotron — without any environment-side changes. The spec also enforces typed observation and action schemas via Pydantic, making integration errors surface at import time rather than mid-episode.

---

## Three Tasks — Easy to Hard

### Task 1 — Suppression (Easy)

**Scenario:** A chronically hypertensive patient with a personal baseline of ~150/95 mmHg. Their BP reads high all the time — that is simply who they are. The agent must learn to suppress false alarms triggered by this patient's naturally elevated vitals, while still responding to a genuine acute event (a tachycardic crisis injected between steps 30–55).

**Grader:** F1 score — harmonic mean of sensitivity (catching real emergencies) and specificity (suppressing false alarms). ALERT scores full credit on emergencies; VERIFY scores partial. Spamming ALERT on stable vitals is penalised via false positives.

**Why this is hard for rule-based agents:** Any threshold-based agent that fires on high BP will generate constant false alarms. Only an agent that learns this patient's personal baseline can achieve high specificity without sacrificing sensitivity.

**Score range:** Rule-based baseline ~0.55–0.65 · LLM target ~0.75+

---

### Task 2 — Deterioration (Medium)

**Scenario:** A patient developing sepsis over 60 simulated timesteps. Deterioration begins at step 30 — SpO2 drifts down, temperature climbs, BP gradually falls. The agent must detect the trend early and escalate before the patient reaches crisis.

**Grader:** Onset-delay scoring: `score = 0.4 + 0.6 × (1 − delay/30)`. Earlier detection scores higher. Missing the deterioration entirely scores 0. A single false ALERT before deterioration starts incurs a false alarm cap (threshold: 8% of steps).

**Why this is hard for rule-based agents:** Single-step thresholds miss gradual trends. Only agents reading the vitals history table can reliably catch the drift before it becomes obvious.

**Score range:** Rule-based baseline ~0.30–0.50 · LLM target ~0.55+

---

### Task 3 — Triage (Hard)

**Scenario:** Four concurrent patients — a healthy patient, a post-operative patient, a deteriorating sepsis patient, and another healthy patient. The agent receives all four observations simultaneously and must allocate limited attention correctly.

**Grader:** Composite score:
- **NDCG@4 (50%)** — measures whether the agent correctly ranks patients by urgency
- **ALERT-F1 (30%)** — precision and recall on committing ALERT (not just VERIFY) to emergency patients
- **Responsiveness (20%)** — how quickly the agent escalates after a patient's condition worsens
- **Concentration penalty** — agents that apply the same action to every patient (e.g. VERIFY everyone) are penalised for failing to differentiate
- **Hesitation penalty** — VERIFY on an EMERGENCY patient wastes critical time

**Why this is the hardest task:** Getting the ordering right requires comparing patients against each other, not just evaluating each independently. A VERIFY-everything agent scores ~0.22. An agent that correctly identifies the deteriorating patient and escalates early scores ~0.65+.

**Score range:** Rule-based baseline ~0.22–0.28 · LLM target ~0.65+

---

## Why 60 Steps, Not 360

The original episode length was 360 steps (simulating 6 real hours at 1-minute resolution). This was reduced to 60 steps for three reasons:

1. **Eval time budget:** The hackathon validator enforces a 20-minute runtime cap for `inference.py`. At 60 steps × 3 tasks × ~4s per LLM call, the worst-case runtime is ~12 minutes — safely within budget. At 360 steps, the same LLM call overhead would run ~72 minutes, a guaranteed timeout.

2. **Signal density:** The key clinical events (hypertensive crisis, sepsis onset, triage urgency ranking) all occur within the first 60 steps at the adjusted simulation rate. Extending to 360 only adds stable baseline steps that add noise without clinical information.

3. **Reproducibility:** Shorter episodes are faster to verify, easier to debug, and produce tighter score variance across seeds.

The `hours_observed` field in observations uses `step / 10.0` so the agent's internal sense of elapsed time is calibrated correctly.

---

## Environment Spec

**Observation Space** — 10 fields per patient:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `heart_rate` | float | 0–1 | Normalized HR (30–200 bpm) |
| `systolic_bp` | float | 0–1 | Normalized systolic BP (60–220 mmHg) |
| `diastolic_bp` | float | 0–1 | Normalized diastolic BP (30–140 mmHg) |
| `spo2` | float | 0–1 | Normalized SpO2 (70–100%) |
| `respiratory_rate` | float | 0–1 | Normalized RR (5–40 /min) |
| `temperature` | float | 0–1 | Normalized temp (34–42°C) |
| `baseline_delta` | float | 0–1 | Combined deviation from this patient's personal 3-hour rolling baseline |
| `hours_observed` | float | ≥0 | Elapsed time (step / 10.0) |
| `activity` | int | 0–4 | Current activity: 0=resting, 1=eating, 2=walking, 3=distressed, 4=falling |
| `vitals_history` | list | [10][6] | Last 10 timesteps of all 6 normalized vitals |

The `activity` field is compatible with MediaPipe Pose-based activity detection — in a real deployment, a camera feed processed through MediaPipe's pose landmark model can classify patient activity (ambulating, lying still, distressed posture) and feed directly into this observation field, enabling the environment to run on live ICU data without any interface changes.

**Action Space:** `Discrete(3)` — 0=Ignore, 1=Verify, 2=Alert
**Episode Length:** 60 steps
**Reward:** Normalized to [0.0, 1.0]; shaped by activity-context multipliers, alarm fatigue modifier, and personalization bonus
**Seed:** 42 (fully reproducible)

---

## Reward Design

```
Base reward table (action × patient condition):

              | STABLE | BORDERLINE | EMERGENCY | DRUG_MASKED |
ALERT         |  -0.5  |    +0.2    |   +1.0    |    +1.0     |
VERIFY        |  -0.1  |    +0.7    |   +0.3    |    +0.3     |
IGNORE        |  +0.2  |    -0.2    |   -1.0    |    -1.0     |

Activity context multipliers (same vitals, different clinical meaning):
  Resting    → 1.00x  (full weight — anomaly is unexplained)
  Eating     → 0.40x  (slight HR/BP rise expected)
  Walking    → 0.50x  (elevated vitals expected during ambulation)
  Distressed → 1.25x  (amplify — distress compounds risk)
  Falling    → 1.60x  (immediate danger, maximum weight)

Additional modifiers:
  Alarm fatigue:   >5 alerts in last 30 steps → 0.6× reward multiplier
  Personalization: Correctly ignoring a stable patient after step 20 → +0.2 bonus
```

---

## Project Structure

```
MediGuard-AI/
├── app.py / server/app.py  # FastAPI (REST endpoints) + Gradio (interactive UI)
├── inference.py            # LLM agent with rule-based fallback + structured logging
├── mediguard_env.py        # OpenEnv-compliant RL environment (core)
├── patient_simulator.py    # Vital sign generator (5 patient types, seeded)
├── reward_function.py      # Stateful reward calculator with all modifiers
├── task1_suppression.py    # Grader: F1 (sensitivity + specificity)
├── task2_deterioration.py  # Grader: onset-delay scoring
├── task3_triage.py         # Grader: NDCG@4 + F1 + responsiveness + penalties
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container config
├── requirements.txt        # Python dependencies
└── validation.sh           # Pre-submission validation script
```

---

## Quick Start

```bash
git clone https://github.com/SutikshanUpman/MediGuard-AI
cd MediGuard-AI
pip install -r requirements.txt

# Launch the interactive UI + REST API
python server/app.py

# Run inference across all 3 tasks
python inference.py
```

**Docker:**
```bash
docker build -t mediguard-ai .
docker run -p 7860:7860 \
  -e API_KEY=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  mediguard-ai
```

**Use the environment directly:**
```python
from mediguard_env import MediGuardEnv

# Task 1 — single hypertensive patient
env = MediGuardEnv(task="suppression", seed=42)
obs = env.reset()
while True:
    obs, reward, done, info = env.step(1)  # Verify
    if done:
        break
print(f"Score: {env.false_alarm_rate_grader():.4f}")

# Task 3 — four concurrent patients
env = MediGuardEnv(task="triage", seed=42)
obs = env.reset()  # list of 4 observation dicts
obs, reward, done, info = env.step([2, 0, 1, 0])
print(f"Score: {env.triage_grader():.4f}")
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `API_KEY` | Yes (eval) | — | Injected by validator for LLM calls |
| `HF_TOKEN` | Yes (self-host) | — | Your HuggingFace token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |

---

## Inference Log Format

```
[VALIDATE] pass
[CONFIG] api_key_source=API_KEY base_url=... model=...
[LLM_CHECK] has_key=True client_ready=true
[START] task=suppression env=mediguard model=Qwen/Qwen2.5-72B-Instruct
[AGENT] type=llm model=Qwen/Qwen2.5-72B-Instruct temperature=0.0
[STEP] step=1 action=0 reward=0.63 done=false error=null
...
[END] success=true steps=60 score=0.7234 rewards=0.63,0.55,...
[REASONING] last_action=0 reasoning="SpO2 stable, patient eating"
[SUMMARY] suppression=0.7234 deterioration=0.6102 triage=0.5381
[IMPROVEMENT] vs_baseline suppression=+0.07 deterioration=+0.22 triage=+0.23
```

---

## Authors

**Sutikshan Upman** · **Rajveer Singh**

Built for the Meta PyTorch × Scaler OpenEnv Hackathon 2026.

*MediGuard-AI — because missing a deteriorating patient is not an option.*
