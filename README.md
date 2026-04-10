---
title: MediGuard AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏥 MediGuard-AI

**An OpenEnv-compliant RL environment for ICU patient monitoring — built for the Meta PyTorch OpenEnv Hackathon 2026.**

[![Deployed on HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/openenv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

---

## 🧠 What is MediGuard-AI?

Imagine you're a nurse watching ICU monitors for multiple patients at once. Every minute you see heart rate, blood pressure, oxygen levels, temperature — and you have three choices:

| Action | Meaning |
|--------|---------|
| **0 — Ignore** | Everything looks fine, carry on |
| **1 — Verify** | Something seems off, investigate |
| **2 — Alert** | This is serious — get a doctor now |

The challenge is that some patients naturally have high blood pressure (it's their normal). Some are walking around (elevated heart rate is expected). A naive AI panics at every spike. MediGuard-AI trains an RL agent to learn each patient's **personal baseline** and only respond when something is genuinely wrong.

---

## 🎯 Three Tasks (Easy → Hard)

### Task 1 — Suppression (Easy)
A chronically hypertensive patient with baseline BP ~150/95 mmHg. The agent must learn this is *normal for them* and suppress false alarms. Graded on false alarm rate: score = 1.0 when FAR < 5%.

### Task 2 — Deterioration (Medium)
A patient slowly developing sepsis over 6 simulated hours. Temperature rises, BP drops, SpO2 falls — all gradually. The agent must catch the trend early and alert before it becomes critical. Three-phase scoring rewards early detection.

### Task 3 — Triage (Hard)
Four concurrent patients: healthy, post-op, deteriorating, healthy. The agent must identify who genuinely needs care vs. who is stable. Graded on F1 score (50%) + masked detection (30%) + triage priority (20%).

---

## 🏗️ Architecture

```
mediguard-ai/
├── patient_simulator.py    # Realistic vital sign generator (5 patient types)
├── reward_function.py      # RewardFunction: action × condition → reward
├── mediguard_env.py        # OpenEnv-compliant RL environment (core)
├── inference.py            # Baseline rule-based agent + structured logging
├── app.py                  # Gradio UI + API endpoints (HF Spaces entry point)
├── task1_suppression.py    # Grader: false alarm rate
├── task2_deterioration.py  # Grader: detection timing (3-phase)
├── task3_triage.py         # Grader: F1 + masked detection + priority
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container config for HF Spaces
└── requirements.txt        # Python dependencies
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
| `hours_observed` | float | ≥0 | Elapsed time (step / 10.0) |
| `activity` | int | 0–4 | 0=resting, 1=eating, 2=ambulating, 3=distressed, 4=falling |
| `vitals_history` | list | [10][6] | Last 10 timesteps of normalized vitals |

**Action Space:** `Discrete(3)` — 0=Ignore, 1=Verify, 2=Alert  
**Episode Length:** 60 steps (6 simulated hours)  
**Reward:** Normalized to [0.0, 1.0]; shaped by alarm fatigue modifier + personalization bonus  
**Seed:** 42 (reproducible)

---

## 🚀 Quick Start

### Run locally

```bash
git clone <your-repo>
cd mediguard-ai
pip install -r requirements.txt

# Launch Gradio UI
python app.py

# Run baseline inference across all 3 tasks
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

## 🧪 Reward Design

```
Base reward table (action × patient condition):

             | STABLE | BORDERLINE | EMERGENCY | DRUG_MASKED |
ALERT        |  -0.5  |    +0.2    |   +1.0    |    +1.0     |
VERIFY       |  +0.1  |    +0.4    |   -0.8    |    -0.5     |
IGNORE       |  +0.2  |    -0.1    |   -2.0    |    -2.0     |

Modifiers:
  - Alarm fatigue:     >5 alerts in last 30 steps → 0.6× multiplier
  - Personalization:   Correctly ignoring stable patient after step 200 → +0.2 bonus
  - Normalization:     Raw [-2.0, 1.2] mapped to [0.0, 1.0]
```

---

## 📊 Baseline Agent Performance

The included rule-based agent (`inference.py`) uses three detection strategies:

1. **Delta-based** — reacts to sudden deviations from rolling personal baseline
2. **Absolute thresholds** — catches slow drifts (SpO2, temp, HR out-of-range)
3. **Trend detection** — compares recent vs. oldest vitals in history window

| Task | Strategy | Typical Mean Reward |
|------|----------|-------------------|
| Suppression | Delta + absolute | ~0.65–0.75 |
| Deterioration | Trend + absolute | ~0.55–0.65 |
| Triage | Per-patient baseline | ~0.50–0.60 |

---

## 🐳 Docker / HF Spaces

The app auto-detects its environment:
- HuggingFace Spaces or Docker → binds to `0.0.0.0:7860`
- Local development → binds to `127.0.0.1:7860`

```dockerfile
# Dockerfile uses python:3.11-slim
# Installs requirements and launches app.py
```

---

## 🔧 Development Workflow

```bash
# Work on feature branch
git checkout sutikshan

# Edit → commit → push to HF
git add .
git commit -m "your message"
git push hf sutikshan:main --force

# Sync from team main branch
git pull origin main
```

> ⚠️ Edit README locally only — avoid simultaneous edits in HF UI + local. The YAML front-matter block **must** remain at the very top of this file for HF Spaces to parse it correctly.

---

## 📦 Dependencies

```
numpy       — vital sign simulation + normalization
pydantic    — OpenEnv observation/action schema validation
openai      — LLM client (ready for agent swap-in)
gradio      — UI + API server
audioop-lts — Python 3.13 audio compatibility for Gradio
setuptools  — build tooling
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