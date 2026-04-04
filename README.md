# 🏥 MediGuard-AI

**An AI-powered ICU patient monitoring system built for the Meta PyTorch OpenEnv Hackathon 2026.**

MediGuard-AI uses reinforcement learning to teach an AI agent when to raise alarms about ICU patients — learning to catch real emergencies while ignoring false alarms.

---

## 🧠 The Simple Explanation

Imagine you're a nurse watching heart monitors for 4 ICU patients at once. Every minute, you see their heart rate, blood pressure, oxygen levels, etc. You have three choices:

| Action | What it means |
|--------|--------------|
| **Ignore** | "Everything looks fine, carry on." |
| **Verify** | "Something seems off, let me take a closer look." |
| **Alert** | "This is serious — get a doctor NOW!" |

The challenge? Some patients naturally have high blood pressure (it's their normal). Some are walking around (so of course their heart rate is up). A bad AI would panic at every spike. A good AI learns each patient's **personal baseline** and only alerts when something is *genuinely* wrong.

**MediGuard-AI trains an RL agent to make these decisions smartly across 3 scenarios:**

1. **Suppression** (Easy) — A patient with chronically high blood pressure. The AI must learn that 150/95 is *normal for them* and stop crying wolf.
2. **Deterioration** (Medium) — A patient slowly developing sepsis over 6 hours. Vitals drift so gradually a human might miss it. The AI must catch it early.
3. **Triage** (Hard) — 4 patients at once. The AI must figure out who actually needs attention and who is fine.

---

## 📁 Project Structure

```
RL Hackathon/
├── patient_simulator.py     ← Generates realistic patient vital signs
├── reward_function.py       ← RewardFunction class (action × condition → reward)
├── task1_suppression.py     ← Easy task grader (false alarm rate)
├── task2_deterioration.py   ← Medium task grader (detection timing)
├── task3_triage.py          ← Hard task grader (F1 + masked detection + priority)
├── mediguard_env.py         ← RL environment (the "game" the AI plays)
├── inference.py             ← Baseline inference script with structured logs
├── server.py                ← Gradio server for HF Space deployment
├── openenv.yaml             ← OpenEnv spec metadata
├── Dockerfile               ← Container build for HF Spaces
├── requirements.txt         ← Python dependencies
├── walkthrough.md           ← Detailed layman-friendly project walkthrough
└── README.md                ← You are here
```

---

## 📄 File 1: `patient_simulator.py`

### In Plain English
This is a **virtual patient**. It generates realistic vital signs (heart rate, blood pressure, oxygen, etc.) that change over time — just like a real ICU patient. It simulates different patient types: healthy, high blood pressure, slowly getting sicker, post-surgery recovery, and unstable.

### Technical Details

| Component | Description |
|-----------|-------------|
| **Class** | `PatientSimulator` |
| **Input** | `patient_type` (str), `seed` (int) |
| **Output** | Vital signs dict with 6 keys |

**Key methods:**

```python
sim = PatientSimulator(patient_type="healthy", seed=42)

sim.get_vitals()    # → dict with heart_rate, systolic_bp, diastolic_bp, spo2, respiratory_rate, temperature
sim.get_activity()  # → int: 0=resting, 1=eating, 2=ambulating, 3=distressed, 4=falling
sim.tick()          # Advances time by 1 step, updates all vitals
sim.get_state()     # → full debug state dict
sim.reset()         # Resets to initial conditions
```

**Patient types and their clinical profiles:**

| Type | Baseline BP | Clinical Scenario |
|------|------------|-------------------|
| `healthy` | 120/80 | Normal vitals, small random noise |
| `hypertensive` | 150/95 | Chronically elevated BP (their normal) |
| `deteriorating` | 118/78 → declining | Slow sepsis drift: temp↑ HR↑ BP↓ SpO2↓ over 360 steps |
| `post_op` | 100/65 | Low BP, recovering from surgery |
| `unstable` | 125/82 | Random spikes and drops (10% chance per step) |

**How vitals are generated each tick:**
1. Fresh baseline values + Gaussian noise
2. Activity effects applied (walking raises HR, etc.)
3. Deterioration drift applied (for deteriorating/unstable types)
4. Smoothing: 70% previous + 30% new (prevents unrealistic jumps)
5. Clipped to physiological limits (HR: 30–200, SpO2: 70–100, etc.)

---

## 📄 File 2: `mediguard_env.py`

### In Plain English
This is the **game board**. It wraps the patient simulator into a standard RL environment. Every "turn," the AI agent sees the patient's vitals and decides: Ignore, Verify, or Alert. The environment scores the decision and moves time forward. After 360 turns (representing 6 hours), the episode ends.

It also tracks the patient's **personal baseline** — a running average of their vitals — so the AI can tell "this is unusual *for this specific patient*" rather than just "this number is high."

### Technical Details

**Class:** `MediGuardEnv` — OpenEnv-compliant with 3 public methods.

```python
env = MediGuardEnv(task="suppression", seed=42)

obs = env.reset()                          # → observation dict
obs, reward, done, info = env.step(action) # → (obs, reward, done, info)
state = env.state()                        # → debug state dict
```

**Pydantic Models (OpenEnv spec):**
- `ObservationModel` — validates observation dict schema
- `ActionModel` — validates action schema (int or List[int])

**Observation Space — 10 fields per patient:**

| Key | Type | Description |
|-----|------|-------------|
| `heart_rate` | float [0,1] | Normalized HR. Raw range: 30–200 bpm |
| `systolic_bp` | float [0,1] | Normalized systolic BP. Raw: 60–220 mmHg |
| `diastolic_bp` | float [0,1] | Normalized diastolic BP. Raw: 30–140 mmHg |
| `spo2` | float [0,1] | Normalized oxygen saturation. Raw: 70–100% |
| `respiratory_rate` | float [0,1] | Normalized resp rate. Raw: 5–40 breaths/min |
| `temperature` | float [0,1] | Normalized temp. Raw: 34–42°C |
| `baseline_delta` | float [0,1] | Mean abs deviation from personal rolling baseline |
| `hours_observed` | float | `step / 60.0` — how long we've been watching |
| `activity` | int {0–4} | What the patient is doing right now |
| `vitals_history` | list [10][6] | Last 10 timesteps of normalized vitals (zero-padded) |

**Normalization formula:**
```
normalized = clip((raw - min) / (max - min), 0, 1)
```

**Action Space:** `Discrete(3)` — `0`=Ignore, `1`=Verify, `2`=Alert

**Task configurations:**

| Task | # Patients | Patient Type(s) | Action Shape | Obs Shape |
|------|-----------|-----------------|-------------|-----------|
| `suppression` | 1 | hypertensive | `int` | `dict` |
| `deterioration` | 1 | deteriorating | `int` | `dict` |
| `triage` | 4 | healthy, post_op, deteriorating, healthy | `List[int]` len 4 | `List[dict]` len 4 |

**Episode length:** 360 steps (done=True when `step >= 360`)

---

## 📄 File 3: `inference.py`

### In Plain English
This is the **test run**. It plays the game using a rule-based strategy with 3 detection strategies. It runs all 3 tasks, and for each one, it prints detailed logs in the strict format the hackathon's automated scoring system reads.

### Technical Details

**Baseline agent — 3 detection strategies:**

| Strategy | What it catches | How |
|----------|---------------|-----|
| **Delta-based** | Sudden changes | `baseline_delta > 0.6` → ALERT, `> 0.35` → VERIFY |
| **Absolute thresholds** | Slow drift | `spo2 < 0.35` → ALERT, `temp > 0.75` → ALERT |
| **Trend detection** | Directional change | Compare recent 3 vs oldest 3 vitals in history |

**Mandatory log format:**
```
[START] task=suppression env=mediguard model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=1 reward=0.66 done=false error=null
...
[END] success=true steps=360 score=1.00 rewards=0.66,0.69,...
```

---

## 📄 File 4: `server.py`

### In Plain English
This turns the environment into a **web app** that can run on HuggingFace Spaces. It provides both an interactive UI and API endpoints.

### Technical Details
Gradio app with interactive buttons and API endpoints:

| Function | What it does |
|----------|-------------|
| Reset button | Reset env with selected task and seed |
| Step button | Take action, see observation and reward |
| `/api/reset_env` | API: reset environment |
| `/api/step_env` | API: take a step |
| `/api/get_state` | API: get current state |
| `/api/health_check` | API: health check |

---

## 📄 File 5: `reward_function.py`

### In Plain English
This tells the AI "good job" or "bad job" after every decision, based on a reward table that maps each action to each patient condition.

### Technical Details

**Base reward table:**

| | Emergency 🚨 | Borderline ⚠️ | Stable ✅ | Drug-Masked 💊 |
|---|---|---|---|---|
| **Alert** | +1.0 | +0.2 | -0.5 | +1.0 |
| **Verify** | -0.8 | +0.4 | +0.1 | -0.5 |
| **Ignore** | -2.0 | -0.1 | +0.2 | -2.0 |

**Modifiers:**
- **Alarm fatigue**: >5 alerts in last 30 steps → reward × 0.6
- **Personalization bonus**: After 200 steps, IGNORE + STABLE → +0.2 bonus

---

## 📄 Files 6-8: Task Graders

| Grader | File | Scoring Method |
|--------|------|---------------|
| `false_alarm_rate_grader` | `task1_suppression.py` | False alarm rate: <5% → 1.0, >60% → 0.0 |
| `deterioration_grader` | `task2_deterioration.py` | Two-phase: VERIFY early (0.3) + ALERT on time (0.7) |
| `triage_grader` | `task3_triage.py` | 50% F1 + 30% masked detection + 20% priority |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke-test the environment (5 steps per task)

```bash
python mediguard_env.py
```

### 3. Run the full baseline inference (360 steps × 3 tasks)

```bash
python inference.py
```

### 4. Run the server locally

```bash
python server.py
```

Then open: `http://localhost:7860`

### 5. Docker build & run

```bash
docker build -t mediguard-ai .
docker run -p 7860:7860 mediguard-ai
```

---

## 📊 Baseline Scores

| Task | Difficulty | Score | How |
|------|-----------|-------|-----|
| Suppression | ⭐ Easy | **1.00** | Zero false alarms on hypertensive patient |
| Deterioration | ⭐⭐ Medium | **0.80** | Detected sepsis via SpO2/temp trends |
| Triage | ⭐⭐⭐ Hard | **0.40** | Rule-based baseline; trained agent would score higher |

> For a detailed explanation of every file and concept, see **walkthrough.md**

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py                          │
│  ┌────────────┐    ┌───────────────────────────────┐    │
│  │  Baseline   │───▶│       MediGuardEnv            │    │
│  │   Agent     │◀───│  (mediguard_env.py)           │    │
│  │             │    │                               │    │
│  │ 3 Strategies│    │  • Normalizes vitals 0–1      │    │
│  │ • Delta     │    │  • Tracks personal baseline   │    │
│  │ • Absolute  │    │  • Keeps 10-step history      │    │
│  │ • Trends    │    │  • Classifies condition       │    │
│  │             │    │  • Computes reward via RF      │    │
│  └────────────┘    │  • Manages 360-step episodes  │    │
│                     │                               │    │
│  ┌────────────┐    │  ┌───────────────────────┐    │    │
│  │  OpenAI    │    │  │  PatientSimulator     │    │    │
│  │  Client    │    │  │  (patient_simulator)  │    │    │
│  │ (ready for │    │  │                       │    │    │
│  │  LLM agent)│    │  │  Generates vitals     │    │    │
│  └────────────┘    │  └───────────────────────┘    │    │
│                     │                               │    │
│                     │  ┌───────────────────────┐    │    │
│                     │  │  RewardFunction       │    │    │
│                     │  │  (reward_function.py)  │    │    │
│                     │  │  + Task Graders       │    │    │
│                     │  └───────────────────────┘    │    │
│                     └───────────────────────────────┘    │
│                                                          │
│  Output: [START] / [STEP] / [END] logs                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  server.py (Gradio)           │  openenv.yaml           │
│  Interactive UI + API         │  Metadata + task defs   │
│  Served via Dockerfile        │                         │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Hackathon Checklist

| Requirement | Status |
|------------|--------|
| Real-world task simulation (ICU monitoring) | ✅ Done |
| OpenEnv spec: typed Pydantic models | ✅ Done |
| OpenEnv spec: `step()` / `reset()` / `state()` | ✅ Done |
| `openenv.yaml` with metadata | ✅ Done |
| 3 tasks (easy → medium → hard) | ✅ Done |
| Agent graders (0.0–1.0 scores) | ✅ Done |
| Meaningful reward function (varying signal) | ✅ Done |
| Alarm fatigue modifier + personalization bonus | ✅ Done |
| Baseline `inference.py` with reproducible scores | ✅ Done |
| Structured stdout: `[START]` / `[STEP]` / `[END]` | ✅ Done |
| OpenAI client for LLM calls | ✅ Done |
| Environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | ✅ Done |
| `Dockerfile` builds | ✅ Done |
| Gradio server on port 7860 | ✅ Done |
| `requirements.txt` | ✅ Done |
| Runs on vcpu=2, 8GB RAM, no GPU | ✅ Done |
| Completes in under 20 minutes | ✅ Done |
| `seed=42` reproducible | ✅ Done |
| Deploy to HuggingFace Spaces | 🔜 Pending |

---

## 📋 Constraints & Requirements

- **Hardware:** vcpu=2, 8GB RAM — no GPU required
- **Runtime:** Must complete all 3 tasks in under 20 minutes
- **Reproducibility:** `seed=42` must produce identical output every run
- **Python version:** 3.9+
- **Dependencies:** `numpy`, `pydantic`, `openai`, `gradio`
