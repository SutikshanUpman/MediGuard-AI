# 🏥 MediGuard-AI — Complete Project Walkthrough

> **Reading time:** 10 minutes. No coding knowledge needed.

---

## 🤔 What is this project?

Imagine you're a nurse in an ICU (Intensive Care Unit) watching monitors for 4 patients. Every minute, you see numbers on a screen:

- Heart rate: 82 bpm
- Blood pressure: 140/90
- Oxygen level: 96%
- Temperature: 37.2°C

**Your job:** Decide whether to **ignore** (everything's fine), **check closer** (something seems off), or **call the doctor immediately** (this is serious).

The problem? Some patients naturally have high blood pressure — 140/90 is their normal. If you panic every time you see 140, you'll be running around all day for nothing. Doctors call this **alarm fatigue** — when you see so many false alarms that you start ignoring the real ones.

**MediGuard-AI trains an AI to make these decisions.** It learns when to worry and when to relax, just like an experienced nurse would.

---

## 🧠 The Key Concept: Reinforcement Learning (RL)

Think of training a puppy:

1. Puppy sits when you say "sit" → You give a treat (**positive reward**)
2. Puppy barks at nothing → You say "no" (**negative reward**)
3. Over time, the puppy learns what earns treats and what doesn't

**Reinforcement Learning works the same way:**

1. The AI sees patient vitals (the **observation**)
2. The AI makes a decision — ignore, verify, or alert (the **action**)
3. The system tells the AI if that was smart or dumb (the **reward**)
4. Over thousands of rounds, the AI learns what works

Our project builds the **training gym** — the fake hospital that the AI practices in. We don't train the AI ourselves; we build the place where it learns.

---

## 📁 The Files — What Each One Does

Think of the project like a hospital simulation game:

```
🏥 MediGuard-AI/
│
├── 🫀 patient_simulator.py    ← The fake patients
├── 📊 reward_function.py      ← The scoring system  
├── 📝 task1_suppression.py    ← Easy exam
├── 📝 task2_deterioration.py  ← Medium exam
├── 📝 task3_triage.py         ← Hard exam
├── 🎮 mediguard_env.py        ← The game engine
├── 🤖 inference.py            ← A test player (simple robot)
├── 🌐 server.py               ← Puts the game online
├── ⚙️ openenv.yaml            ← Game settings file
├── 🐳 Dockerfile              ← Packaging instructions
├── 📦 requirements.txt        ← Shopping list of tools needed
├── 📖 README.md               ← Project manual
└── 📖 walkthrough.md          ← You are here!
```

---

### 🫀 `patient_simulator.py` — The Fake Patients

**What it does:** Creates virtual patients with realistic vital signs.

**In real life:** A patient's heart rate isn't always 75. It goes up when they walk, down when they sleep, and spikes when something's wrong. This file simulates all of that.

**5 types of patients:**

| Patient Type | What's Going On | Example |
|---|---|---|
| **Healthy** | Everything's normal | Heart rate ~72, BP 120/80 |
| **Hypertensive** | Always has high blood pressure (it's their normal) | BP always ~150/95 — this is OK for THEM |
| **Deteriorating** | Slowly getting sicker over 6 hours (sepsis) | Starts normal, slowly: temp rises, oxygen drops |
| **Post-op** | Just had surgery, recovering | Low BP ~100/65, gradually improving |
| **Unstable** | Random spikes and drops | Unpredictable vitals |

**How it works inside:**

```
Every "minute" (timestep):
  1. Start with the patient's baseline (e.g., HR=72 for healthy)
  2. Add small random noise (±3 bpm) — like real life
  3. Adjust for activity — if walking, HR goes up by ~15
  4. If deteriorating, slowly shift vitals toward danger
  5. Return the final numbers
```

**Key detail:** The patient has "activities" throughout the day:
- 0 = Resting (lying in bed)
- 1 = Eating
- 2 = Walking around (this makes heart rate go UP — the AI must learn this is normal!)
- 3 = In distress
- 4 = Falling

---

### 📊 `reward_function.py` — The Scoring System

**What it does:** Tells the AI "good job" or "bad job" after every decision.

**Think of it like a report card for each action:**

| | Emergency 🚨 | Borderline ⚠️ | Stable ✅ | Drug-Masked 💊 |
|---|---|---|---|---|
| **Alert** 🔔 | +1.0 (Great!) | +0.2 (Okay) | -0.5 (Overreacting) | +1.0 (Great!) |
| **Verify** 🔍 | -0.8 (Too slow!) | +0.4 (Smart!) | +0.1 (Fine) | -0.5 (Too slow!) |
| **Ignore** 😴 | -2.0 (TERRIBLE!) | -0.1 (Risky) | +0.2 (Perfect) | -2.0 (TERRIBLE!) |

**Reading the table:**
- If a patient is having an emergency and the AI **ignores** it → **-2.0** (worst possible score)
- If a patient is stable and the AI **ignores** it → **+0.2** (correct — don't bother them)
- If a patient is stable and the AI **alerts** → **-0.5** (you just woke up the doctor for nothing)

**Two bonus features:**

1. **Alarm Fatigue Penalty:** If the AI fires more than 5 alerts in the last 30 steps → rewards get cut by 40%. This teaches the AI not to spam alerts.

2. **Personalization Bonus:** After 200 steps (when the AI has watched the patient long enough to know their baseline), correctly ignoring a known-normal reading earns a +0.2 bonus. This rewards learning each patient's personal "normal."

---

### 📝 The Three Exams (Task Graders)

These grade how well the AI did at the end of each 360-step episode.

#### Task 1: Suppression (Easy) — `task1_suppression.py`

**The scenario:** A patient with naturally high blood pressure (150/95). Their vitals look "alarming" but are actually normal FOR THEM.

**What the AI must learn:** "This person always has high BP. Stop freaking out."

**How it's scored:**
- Count how many times the AI triggered a false alarm (ALERT when patient was actually fine)
- If false alarm rate < 5% → score = **1.0** (perfect)
- If false alarm rate > 60% → score = **0.0** (terrible — a naive AI that alerts at everything)
- In between → proportional score

**Our baseline scores: 1.00** ✅ (The rule-based agent learns the baseline and stops alerting)

---

#### Task 2: Deterioration (Medium) — `task2_deterioration.py`

**The scenario:** A patient who starts healthy but slowly develops sepsis. Over 6 hours, their oxygen drops, temperature rises, and heart rate increases — but SO SLOWLY that it's hard to notice.

**What the AI must learn:** "Something is changing. I should verify early and alert before it becomes critical."

**How it's scored (two phases):**
- **VERIFY early (before step 180, ~3 hours):** Did the AI notice something was off? → up to **0.3 bonus**
- **ALERT on time (before step 300, ~5 hours):** Did the AI call for help before it was too late? → up to **0.7**

Total possible: **1.0**

**Our baseline scores: 0.80** ✅ (Detects via SpO2 dropping and temperature trends)

---

#### Task 3: Triage (Hard) — `task3_triage.py`

**The scenario:** 4 patients at once:
- Patient 0: Healthy (fine, leave them alone)
- Patient 1: Post-op (recovering, just monitor)
- Patient 2: Deteriorating (the one who actually needs help)
- Patient 3: Healthy (fine)

**What the AI must learn:** "Patient 2 is the one in trouble. Focus on them, don't waste time on the others."

**How it's scored (three components):**
- **50% — F1 score:** Are you correctly identifying sick vs healthy patients?
- **30% — Masked detection:** Can you catch emergencies even when the patient is walking (making their heart rate look normal)?
- **20% — Triage priority:** Are you giving the sickest patient the most attention?

**Our baseline scores: 0.40** ⚠️ (The simple rule-based agent struggles with multi-patient management — a trained AI would do better)

---

### 🎮 `mediguard_env.py` — The Game Engine

**What it does:** This is the main file that ties everything together. It's the "game" that the AI plays.

**Think of it as a video game loop:**

```
1. RESET — New episode starts
   └── Create fresh patients
   └── Clear all history
   └── Return first observation

2. STEP (repeated 360 times) — Each "minute"
   ├── Receive the AI's action (ignore/verify/alert)
   ├── Tick the patient simulator forward 1 minute
   ├── Get new vital signs
   ├── Normalize everything to 0-1 range
   ├── Update the patient's personal baseline
   ├── Classify condition (STABLE / BORDERLINE / EMERGENCY)
   ├── Calculate reward using reward_function.py
   ├── Check if episode is done (360 steps = 6 hours)
   └── Return: new observation, reward, done, info

3. STATE — Debug info
   └── Return everything about the current state
```

**What the AI sees each step (the observation):**

| Field | What It Is | Example |
|---|---|---|
| `heart_rate` | Normalized HR (0=30bpm, 1=200bpm) | 0.33 = ~86 bpm |
| `systolic_bp` | Normalized systolic BP | 0.56 = ~150 mmHg |
| `diastolic_bp` | Normalized diastolic BP | 0.59 = ~95 mmHg |
| `spo2` | Normalized oxygen (0=70%, 1=100%) | 0.92 = ~97.6% |
| `respiratory_rate` | Normalized breathing rate | 0.41 = ~19 breaths/min |
| `temperature` | Normalized temp (0=34°C, 1=42°C) | 0.39 = ~37.1°C |
| `baseline_delta` | How far from their personal normal | 0.02 = very close to normal |
| `hours_observed` | How long we've been watching | 2.5 = 2.5 hours in |
| `activity` | What they're doing right now | 2 = walking |
| `vitals_history` | Last 10 readings | [list of past vitals] |

**Why normalize?** Raw values are different scales (HR: 30-200, SpO2: 70-100, Temp: 34-42). Normalizing them all to 0-1 makes it easier for the AI to learn.

**What the AI does each step (the action):**
- `0` = Ignore ("all good")
- `1` = Verify ("let me check closer")
- `2` = Alert ("GET A DOCTOR!")
- For triage: `[0, 1, 2, 0]` = one action per patient

---

### 🤖 `inference.py` — The Test Player

**What it does:** A simple "robot" that plays the game using basic IF-THEN rules. It's NOT the final AI — it's just a baseline to prove the environment works.

**The robot's brain (3 strategies):**

```
Strategy 1: Baseline Delta
  "If vitals suddenly changed a LOT from their personal normal → ALERT"
  "If vitals changed a MODERATE amount → VERIFY"

Strategy 2: Absolute Thresholds
  "If oxygen drops below 86% → VERIFY"
  "If oxygen drops below 80% → ALERT"  
  "If temperature goes above 39°C → VERIFY"
  "If temperature goes above 40°C → ALERT"

Strategy 3: Trend Detection
  "Compare the last 3 readings to the oldest 3 in history"
  "If oxygen is dropping OR temperature is rising → escalate"
```

**Output format (required by hackathon):**
```
[START] task=suppression env=mediguard model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=1 reward=0.66 done=false error=null
[STEP] step=2 action=0 reward=0.69 done=false error=null
...
[END] success=true steps=360 score=1.00 rewards=0.66,0.69,...
```

---

### 🌐 `server.py` — The Online Interface

**What it does:** Puts your environment on the internet so anyone can interact with it. Uses **Gradio** (a Python library for building web apps).

**What you see when you open it:**

```
┌──────────────────────────────────────────────┐
│  🏥 MediGuard-AI                              │
│                                                │
│  Task: [suppression ▼]    Action: [1        ]  │
│  Seed: [42         ]      [▶️ Step]            │
│  [🔄 Reset Environment]                       │
│                                                │
│  Status: Step 5 | Reward: 0.688 | Done: False  │
│  Observation: HR=0.333 SpO2=0.920 ...          │
└──────────────────────────────────────────────┘
```

**Why Gradio?** HuggingFace Spaces (where we deploy) natively supports Gradio.

---

### ⚙️ `openenv.yaml` — The Game Settings

**What it does:** A configuration file that tells the hackathon's automated grading system about our environment.

**Contains:**
- Environment name and description
- Which Python class to load (`MediGuardEnv`)
- List of all 3 tasks with difficulty levels
- What the actions and observations look like
- Hardware requirements (2 CPUs, 8GB RAM, no GPU)

---

### 🐳 `Dockerfile` — The Packaging Box

**What it does:** Instructions for putting the entire project into a **container** (a self-contained box that runs anywhere).

**Analogy:** It's like a recipe card:
```
1. Start with Python 3.11 (the oven)
2. Install gradio, numpy, etc. (the ingredients)
3. Copy all our code files (the recipe)
4. Run server.py on port 7860 (serve the dish)
```

---

## 🔄 How Everything Connects

```
YOU (or an AI agent)
  │
  │  "action = 1 (verify)"
  ▼
🎮 mediguard_env.py (Game Engine)
  │
  ├──→ 🫀 patient_simulator.py
  │      "tick forward 1 minute"
  │      "return: HR=88, BP=142/93, SpO2=96%..."
  │
  ├──→ 📊 reward_function.py  
  │      "action=VERIFY, condition=STABLE"
  │      "reward = +0.1 (base) - 0 (no fatigue) = +0.1"
  │      "normalized: (0.1 + 2.0) / 3.2 = 0.656"
  │
  └──→ 📝 task grader (at episode end)
         "false alarm rate = 3%, score = 0.96"
  │
  │  returns: observation, reward=0.656, done=false
  ▼
YOU see the result and choose next action
```

---

## 📊 Current Scores

| Task | Difficulty | Score | What It Means |
|------|-----------|-------|---------------|
| Suppression | ⭐ Easy | **1.00** | Perfect — zero false alarms on the hypertensive patient |
| Deterioration | ⭐⭐ Medium | **0.80** | Good — detected sepsis via SpO2/temp trends, alerted on time |
| Triage | ⭐⭐⭐ Hard | **0.40** | Baseline — a trained AI agent would score much higher |

**These are baseline scores from a simple rule-based robot.** The whole point of the environment is that a real RL-trained agent (or an LLM like GPT/Qwen) would learn to score higher by playing thousands of episodes.

---

## 🔑 Key Concepts Glossary

| Term | Plain English |
|------|--------------|
| **Reinforcement Learning** | Teaching AI through trial-and-error with rewards |
| **Environment** | The simulated world the AI practices in |
| **Agent** | The AI that makes decisions |
| **Observation** | What the AI sees (vital signs) |
| **Action** | What the AI does (ignore/verify/alert) |
| **Reward** | Score for each action (+good, -bad) |
| **Episode** | One complete playthrough (360 steps = 6 hours) |
| **Step** | One "minute" in the simulation |
| **Baseline** | A patient's personal normal (differs per person) |
| **Alarm Fatigue** | When too many false alarms make you ignore real ones |
| **Normalization** | Converting different scales (30-200 bpm, 70-100% SpO2) to a common 0-1 range |
| **Grader** | Final exam score (0.0 to 1.0) at end of episode |
| **OpenEnv** | The hackathon's standard format for RL environments |
| **Pydantic** | A Python library that validates data types |
| **Gradio** | A Python library for building web interfaces |
| **Docker** | A container system that packages the entire app so it runs anywhere |
| **HuggingFace Spaces** | A website that hosts AI demos for free |
| **Seed** | A number (42) that makes random results reproducible |

---

## 🏁 Summary

We built a **virtual ICU** where AI agents can practice monitoring patients. The environment:

1. **Generates realistic patients** with different conditions
2. **Rewards smart decisions** (catch emergencies, don't cry wolf)
3. **Grades the AI** on three increasingly difficult scenarios
4. **Runs as a web app** anyone can interact with

It's like building a flight simulator for doctors — the AI crashes a thousand times in simulation so it never crashes in real life. 🏥✈️
