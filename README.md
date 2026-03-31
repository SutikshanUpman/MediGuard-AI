# MediGuard AI - Smart Patient Monitoring 🏥

**Meta PyTorch OpenEnv Hackathon x SST | India AI Hackathon 2026**

---

## 🚨 The Problem

Hospital monitors generate **100+ false alarms per day** per patient. They beep when BP hits 130 — even if that's normal for YOU. Nurses start ignoring alarms. Real emergencies get missed.

---

## 💡 Our Solution

An **RL-powered patient monitoring system** that:
- Learns each patient's **personal baseline** over 3 days
- Combines **vitals + live camera context** before alerting
- Uses **3-action decision making**: Alert 🔴 / Verify 🟡 / Ignore 🟢
- Adapts medication dosing based on real-time lab results

**The Pitch**: *"We don't ask if 130 is high. We ask if 130 is high for YOU, today."*

---

## 🏗️ Project Structure

```
MediGuard-AI/
│
├── agents/                          # RL Agent implementations
│   ├── alert_agent/                 # 3-action alert decision agent
│   │   ├── agent.py                 # PPO/DQN agent training logic
│   │   ├── policy.py                # Neural network policy
│   │   └── environment.py           # Custom gym environment for alerts
│   └── dosing_agent/                # Drug dosing optimization agent
│       ├── agent.py                 # RL agent for medication adjustment
│       ├── policy.py                # Dosing policy network
│       └── environment.py           # Dosing environment with safety constraints
│
├── configs/                         # Configuration files
│   ├── base/
│   │   ├── model_config.yaml        # Model hyperparameters (hidden layers, lr, etc.)
│   │   └── training_config.yaml     # Training settings (epochs, batch size, etc.)
│   └── experiments/
│       ├── alert_agent.yaml         # Alert agent specific configs
│       └── dosing_agent.yaml        # Dosing agent specific configs
│
├── data/                            # Data storage (gitignored)
│   ├── raw/                         # Raw patient vitals data
│   ├── processed/                   # Preprocessed training data
│   └── synthetic/                   # Synthetic patient data for training
│
├── environments/                    # Custom RL environments
│   ├── hospital_monitor/
│   │   └── vitals_env.py            # Patient vitals monitoring environment
│   └── drug_dosing/
│       └── dosing_env.py            # Medication dosing environment
│
├── models/                          # Model storage (gitignored)
│   ├── checkpoints/                 # Training checkpoints
│   └── pretrained/                  # Pre-trained models
│
├── utils/                           # Helper utilities
│   ├── metrics/
│   │   └── evaluation.py            # Precision, recall, false alarm rate metrics
│   └── visualization/
│       └── plots.py                 # Plot vitals, alerts, RL decisions
│
├── notebooks/                       # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb    # Explore patient vitals data
│   ├── 02_baseline_simulation.ipynb # Simulate traditional alarm system
│   ├── 03_rl_training.ipynb         # Train RL agents
│   └── 04_demo.ipynb                # Live demo of the system
│
├── app/                             # Demo applications
│   ├── dashboard/
│   │   └── app.py                   # Gradio/Streamlit dashboard
│   └── api/
│       └── main.py                  # FastAPI serving endpoint
│
├── tests/                           # Unit tests
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation
└── README.md                        # This file
```

---

## 🧠 System Architecture

### 1️⃣ Context Filter
Combines **vitals** (HR, BP, SpO2) + **camera feed** (eating? lying still? falling?) before making decisions.

**Example**: Heart rate 110 while eating = fine. Heart rate 110 while motionless = problem.

### 2️⃣ Alert Agent (3-Action RL)
Instead of binary beep/silence:
- 🔴 **Alert** → Real emergency, immediate attention
- 🟡 **Verify** → Silent nudge to nurse ("check when free")
- 🟢 **Ignore** → All good, suppress alarm

### 3️⃣ Personalization Module
Spends **3 days learning YOUR baseline**. 145 BP might be normal for you. The AI adapts to the person, not a textbook.

### 4️⃣ Drug Dosing Agent
Separate RL agent that adjusts medication doses daily based on lab results. No more weekly trial-and-error.

---

## 🚀 Development Stages

### ✅ Stage 1: Setup & Structure (Current)
- [x] Project scaffolding
- [x] Environment setup
- [x] Git initialization

### 🔄 Stage 2: Data & Simulation (Week 1)
- [ ] Create synthetic patient vitals generator
- [ ] Simulate traditional alarm system (baseline)
- [ ] Build patient activity simulator (camera context)
- [ ] Generate training dataset

### 🔄 Stage 3: RL Agent Development (Week 2-3)
- [ ] Implement Alert Agent environment (state, action, reward)
- [ ] Train PPO/DQN agent for 3-action decision making
- [ ] Implement personalization baseline learning
- [ ] Implement Dosing Agent environment
- [ ] Train medication optimization agent

### 🔄 Stage 4: Integration & Testing (Week 4)
- [ ] Combine vitals + camera context
- [ ] Build real-time inference pipeline
- [ ] Evaluate metrics: precision, recall, false alarm rate
- [ ] Compare RL vs traditional alarms

### 🔄 Stage 5: Demo & Deployment (Week 5)
- [ ] Build Gradio dashboard
- [ ] Create API endpoints
- [ ] Record demo video
- [ ] Prepare hackathon presentation

---

## 📊 Future Plans

### Short-term (Post-Hackathon)
- Integration with real hospital FHIR/HL7 APIs
- Multi-patient dashboard for nurse stations
- Mobile app for real-time alerts
- A/B testing in simulated ICU environment

### Long-term (Research)
- Federated learning across hospitals (privacy-preserving)
- Multi-modal context (voice, movement sensors, EEG)
- Explainable AI for clinical trust
- Integration with electronic health records (EHR)

---

## 🛠️ Setup Instructions

### Prerequisites
- Windows 11 (or Linux/macOS)
- NVIDIA GPU (RTX 4060 or better) with CUDA 12.1
- Miniconda installed

### Installation

```bash
# Clone the repository
git clone https://github.com/SutikshanUpman/MediGuard-AI.git
cd MediGuard-AI

# Create conda environment
conda create -n openenv-triage python=3.11
conda activate openenv-triage

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Quick Start

```bash
# Activate environment
conda activate openenv-triage

# Navigate to notebooks
cd notebooks

# Launch Jupyter
jupyter lab

# Open 01_data_exploration.ipynb
```

---

## 🎯 Why RL?

Every patient is different. Responses are delayed. Conditions keep changing. This is **exactly what RL is built for**:

- **Personalization** → learns individual patient patterns
- **Delayed rewards** → handles time-lagged consequences
- **Non-stationary environments** → adapts as patient condition evolves
- **Multi-objective optimization** → balances safety vs. alarm fatigue

---

**Built with ❤️ for better healthcare**
