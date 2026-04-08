"""
MediGuard-AI — HuggingFace Spaces App
Game-style bright UI. Clean. Readable. No ulti vibes.
"""

import json
import os
import sys

import gradio as gr
from mediguard_env import MediGuardEnv
from inference import (
    baseline_agent, triage_baseline,
    obs_to_user_message, triage_obs_to_message,
    ACTIVITY_NAMES, IGNORE, VERIFY, ALERT,
    MODEL_BY_TASK, MODEL_NAME, API_KEY, API_BASE_URL,
    llm_agent, triage_llm_agent,
)

# ── Token check ────────────────────────────────────────────────────
_llm_available = API_KEY not in (None, "", "dummy")

# ── Global episode state ───────────────────────────────────────────
_env            = None
_current_task   = None
_step_count     = 0
_total_reward   = 0.0
_episode_log    = []
_conv_history   = []
_vitals_history = []
_last_obs       = None

ACTION_LABELS = {0: "IGNORE", 1: "VERIFY", 2: "ALERT"}
ACTION_EMOJI  = {0: "😴", 1: "🔍", 2: "🚨"}

ACTIVITY_EMOJI = {
    0: "Resting",
    1: "Eating",
    2: "Walking",
    3: "Distressed",
    4: "Falling",
}

_agent_choices = ["Rule-Based", "Manual"]
_agent_default = "Rule-Based"
if _llm_available:
    _agent_choices = ["LLM Agent"] + _agent_choices
    _agent_default = "LLM Agent"


# ══════════════════════════════════════════════════════════════════
#  Observation formatting  — clean, no tick symbols
# ══════════════════════════════════════════════════════════════════

def _risk_tag(delta, spo2, temp):
    if delta > 0.5 or spo2 < 0.4 or temp > 0.75:
        return "CRITICAL"
    elif delta > 0.25 or spo2 < 0.6 or temp > 0.55:
        return "BORDERLINE"
    return "STABLE"


def _fmt_single(obs: dict) -> str:
    hr   = obs.get("heart_rate", 0)
    spo2 = obs.get("spo2", 0)
    bp   = obs.get("systolic_bp", 0)
    temp = obs.get("temperature", 0)
    rr   = obs.get("respiratory_rate", 0)
    dbp  = obs.get("diastolic_bp", 0)
    delta= obs.get("baseline_delta", 0)
    hours= obs.get("hours_observed", 0)
    act  = ACTIVITY_EMOJI.get(obs.get("activity", 0), "Unknown")

    hr_raw   = int(30  + hr   * 170)
    spo2_raw = int(70  + spo2 * 30)
    bp_raw   = int(60  + bp   * 160)
    dbp_raw  = int(40  + dbp  * 80)
    temp_raw = round(34 + temp * 8, 1)
    rr_raw   = int(5   + rr   * 35)
    risk     = _risk_tag(delta, spo2, temp)
    risk_map = {"CRITICAL": "[CRITICAL]", "BORDERLINE": "[BORDERLINE]", "STABLE": "[STABLE]"}

    lines = [
        "  PATIENT VITALS",
        "  " + "─"*40,
        f"  Heart Rate         {hr_raw:>4} bpm",
        f"  SpO2               {spo2_raw:>4} %",
        f"  Blood Pressure     {bp_raw}/{dbp_raw} mmHg",
        f"  Temperature        {temp_raw} C",
        f"  Resp Rate          {rr_raw} /min",
        "  " + "─"*40,
        f"  Baseline Delta     {delta:.3f}",
        f"  Time Observed      {hours:.1f} hours",
        f"  Activity           {act}",
        "  " + "─"*40,
        f"  Status             {risk_map[risk]}",
    ]
    return "\n".join(lines)


def _fmt_triage(obs_list: list) -> str:
    lines = ["  4-PATIENT TRIAGE BOARD", "  " + "─"*42]
    risk_map = {"CRITICAL": "[CRITICAL]", "BORDERLINE": "[BORDERLINE]", "STABLE": "[STABLE]"}
    for i, obs in enumerate(obs_list):
        hr   = obs.get("heart_rate", 0)
        spo2 = obs.get("spo2", 0)
        temp = obs.get("temperature", 0)
        delta= obs.get("baseline_delta", 0)
        act  = ACTIVITY_EMOJI.get(obs.get("activity", 0), "Unknown")
        risk = _risk_tag(delta, spo2, temp)
        lines += [
            f"  Patient {i}   {risk_map[risk]}",
            f"    HR {int(30+hr*170)} bpm   SpO2 {int(70+spo2*30)}%   "
            f"Temp {34+temp*8:.1f}C   Delta {delta:.2f}",
            f"    Activity: {act}",
            "  " + "─"*40,
        ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  Agent helpers
# ══════════════════════════════════════════════════════════════════

def _agent_action(obs, task, agent_mode):
    if agent_mode == "LLM Agent" and _llm_available:
        model = MODEL_BY_TASK.get(task, MODEL_NAME)
        try:
            if task == "triage":
                action, reasoning = triage_llm_agent(obs, _conv_history, model)
            else:
                action, reasoning = llm_agent(obs, task, _conv_history, _vitals_history, model)
            return action, reasoning, True
        except Exception as e:
            err = f"llm_err:{type(e).__name__}"
            return (triage_baseline(obs) if task == "triage" else baseline_agent(obs)), err, False
    if task == "triage":
        return triage_baseline(obs), "rule-based", False
    return baseline_agent(obs), "rule-based", False


def _compute_score(env, task):
    fn = {"suppression": env.false_alarm_rate_grader,
          "deterioration": env.deterioration_grader,
          "triage": env.triage_grader}.get(task)
    return float(fn()) if fn else 0.0


# ══════════════════════════════════════════════════════════════════
#  Demo functions
# ══════════════════════════════════════════════════════════════════

def demo_reset(task, seed, agent_mode):
    global _env, _current_task, _step_count, _total_reward
    global _episode_log, _conv_history, _vitals_history, _last_obs

    _env = MediGuardEnv(task=task, seed=int(seed))
    _current_task = task
    _step_count = 0
    _total_reward = 0.0
    _episode_log = []
    _conv_history = []
    _vitals_history = []

    obs = _env.reset()
    _last_obs = obs

    obs_display = _fmt_triage(obs) if isinstance(obs, list) else _fmt_single(obs)
    _episode_log.append(f"[RESET] task={task}  seed={seed}  agent={agent_mode}")

    manual = agent_mode == "Manual"
    is_tri = task == "triage"
    difficulty = {"suppression": "Easy", "deterioration": "Medium", "triage": "Hard"}[task]
    status = f"READY   {task.upper()}  [{difficulty}]   Seed {seed}   {agent_mode}"

    # FIX 1: run_all_btn removed from shared outputs — stays always interactive
    return (
        status, obs_display,
        "—", "—", "—", "0 / 360",
        "\n".join(_episode_log),
        gr.update(interactive=True),   # step_btn
        gr.update(visible=manual and not is_tri),
        gr.update(visible=manual and is_tri),
    )


def demo_step(action_radio, triage_txt, agent_mode):
    global _env, _step_count, _total_reward
    global _episode_log, _conv_history, _vitals_history, _last_obs

    if _env is None or _last_obs is None:
        return ("Reset the environment first!",) + ("",)*6 + ("",) + (gr.update(),)*3

    task = _current_task
    obs  = _last_obs

    if agent_mode == "Manual":
        if task == "triage":
            try:
                parsed = [max(0, min(2, int(a.strip()))) for a in triage_txt.split(",")]
                while len(parsed) < 4: parsed.append(0)
                action = parsed[:4]
            except ValueError:
                action = [0, 0, 0, 0]
        else:
            try:
                action = int(action_radio[0]) if action_radio else 1
            except Exception:
                action = 1
        reasoning, used_llm = "manual", False
    else:
        action, reasoning, used_llm = _agent_action(obs, task, agent_mode)

    obs_next, reward, done, info = _env.step(action)
    _step_count += 1
    _total_reward += reward
    mean_r = _total_reward / _step_count
    _last_obs = obs_next

    if task == "triage":
        obs_text = triage_obs_to_message(
            obs_next if isinstance(obs_next, list) else [obs_next], _conv_history)
    else:
        obs_text = obs_to_user_message(obs_next, task, _vitals_history, _conv_history)
        hist = obs_next.get("vitals_history", [])
        if hist:
            for entry in reversed(hist):
                if any(v != 0.0 for v in entry):
                    _vitals_history.append(entry)
                    break
            _vitals_history = _vitals_history[-8:]

    _conv_history.append({
        "obs_text": obs_text,
        "response": json.dumps({"action": action, "reasoning": reasoning}),
        "action": action, "reward": reward,
    })
    if len(_conv_history) > 4:
        _conv_history = _conv_history[-4:]

    if isinstance(action, list):
        act_str = "  ".join(
            f"P{i}:{ACTION_EMOJI.get(a,'?')}{ACTION_LABELS.get(a,'?')}"
            for i, a in enumerate(action))
    else:
        act_str = f"{ACTION_EMOJI.get(action,'?')} {ACTION_LABELS.get(action,'?')}"

    obs_display = _fmt_triage(obs_next) if isinstance(obs_next, list) else _fmt_single(obs_next)

    tag = "LLM" if used_llm else ("RULE" if agent_mode == "Rule-Based" else "USER")
    _episode_log.append(
        f"[{_step_count:03d}] {tag}  {act_str[:38]}  R={reward:+.3f}")
    if len(_episode_log) > 200:
        _episode_log = _episode_log[-200:]

    manual = agent_mode == "Manual"
    is_tri = task == "triage"

    if done:
        score = _compute_score(_env, task)
        status = f"COMPLETE   Score: {score:.4f}   Mean Reward: {mean_r:.3f}   Steps: {_step_count}"
        _episode_log += ["=" * 50,
                         f"  FINAL SCORE  :  {score:.4f}",
                         f"  MEAN REWARD  :  {mean_r:.4f}",
                         "=" * 50]
        # FIX 1: removed run_all_btn from outputs
        return (status, obs_display, act_str, f"{reward:+.3f}", f"{mean_r:.4f}",
                f"{_step_count}/360", "\n".join(_episode_log[-80:]),
                gr.update(interactive=False),
                gr.update(visible=manual and not is_tri), gr.update(visible=manual and is_tri))

    status = f"Step {_step_count}/360   Last Reward: {reward:+.3f}   Mean: {mean_r:.3f}"
    # FIX 1: removed run_all_btn from outputs
    return (status, obs_display, act_str, f"{reward:+.3f}", f"{mean_r:.4f}",
            f"{_step_count}/360", "\n".join(_episode_log[-80:]),
            gr.update(interactive=True),
            gr.update(visible=manual and not is_tri), gr.update(visible=manual and is_tri))


def demo_run_all(_):
    import subprocess
    try:
        r = subprocess.run([sys.executable, "inference.py"],
                           capture_output=True, text=True,
                           env=os.environ.copy(), timeout=900)
        return (r.stdout + ("\n" + r.stderr if r.stderr else "")) or "No output."
    except subprocess.TimeoutExpired:
        return "Timed out after 15 min."
    except Exception as e:
        return f"Error: {e}"


def on_agent_change(agent_mode, task):
    m, t = agent_mode == "Manual", task == "triage"
    return gr.update(visible=m and not t), gr.update(visible=m and t)


def on_task_change(task, agent_mode):
    m, t = agent_mode == "Manual", task == "triage"
    return gr.update(visible=m and not t), gr.update(visible=m and t)


# ══════════════════════════════════════════════════════════════════
#  API wrappers
# ══════════════════════════════════════════════════════════════════

def reset_env(task="suppression", seed=42):
    global _env, _current_task, _last_obs
    _env = MediGuardEnv(task=task, seed=int(seed))
    _current_task = task
    obs = _env.reset()
    _last_obs = obs
    return json.dumps({"observation": obs, "info": {"task": task, "seed": seed}}, default=str)

def step_env(action: str):
    if _env is None: return json.dumps({"error": "reset first"})
    try:
        parsed = [int(a.strip()) for a in action.split(",")] if "," in action else int(action)
        obs, rew, done, info = _env.step(parsed)
        return json.dumps({"observation": obs, "reward": rew, "done": done, "info": info}, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_state():
    if _env is None: return json.dumps({"error": "reset first"})
    return json.dumps(_env.state(), default=str)

def health_check():
    return json.dumps({"status": "ok", "llm_available": _llm_available})


# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&family=Nunito:wght@400;600;700;800&display=swap');

:root {
    --bg:      #f0f4ff;
    --bg2:     #ffffff;
    --bg3:     #e8edf8;
    --border:  #d0d8f0;
    --text:    #1a1f3a;
    --muted:   #6b7280;
    --blue:    #2563eb;
    --green:   #16a34a;
    --amber:   #d97706;
    --red:     #dc2626;
    --mono:    'JetBrains Mono', monospace;
    --display: 'Rajdhani', sans-serif;
    --body:    'Nunito', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--body) !important;
    color: var(--text) !important;
}

/* HEADER */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #7c3aed 100%);
    border-radius: 20px;
    padding: 30px 36px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(29,78,216,0.25);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.07), transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    font-family: var(--display);
    font-size: 2.5em;
    font-weight: 700;
    color: #fff;
    margin: 0 0 5px;
    letter-spacing: 1px;
}
.app-header .tagline { color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 0 0 18px; }
.pill {
    display: inline-block;
    border-radius: 30px;
    padding: 4px 13px;
    font-size: 0.75em;
    font-weight: 700;
    font-family: var(--mono);
    margin: 0 5px 0 0;
    letter-spacing: 0.5px;
}
.p-white  { background: rgba(255,255,255,0.18); color: #fff; }
.p-green  { background: #dcfce7; color: #15803d; }
.p-yellow { background: #fef9c3; color: #a16207; }
.p-red    { background: #fee2e2; color: #b91c1c; }
.p-llm-on  { background: #dcfce7; color: #15803d; }
.p-llm-off { background: #fee2e2; color: #b91c1c; }

/* METRIC STRIP */
.gr-textbox { font-family: var(--body) !important; }

/* STATUS BAR */
.status-bar textarea {
    background: #1e3a8a !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 12px !important;
    font-family: var(--mono) !important;
    font-size: 0.87em !important;
    font-weight: 600 !important;
    color: #bfdbfe !important;
}

/* VITALS */
.vitals-box textarea {
    background: #0f172a !important;
    border: 2px solid #2563eb !important;
    border-radius: 14px !important;
    font-family: var(--mono) !important;
    font-size: 0.92em !important;
    font-weight: 500 !important;
    color: #e0f2fe !important;
    line-height: 1.8 !important;
}

/* LOG */
.log-box textarea {
    background: #f8fafc !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: 14px !important;
    font-family: var(--mono) !important;
    font-size: 0.8em !important;
    color: #334155 !important;
    line-height: 1.6 !important;
}

/* FULL LOG */
.full-log textarea {
    background: #fffbeb !important;
    border: 2px solid #fde047 !important;
    border-radius: 14px !important;
    font-family: var(--mono) !important;
    font-size: 0.8em !important;
    color: #78350f !important;
    line-height: 1.6 !important;
}

/* BUTTONS */
.btn-reset {
    background: linear-gradient(135deg,#1d4ed8,#4f46e5) !important;
    border: none !important; border-radius: 12px !important;
    font-family: var(--display) !important; font-size: 1.05em !important;
    font-weight: 700 !important; color: #fff !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(79,70,229,.4) !important;
    transition: all .2s !important;
}
.btn-reset:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(79,70,229,.55) !important; }

.btn-step {
    background: linear-gradient(135deg,#059669,#10b981) !important;
    border: none !important; border-radius: 12px !important;
    font-family: var(--display) !important; font-size: 1.05em !important;
    font-weight: 700 !important; color: #fff !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(16,185,129,.35) !important;
    transition: all .2s !important;
}
.btn-step:hover { transform: translateY(-1px) !important; }

.btn-run {
    background: linear-gradient(135deg,#b45309,#d97706) !important;
    border: none !important; border-radius: 12px !important;
    font-family: var(--display) !important; font-size: 1em !important;
    font-weight: 700 !important; color: #fff !important;
    box-shadow: 0 4px 14px rgba(217,119,6,.3) !important;
}

/* SECTION HEADING */
.sec-h {
    font-family: var(--display);
    font-size: 1.1em;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--blue);
    padding-left: 10px;
    margin: 14px 0 8px;
}

/* Inputs & dropdowns */
.gradio-container input, .gradio-container select,
.gradio-container .svelte-1gfkn6j {
    font-family: var(--body) !important;
    border: 2px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg2) !important;
    color: var(--text) !important;
}

/* Tabs */
.tab-nav { border-bottom: 2px solid var(--border) !important; }
.tab-nav button {
    font-family: var(--display) !important; font-weight: 600 !important;
    font-size: 1em !important; color: var(--muted) !important;
}
.tab-nav button.selected { color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important; }

/* ── FIX 2: Radio button selected state color ── */
/* Gradio wraps radio items in a label with a span for the dot */
.gradio-container .wrap .wrap-inner label {
    border: 2px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    background: var(--bg2) !important;
    color: var(--muted) !important;
    font-family: var(--body) !important;
    font-weight: 600 !important;
}
.gradio-container .wrap .wrap-inner label:hover {
    border-color: var(--blue) !important;
    background: #eff6ff !important;
    color: var(--blue) !important;
}
/* Selected state — Gradio marks selected radio label with aria-checked or adds .selected */
.gradio-container .wrap .wrap-inner label:has(input[type="radio"]:checked) {
    background: var(--blue) !important;
    border-color: var(--blue) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
}
/* The radio dot itself — hide native dot, we use background color as indicator */
.gradio-container .wrap .wrap-inner label input[type="radio"] {
    accent-color: #ffffff !important;
    width: 14px !important;
    height: 14px !important;
}
/* LLM Agent option — green when selected */
.gradio-container .wrap .wrap-inner label:has(input[type="radio"]:checked):first-child {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    border-color: #059669 !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.35) !important;
}
/* Rule-Based option — blue when selected (default already handled above) */
/* Manual option — amber when selected */
.gradio-container .wrap .wrap-inner label:last-child:has(input[type="radio"]:checked) {
    background: linear-gradient(135deg, #b45309, #d97706) !important;
    border-color: #b45309 !important;
    box-shadow: 0 2px 10px rgba(180,83,9,0.35) !important;
}

/* How it works cards */
.hw-card {
    background: var(--bg2);
    border: 2px solid var(--border);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 14px;
}
.hw-card h3 {
    font-family: var(--display);
    font-size: 1.2em;
    font-weight: 700;
    color: var(--blue);
    margin: 0 0 14px;
}
.score-block {
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.sb-g { background: #dcfce7; border-left: 5px solid #16a34a; }
.sb-y { background: #fef9c3; border-left: 5px solid #d97706; }
.sb-r { background: #fee2e2; border-left: 5px solid #dc2626; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg3); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

# ══════════════════════════════════════════════════════════════════
#  HTML blocks
# ══════════════════════════════════════════════════════════════════

_llm_pill = (
    '<span class="pill p-llm-on">LLM CONNECTED</span>'
    if _llm_available else
    '<span class="pill p-llm-off">LLM OFFLINE — set HF_TOKEN</span>'
)

HEADER_HTML = f"""
<div class="app-header">
  <h1>🏥 MediGuard-AI</h1>
  <p class="tagline">Context-Aware ICU Patient Monitoring &nbsp;·&nbsp; OpenEnv Hackathon 2026</p>
  <span class="pill p-green">🟢 Suppression · Easy</span>
  <span class="pill p-yellow">🟡 Deterioration · Medium</span>
  <span class="pill p-red">🔴 Triage · Hard</span>
  <span class="pill p-white">OpenEnv Spec</span>
  <span class="pill p-white">NDCG@4</span>
  {_llm_pill}
</div>
"""

HOW_INSIGHT_HTML = """
<div class="hw-card">
  <h3>The Core Insight</h3>
  <p style="color:#374151;line-height:1.7;margin:0 0 16px">
    Vitals are z-scores from <strong>this patient's own baseline</strong> — not population norms.
    The same reading means something entirely different depending on what the patient is doing.
  </p>
  <table style="width:100%;border-collapse:collapse;font-size:0.9em">
    <tr style="border-bottom:2px solid #e5e7eb">
      <th style="text-align:left;padding:8px 6px;color:#6b7280;font-size:0.83em;font-family:JetBrains Mono,monospace">Situation</th>
      <th style="text-align:center;padding:8px;color:#6b7280;font-size:0.83em;font-family:JetBrains Mono,monospace">HR 130 bpm</th>
      <th style="text-align:center;padding:8px;color:#6b7280;font-size:0.83em;font-family:JetBrains Mono,monospace">Correct Action</th>
    </tr>
    <tr style="border-bottom:1px solid #f3f4f6">
      <td style="padding:10px 6px">🍽 Patient is eating</td>
      <td style="text-align:center;color:#16a34a;font-weight:700">Expected</td>
      <td style="text-align:center"><span style="background:#dcfce7;color:#15803d;padding:4px 16px;border-radius:20px;font-weight:700;font-size:0.88em">😴 IGNORE</span></td>
    </tr>
    <tr style="border-bottom:1px solid #f3f4f6">
      <td style="padding:10px 6px">🚶 Patient is walking</td>
      <td style="text-align:center;color:#d97706;font-weight:700">Possible</td>
      <td style="text-align:center"><span style="background:#fef9c3;color:#a16207;padding:4px 16px;border-radius:20px;font-weight:700;font-size:0.88em">🔍 VERIFY</span></td>
    </tr>
    <tr>
      <td style="padding:10px 6px">🛏 Patient is resting</td>
      <td style="text-align:center;color:#dc2626;font-weight:700">DANGER</td>
      <td style="text-align:center"><span style="background:#fee2e2;color:#b91c1c;padding:4px 16px;border-radius:20px;font-weight:700;font-size:0.88em">🚨 ALERT</span></td>
    </tr>
  </table>
</div>
"""

SCORING_HTML = f"""
<div class="hw-card">
  <h3>Scoring System</h3>
  <div class="score-block sb-g">
    <strong style="color:#15803d;font-size:1.05em">🟢 Suppression &nbsp;·&nbsp; Easy</strong>
    <p style="margin:6px 0 0;color:#374151;font-size:0.87em;line-height:1.6">
      F1 score — harmonic mean of sensitivity and specificity.
      Ignoring everything gives 0. Alerting everything also penalised.
      <br><strong>Baseline ~0.63 &nbsp;·&nbsp; LLM target ~0.80+</strong>
    </p>
  </div>
  <div class="score-block sb-y">
    <strong style="color:#a16207;font-size:1.05em">🟡 Deterioration &nbsp;·&nbsp; Medium</strong>
    <p style="margin:6px 0 0;color:#374151;font-size:0.87em;line-height:1.6">
      Onset-delay: score = 0.4 + 0.6 × (1 − delay/80).
      Detect sepsis drift early for high score. Miss it completely → 0.
      <br><strong>Baseline ~0.30 &nbsp;·&nbsp; LLM target ~0.55+</strong>
    </p>
  </div>
  <div class="score-block sb-r">
    <strong style="color:#b91c1c;font-size:1.05em">🔴 Triage &nbsp;·&nbsp; Hard</strong>
    <p style="margin:6px 0 0;color:#374151;font-size:0.87em;line-height:1.6">
      NDCG@4 (50%) + ALERT-F1 (30%) + Responsiveness (20%) − penalties.
      Sending VERIFY to every patient is penalised. Must differentiate.
      <br><strong>Baseline ~0.22 &nbsp;·&nbsp; LLM target ~0.65+</strong>
    </p>
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════
#  Gradio layout
# ══════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="MediGuard-AI — ICU Monitoring",
    css=CSS,
    theme=gr.themes.Base(),
) as app:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── TAB 1: PLAY ──────────────────────────────────────────
        with gr.Tab("🎮 Interactive Demo"):

            with gr.Row():
                metric_step   = gr.Textbox(value="0 / 360", label="Step",        interactive=False)
                metric_reward = gr.Textbox(value="—",        label="Last Reward", interactive=False)
                metric_mean   = gr.Textbox(value="—",        label="Mean Reward", interactive=False)
                metric_action = gr.Textbox(value="—",        label="Last Action", interactive=False)

            status_out = gr.Textbox(
                label="Status",
                value="Select a task below and hit Reset to start",
                interactive=False,
                elem_classes=["status-bar"],
            )

            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=270):
                    gr.HTML('<div class="sec-h">Configuration</div>')
                    task_dd = gr.Dropdown(
                        choices=["suppression", "deterioration", "triage"],
                        value="suppression",
                        label="Task",
                        info="Easy → Medium → Hard",
                    )
                    seed_num = gr.Number(value=42, label="Random Seed", precision=0)
                    agent_radio = gr.Radio(
                        choices=_agent_choices,
                        value=_agent_default,
                        label="Agent Mode",
                        info=(
                            "HF_TOKEN detected — LLM active"
                            if _llm_available
                            else "LLM requires HF_TOKEN (not set) — using Rule-Based"
                        ),
                    )
                    reset_btn = gr.Button(
                        "🔄  Reset Environment",
                        variant="primary", size="lg",
                        elem_classes=["btn-reset"],
                    )

                    gr.HTML('<div class="sec-h" style="margin-top:18px">Action</div>')
                    gr.Markdown(
                        "_LLM and Rule-Based agents decide automatically.  \n"
                        "Action controls only appear in Manual mode._"
                    )

                    with gr.Column(visible=False) as single_action_row:
                        action_radio = gr.Radio(
                            choices=["0 — 😴 Ignore", "1 — 🔍 Verify", "2 — 🚨 Alert"],
                            value="1 — 🔍 Verify",
                            label="Single-Patient Action",
                        )

                    with gr.Column(visible=False) as triage_action_row:
                        triage_txt = gr.Textbox(
                            value="1,0,2,0",
                            label="Triage Actions [P0, P1, P2, P3]",
                            info="0=Ignore  1=Verify  2=Alert",
                        )

                    step_btn = gr.Button(
                        "▶  Next Step",
                        variant="secondary", size="lg",
                        interactive=False,
                        elem_classes=["btn-step"],
                    )

                with gr.Column(scale=2):
                    obs_out = gr.Textbox(
                        label="Patient Monitor",
                        lines=15, max_lines=22,
                        interactive=False,
                        elem_classes=["vitals-box"],
                        placeholder="Reset to see live patient data...",
                    )
                    log_out = gr.Textbox(
                        label="Episode Log",
                        lines=9,
                        interactive=False,
                        elem_classes=["log-box"],
                        placeholder="Step decisions appear here...",
                    )

            gr.HTML('<hr style="border:none;border-top:2px solid #e5e7eb;margin:22px 0">')
            gr.HTML('<div class="sec-h">Full Inference Pipeline</div>')
            gr.Markdown(
                "Runs `inference.py` end-to-end (all 3 tasks, LLM agent, ~5–15 min). "
                "Requires `HF_TOKEN`."
            )
            # FIX 1: run_all_btn has no interactive=False anywhere — always clickable
            run_all_btn = gr.Button(
                "🏃  Run Full Inference (All 3 Tasks)",
                variant="stop", elem_classes=["btn-run"],
            )
            full_log = gr.Textbox(
                label="Inference Output",
                lines=20, interactive=False,
                elem_classes=["full-log"],
                placeholder="[START] / [STEP] / [END] logs appear here...",
            )

        # ── TAB 2: HOW IT WORKS ──────────────────────────────────
        with gr.Tab("💡 How It Works"):
            with gr.Row():
                with gr.Column():
                    gr.HTML(HOW_INSIGHT_HTML)
                    gr.Markdown("""
### Three Tasks — Easy to Hard

| Task | Patients | Challenge |
|------|:--------:|-----------|
| 🟢 Suppression | 1 hypertensive | His high BP is *normal* — don't alarm |
| 🟡 Deterioration | 1 sepsis patient | Catch the trend before crisis |
| 🔴 Triage | 4 simultaneous | Rank by urgency — ordering matters |

### Action Space `Discrete(3)`

| Code | Action | Use when |
|:----:|--------|----------|
| `0` | 😴 IGNORE | Vitals match this patient's normal baseline |
| `1` | 🔍 VERIFY | Mild deviation — alert the nurse |
| `2` | 🚨 ALERT | Genuine emergency — call the doctor now |

### Observation Fields (10 per patient)

`heart_rate` · `systolic_bp` · `diastolic_bp` · `spo2` · `respiratory_rate` · `temperature`
— all normalized 0–1 relative to this patient's 3-hour rolling baseline

`baseline_delta` — combined deviation score (0 = normal, 1 = extreme)

`hours_observed` · `activity` (0=resting 1=eating 2=walking 3=distressed 4=falling)

`vitals_history` — last 10 readings for trend detection
                    """)

                with gr.Column():
                    gr.HTML(SCORING_HTML)
                    gr.Markdown(f"""
### LLM Agent Design

Task-specific system prompts + 4-turn sliding conversation window for pseudo-online learning.

```
System Prompt (task rules + thresholds)
+ Vitals trend table (last 8 readings)
+ Recent decisions and their rewards
→ LLM picks: IGNORE / VERIFY / ALERT
→ Falls back to rule-based on any error
```

**Models in use:**
- Suppression → `Qwen/Qwen2.5-7B-Instruct` (fast, cheap)
- Deterioration → `{MODEL_NAME}`
- Triage → `{MODEL_NAME}`

**LLM:** {"✅ Connected" if _llm_available else "❌ Offline — set `HF_TOKEN`"}
&nbsp;·&nbsp; **Endpoint:** `{API_BASE_URL}`
                    """)

    # ── Event wiring ──────────────────────────────────────────────
    # FIX 1: _shared no longer contains run_all_btn
    _shared = [
        status_out, obs_out,
        metric_action, metric_reward, metric_mean, metric_step,
        log_out,
        step_btn,                      # only step_btn toggled, not run_all_btn
        single_action_row, triage_action_row,
    ]

    reset_btn.click(demo_reset, [task_dd, seed_num, agent_radio], _shared)
    step_btn.click(demo_step, [action_radio, triage_txt, agent_radio], _shared)
    agent_radio.change(on_agent_change, [agent_radio, task_dd], [single_action_row, triage_action_row])
    task_dd.change(on_task_change, [task_dd, agent_radio], [single_action_row, triage_action_row])
    run_all_btn.click(demo_run_all, [agent_radio], [full_log])

    # Hidden API wrappers (Gradio API endpoint exposure)
    with gr.Row(visible=False):
        _t  = gr.Textbox(value="suppression")
        _s  = gr.Number(value=42)
        _ro = gr.Textbox()
        _ai = gr.Textbox(value="1")
        _ao = gr.Textbox()
        _go = gr.Textbox()
        _ho = gr.Textbox()
    _t.submit(reset_env,    [_t, _s],  [_ro])
    _ai.submit(step_env,    [_ai],     [_ao])
    _go.submit(get_state,   [],        [_go])
    _ho.submit(health_check,[],        [_ho])

# ── Launch ────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = "0.0.0.0" if (os.getenv("SPACE_ID") or os.getenv("DOCKER")) else "127.0.0.1"
    app.launch(server_name=host, server_port=7860)