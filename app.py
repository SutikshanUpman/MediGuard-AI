"""
MediGuard-AI — Gradio App for HuggingFace Spaces / OpenEnv validation.

Exposes the MediGuardEnv as a Gradio interface with API endpoints:
  POST /api/reset   — reset environment
  POST /api/step    — take an action
  POST /api/state   — get current state
"""

import json
import gradio as gr
from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  Global environment instance                                        #
# ------------------------------------------------------------------ #

_env = None


# ------------------------------------------------------------------ #
#  API functions                                                      #
# ------------------------------------------------------------------ #

def reset_env(task: str = "suppression", seed: int = 42):
    """Reset the environment and return the first observation."""
    global _env

    if task not in ("suppression", "deterioration", "triage"):
        return json.dumps({"error": f"Unknown task: {task}"})

    _env = MediGuardEnv(task=task, seed=int(seed))
    obs = _env.reset()

    return json.dumps({
        "observation": obs,
        "info": {"task": task, "seed": int(seed)},
    }, default=str)


def step_env(action: str):
    """Execute one step. Action is int or comma-separated ints for triage."""
    global _env

    if _env is None:
        return json.dumps({"error": "Environment not initialized. Call reset first."})

    try:
        if "," in action:
            parsed = [int(a.strip()) for a in action.split(",")]
        else:
            parsed = int(action.strip())

        obs, reward, done, info = _env.step(parsed)

        return json.dumps({
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


def get_state():
    """Return the current environment state."""
    global _env

    if _env is None:
        return json.dumps({"error": "Environment not initialized. Call reset first."})

    return json.dumps(_env.state(), default=str)


def health_check():
    """Health check endpoint."""
    return json.dumps({"status": "ok"})


# ------------------------------------------------------------------ #
#  Interactive demo functions (for the Gradio UI)                     #
# ------------------------------------------------------------------ #

def demo_reset(task, seed):
    """Reset and return formatted observation for the UI."""
    result = json.loads(reset_env(task, int(seed)))
    if "error" in result:
        return f"❌ Error: {result['error']}", ""

    obs = result["observation"]
    if isinstance(obs, list):
        lines = []
        for i, p in enumerate(obs):
            lines.append(
                f"Patient {i}: HR={p['heart_rate']:.3f}  SpO2={p['spo2']:.3f}  "
                f"Temp={p['temperature']:.3f}  Delta={p['baseline_delta']:.3f}  "
                f"Activity={p['activity']}"
            )
        obs_str = "\n".join(lines)
    else:
        obs_str = (
            f"HR={obs['heart_rate']:.3f}  SpO2={obs['spo2']:.3f}  "
            f"Temp={obs['temperature']:.3f}  Delta={obs['baseline_delta']:.3f}  "
            f"Activity={obs['activity']}  Hours={obs['hours_observed']:.2f}"
        )

    return f"✅ Environment reset | Task: {task} | Seed: {seed}", obs_str


def demo_step(action_str):
    """Take a step and return formatted result for the UI."""
    if not action_str:
        return "❌ Enter an action", "", ""

    result = json.loads(step_env(action_str))
    if "error" in result:
        return f"❌ Error: {result['error']}", "", ""

    obs = result["observation"]
    reward = result["reward"]
    done = result["done"]
    info = result["info"]

    if isinstance(obs, list):
        lines = []
        for i, p in enumerate(obs):
            lines.append(
                f"Patient {i}: HR={p['heart_rate']:.3f}  SpO2={p['spo2']:.3f}  "
                f"Temp={p['temperature']:.3f}  Delta={p['baseline_delta']:.3f}  "
                f"Activity={p['activity']}"
            )
        obs_str = "\n".join(lines)
    else:
        obs_str = (
            f"HR={obs['heart_rate']:.3f}  SpO2={obs['spo2']:.3f}  "
            f"Temp={obs['temperature']:.3f}  Delta={obs['baseline_delta']:.3f}  "
            f"Activity={obs['activity']}  Hours={obs['hours_observed']:.2f}"
        )

    status = f"Step {info['step']} | Reward: {reward:.3f} | Done: {done}"
    return status, obs_str, json.dumps(info)


# ------------------------------------------------------------------ #
#  Gradio Interface                                                   #
# ------------------------------------------------------------------ #

with gr.Blocks(
    title="MediGuard-AI Environment",
    theme=gr.themes.Soft(),
) as app:

    gr.Markdown("""
    # 🏥 MediGuard-AI — ICU Patient Monitoring Environment

    An OpenEnv-compliant RL environment for training AI agents to monitor ICU patients.

    **Tasks:** Suppression (Easy) → Deterioration (Medium) → Triage (Hard)

    **Actions:** 0 = Ignore, 1 = Verify, 2 = Alert
    """)

    with gr.Row():
        with gr.Column():
            task_input = gr.Dropdown(
                choices=["suppression", "deterioration", "triage"],
                value="suppression",
                label="Task",
            )
            seed_input = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

        with gr.Column():
            action_input = gr.Textbox(
                value="1",
                label="Action (int or comma-separated for triage, e.g. 1,0,2,0)",
            )
            step_btn = gr.Button("▶️ Step", variant="secondary")

    status_output = gr.Textbox(label="Status", interactive=False)
    obs_output = gr.Textbox(label="Observation", lines=5, interactive=False)
    info_output = gr.Textbox(label="Info", interactive=False)

    reset_btn.click(
        fn=demo_reset,
        inputs=[task_input, seed_input],
        outputs=[status_output, obs_output],
    )

    step_btn.click(
        fn=demo_step,
        inputs=[action_input],
        outputs=[status_output, obs_output, info_output],
    )

    gr.Markdown("""
    ---
    ### API Endpoints (for programmatic access)

    | Function | Input | Description |
    |----------|-------|-------------|
    | `/api/reset_env` | `task` (str), `seed` (int) | Reset environment |
    | `/api/step_env` | `action` (str) | Take a step |
    | `/api/get_state` | — | Get current state |
    | `/api/health_check` | — | Health check |

    ### Environment Info
    - **Episode length:** 360 steps (6 simulated hours)
    - **Observation:** 10 fields (6 vitals + baseline_delta + hours + activity + history)
    - **Reward:** 0.0–1.0 (based on action × patient condition)
    - **Seed:** 42 for reproducibility
    """)


# ------------------------------------------------------------------ #
#  Launch                                                             #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import os
    # Use 0.0.0.0 in Docker/HF Spaces, 127.0.0.1 for local dev
    host = "0.0.0.0" if os.getenv("SPACE_ID") or os.getenv("DOCKER") else "127.0.0.1"
    app.launch(server_name=host, server_port=7860)
