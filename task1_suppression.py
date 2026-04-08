"""
Task 1: Suppression Grader (Easy)
=================================
Grades how well the agent suppresses false alarms for a hypertensive patient
while still catching real emergencies.

Uses F1 harmonic mean of sensitivity and specificity:
  - sensitivity = correctly responded to emergencies / total emergencies
    (ALERT counts as full TP, VERIFY counts as 0.5 TP — must commit to ALERT)
  - specificity = correctly ignored stable periods / total stable periods
    (VERIFY counts as 0.7 FP — spamming VERIFY is penalized more heavily)
  - score = 2 × sensitivity × specificity / (sensitivity + specificity)

Calibrated so that:
  - Rule-based agent (mostly VERIFY/IGNORE) scores ~0.50
  - A smart LLM/RL agent that correctly ALERTs emergencies and IGNOREs stable → ~0.80+
  - Pure-IGNORE agent → sensitivity=0 → F1=0.0
  - Pure-ALERT agent → specificity=0 → F1=0.0
"""

from reward_function import Action, PatientCondition


def grade_suppression(stats: dict) -> float:
    """
    Grade the suppression task using F1 harmonic mean.

    Parameters
    ----------
    stats : dict
        Episode statistics from RewardFunction.get_stats().

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    action_history = stats["action_history"]
    condition_history = stats["condition_history"]
    total_steps = stats["total_steps"]

    if total_steps == 0:
        return 0.0

    # Count outcomes
    true_positives  = 0.0   # Alert/Verify during Emergency
    false_negatives = 0.0   # Ignore during Emergency
    true_negatives  = 0.0   # Ignore during Stable
    false_positives = 0.0   # Alert/Verify during Stable

    for action, condition in zip(action_history, condition_history):
        is_emergency = condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED)

        if is_emergency:
            if action == Action.ALERT:
                true_positives += 1.0        # Full credit — committed to ALERT
            elif action == Action.VERIFY:
                true_positives += 0.5        # Partial credit — noticed but didn't escalate
                false_negatives += 0.5       # Partial miss — should have ALERTed
            else:  # IGNORE during emergency
                false_negatives += 1.0

        else:  # STABLE or BORDERLINE
            if action == Action.IGNORE:
                true_negatives += 1.0
            elif action == Action.ALERT:
                false_positives += 1.0       # Fully wrong
            else:  # VERIFY during stable
                false_positives += 0.7       # Raised from 0.3 — VERIFY is not a safe default

    # Sensitivity: how well did we catch real emergencies?
    total_emergencies = true_positives + false_negatives
    if total_emergencies > 0:
        sensitivity = true_positives / total_emergencies
    else:
        # No emergencies in this episode — cap sensitivity so agent can't get free marks.
        # Without emergencies, the agent has no way to prove it can handle them.
        sensitivity = 0.5

    # Specificity: how well did we suppress false alarms?
    total_stable = true_negatives + false_positives
    if total_stable > 0:
        specificity = true_negatives / total_stable
    else:
        specificity = 1.0

    # F1 harmonic mean
    if (sensitivity + specificity) > 0:
        f1 = 2.0 * sensitivity * specificity / (sensitivity + specificity)
    else:
        f1 = 0.0

    return max(0.0, min(1.0, f1))