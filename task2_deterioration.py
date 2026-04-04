"""
Task 2: Deterioration Grader (Medium)
=====================================
Grades how well the agent detects slow patient deterioration (sepsis).

Three-phase scoring:
  - VERIFY during BORDERLINE (early detection): up to 0.3 bonus
  - ALERT during EMERGENCY (before step 300): up to 0.5
  - Any response during EMERGENCY: up to 0.2
"""

from reward_function import Action, PatientCondition


def grade_deterioration(stats: dict) -> float:
    """
    Grade the deterioration task.

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

    first_alert_during_emergency = None
    first_verify_during_emergency = None
    first_verify_during_borderline = None
    any_response_during_emergency = None

    for i, (action, condition) in enumerate(zip(action_history, condition_history)):
        step = i + 1  # 1-indexed

        # Track actions during BORDERLINE conditions
        if condition == PatientCondition.BORDERLINE:
            if action in (Action.VERIFY, Action.ALERT) and first_verify_during_borderline is None:
                first_verify_during_borderline = step

        # Track actions during EMERGENCY / DRUG_MASKED conditions
        if condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED):
            if action == Action.ALERT and first_alert_during_emergency is None:
                first_alert_during_emergency = step
            if action == Action.VERIFY and first_verify_during_emergency is None:
                first_verify_during_emergency = step
            if action in (Action.ALERT, Action.VERIFY) and any_response_during_emergency is None:
                any_response_during_emergency = step

    # Phase 1: Early detection during BORDERLINE (up to 0.3)
    early_bonus = 0.0
    if first_verify_during_borderline is not None:
        if first_verify_during_borderline <= 200:
            early_bonus = 0.3
        elif first_verify_during_borderline <= 280:
            early_bonus = 0.3 * (280 - first_verify_during_borderline) / 80.0
        else:
            early_bonus = 0.05  # some credit for noticing at all

    # Phase 2: Alert during EMERGENCY (up to 0.5)
    alert_score = 0.0
    if first_alert_during_emergency is not None:
        if first_alert_during_emergency <= 300:
            alert_score = 0.5
        elif first_alert_during_emergency <= 340:
            alert_score = 0.5 * (340 - first_alert_during_emergency) / 40.0

    # Phase 3: Any response during EMERGENCY (up to 0.2)
    response_score = 0.0
    if any_response_during_emergency is not None:
        if any_response_during_emergency <= 300:
            response_score = 0.2
        elif any_response_during_emergency <= 340:
            response_score = 0.2 * (340 - any_response_during_emergency) / 40.0

    score = early_bonus + alert_score + response_score
    return max(0.0, min(1.0, score))
