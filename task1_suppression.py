"""
Task 1: Suppression Grader (Easy)
=================================
Grades how well the agent suppresses false alarms for a hypertensive patient.

Score = 1.0 when false alarm rate < 5%
Score = 0.0 when false alarm rate >= 60%
Linear interpolation in between, with a penalty for missed real emergencies.
"""

from reward_function import Action, PatientCondition


def grade_suppression(stats: dict) -> float:
    """
    Grade the suppression task.

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

    # Count false alarms: ALERT when condition is STABLE or BORDERLINE
    false_alarms = 0
    real_emergencies = 0
    missed_emergencies = 0

    for action, condition in zip(action_history, condition_history):
        if condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED):
            real_emergencies += 1
            if action == Action.IGNORE:
                missed_emergencies += 1
        else:
            if action == Action.ALERT:
                false_alarms += 1

    # False alarm rate (among non-emergency steps)
    non_emergency_steps = total_steps - real_emergencies
    if non_emergency_steps > 0:
        false_alarm_rate = false_alarms / non_emergency_steps
    else:
        false_alarm_rate = 0.0

    # Linear score: 5% → 1.0, 60% → 0.0
    if false_alarm_rate <= 0.05:
        score = 1.0
    elif false_alarm_rate >= 0.60:
        score = 0.0
    else:
        score = 1.0 - (false_alarm_rate - 0.05) / (0.60 - 0.05)

    # Penalty for missed real emergencies
    if real_emergencies > 0:
        miss_rate = missed_emergencies / real_emergencies
        score *= (1.0 - 0.5 * miss_rate)  # up to 50% penalty

    return max(0.0, min(1.0, score))
