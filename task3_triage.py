"""
Task 3: Triage Grader (Hard)
=============================
Grades multi-patient triage with 3 scoring components:
  - 50% F1 score (sensitivity + specificity)
  - 30% Masked detection (catching emergencies during activity)
  - 20% Triage priority (giving sickest patient most attention)
"""

from reward_function import Action, PatientCondition


def grade_triage(stats_list: list) -> float:
    """
    Grade the triage task.

    Parameters
    ----------
    stats_list : list[dict]
        List of per-patient episode statistics from RewardFunction.get_stats().

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    if not stats_list or len(stats_list) == 0:
        return 0.0

    num_patients = len(stats_list)

    # ── Component 1: F1 Score (50%) ──
    # For each patient at each step: was the action appropriate?
    tp = 0  # true positive: alert/verify when emergency
    tn = 0  # true negative: ignore when stable
    fp = 0  # false positive: alert when stable
    fn = 0  # false negative: ignore when emergency

    for stats in stats_list:
        for action, condition in zip(stats["action_history"], stats["condition_history"]):
            is_emergency = condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED)
            is_action = action in (Action.ALERT, Action.VERIFY)

            if is_emergency and is_action:
                tp += 1
            elif not is_emergency and not is_action:
                tn += 1
            elif not is_emergency and is_action:
                fp += 1
            elif is_emergency and not is_action:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ── Component 2: Masked Detection (30%) ──
    # Did the agent detect emergencies even during high-activity (ambulating)?
    # We don't have activity in stats, so approximate: check if ALERT was used during EMERGENCY
    masked_detected = 0
    masked_total = 0

    for stats in stats_list:
        for action, condition in zip(stats["action_history"], stats["condition_history"]):
            if condition == PatientCondition.EMERGENCY:
                masked_total += 1
                if action == Action.ALERT:
                    masked_detected += 1

    masked_score = masked_detected / masked_total if masked_total > 0 else 1.0

    # ── Component 3: Triage Priority (20%) ──
    # Did the agent give MORE attention (alerts+verifies) to sicker patients?
    attention_per_patient = []
    emergency_count_per_patient = []

    for stats in stats_list:
        attention = sum(
            1 for a in stats["action_history"]
            if a in (Action.ALERT, Action.VERIFY)
        )
        emergencies = sum(
            1 for c in stats["condition_history"]
            if c in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED)
        )
        attention_per_patient.append(attention)
        emergency_count_per_patient.append(emergencies)

    # Priority score: correlation between attention and emergencies
    max_emergencies = max(emergency_count_per_patient) if emergency_count_per_patient else 0
    if max_emergencies > 0 and sum(attention_per_patient) > 0:
        # Find the patient with most emergencies
        sickest_idx = emergency_count_per_patient.index(max_emergencies)
        sickest_attention = attention_per_patient[sickest_idx]
        total_attention = sum(attention_per_patient)

        # What fraction of total attention went to the sickest patient?
        priority_score = sickest_attention / total_attention if total_attention > 0 else 0.0
        # Scale: if sickest gets >50% of attention → full marks
        priority_score = min(1.0, priority_score * 2)
    else:
        priority_score = 0.5  # no emergencies → baseline

    # ── Weighted sum ──
    score = 0.5 * f1 + 0.3 * masked_score + 0.2 * priority_score

    return max(0.0, min(1.0, score))
