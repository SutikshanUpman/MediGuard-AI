"""
Reward Function for MediGuard-AI
================================
Maps (action × patient_condition) to a reward signal with:
  - Base reward table
  - Alarm fatigue modifier
  - Personalization bonus
"""

from enum import Enum


class Action(Enum):
    IGNORE = 0
    VERIFY = 1
    ALERT  = 2


class PatientCondition(Enum):
    STABLE      = "stable"
    BORDERLINE  = "borderline"
    EMERGENCY   = "emergency"
    DRUG_MASKED = "drug_masked"


# Base reward table: REWARD_TABLE[action][condition]
REWARD_TABLE = {
    Action.ALERT: {
        PatientCondition.EMERGENCY:   +1.0,
        PatientCondition.BORDERLINE:  +0.2,
        PatientCondition.STABLE:      -0.5,
        PatientCondition.DRUG_MASKED: +1.0,
    },
    Action.VERIFY: {
        PatientCondition.EMERGENCY:   -0.8,
        PatientCondition.BORDERLINE:  +0.4,
        PatientCondition.STABLE:      +0.1,
        PatientCondition.DRUG_MASKED: -0.5,
    },
    Action.IGNORE: {
        PatientCondition.EMERGENCY:   -2.0,
        PatientCondition.BORDERLINE:  -0.1,
        PatientCondition.STABLE:      +0.2,
        PatientCondition.DRUG_MASKED: -2.0,
    },
}

# Alarm fatigue settings
FATIGUE_WINDOW = 30          # look at last 30 steps
FATIGUE_THRESHOLD = 5        # more than 5 alerts in the window
FATIGUE_MULTIPLIER = 0.6     # reduce reward to 60%

# Personalization settings
PERSONALIZATION_STEP = 200   # kicks in after this many steps
PERSONALIZATION_BONUS = 0.2  # bonus for correctly ignoring known-normal


class RewardFunction:
    """
    Stateful reward calculator.

    Tracks action history for alarm fatigue and personalization bonuses.
    """

    def __init__(self):
        self.action_history = []
        self.condition_history = []
        self.step_count = 0

    def reset(self):
        """Clear all history for a new episode."""
        self.action_history = []
        self.condition_history = []
        self.step_count = 0

    def compute(self, action: Action, condition: PatientCondition) -> float:
        """
        Compute the reward for a single (action, condition) pair.

        Parameters
        ----------
        action : Action
            The agent's action (IGNORE, VERIFY, ALERT).
        condition : PatientCondition
            The patient's current condition.

        Returns
        -------
        float
            The reward value (can be negative).
        """
        self.step_count += 1
        self.action_history.append(action)
        self.condition_history.append(condition)

        # 1. Base reward from the table
        base_reward = REWARD_TABLE[action][condition]

        # 2. Alarm fatigue modifier
        fatigue_modifier = 1.0
        if len(self.action_history) >= FATIGUE_WINDOW:
            recent_alerts = sum(
                1 for a in self.action_history[-FATIGUE_WINDOW:]
                if a == Action.ALERT
            )
            if recent_alerts > FATIGUE_THRESHOLD:
                fatigue_modifier = FATIGUE_MULTIPLIER

        # 3. Personalization bonus
        personalization = 0.0
        if (self.step_count > PERSONALIZATION_STEP
                and action == Action.IGNORE
                and condition == PatientCondition.STABLE):
            personalization = PERSONALIZATION_BONUS

        # Final reward
        reward = base_reward * fatigue_modifier + personalization
        return reward

    def get_stats(self) -> dict:
        """Return episode statistics for graders."""
        total_alerts = sum(1 for a in self.action_history if a == Action.ALERT)
        total_verifies = sum(1 for a in self.action_history if a == Action.VERIFY)
        total_ignores = sum(1 for a in self.action_history if a == Action.IGNORE)

        return {
            "total_steps": self.step_count,
            "total_alerts": total_alerts,
            "total_verifies": total_verifies,
            "total_ignores": total_ignores,
            "action_history": list(self.action_history),
            "condition_history": list(self.condition_history),
        }
