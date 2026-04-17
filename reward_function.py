# reward_function.py — Reward computation for CI pipeline recovery

from config import STAGE_CRITICALITY


def compute_reward(
    state: dict,
    action: str,
    success: bool,
) -> float:
    """
    Compute scalar reward ∈ [-10, +10] for a recovery attempt.

    Reward logic:
    - Success on attempt 0 → +10
    - Success on attempt 1 → +7
    - Success on attempt 2 → +5
    - Success on attempt 3+ → +3
    - Failed attempt      → -2 × stage_criticality
    - skip_stage on security_scan → additional -5 penalty (HIGH risk)
    - escalate            → -8 × stage_criticality
    """
    stage = state["stage"]
    criticality = STAGE_CRITICALITY.get(stage, 1.0)
    attempt_num = state["attempt_num"]  # number of attempts BEFORE this action

    # Base reward
    if success:
        if attempt_num == 0:
            reward = 10.0
        elif attempt_num == 1:
            reward = 7.0
        elif attempt_num == 2:
            reward = 5.0
        else:
            reward = 3.0
    else:
        reward = -2.0 * criticality

    # Action-specific penalties (applied on top of base)
    if action == "escalate":
        reward -= 8.0 * criticality

    if action == "skip_stage" and stage == "security_scan":
        reward -= 5.0  # HIGH risk: skipping security checks

    # Clamp to [-10, +10]
    return max(-10.0, min(10.0, reward))
