# pipeline_simulator.py — Stochastic CI/CD pipeline failure generator

import random
from typing import Dict, Tuple, Optional
from config import (
    STAGES, STAGE_ERRORS, ACTIONS, STAGE_IDX, ERROR_IDX, ACTION_IDX,
    RECOVERY_PROBS, DEFAULT_RECOVERY_PROB, RANDOM_SEED,
    N_STAGES, N_ERRORS, N_ATTEMPTS, N_LAST_ACTIONS
)


class PipelineSimulator:
    """
    Simulates a stochastic CI/CD pipeline with 5 stages.
    Generates failures and applies recovery actions with realistic success
    probabilities drawn from RECOVERY_PROBS lookup table.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)

    # ── Failure Generation ──────────────────────────────────────────────────

    def generate_failure(self) -> Dict:
        """
        Sample a random (stage, error_type) failure.
        Returns a state dict with attempt_num=0 and last_action=None.
        """
        stage = self.rng.choice(STAGES)
        error_type = self.rng.choice(STAGE_ERRORS[stage])
        return {
            "stage": stage,
            "error_type": error_type,
            "attempt_num": 0,
            "last_action": None,
        }

    # ── Action Application ──────────────────────────────────────────────────

    def apply_action(
        self, state: Dict, action: str
    ) -> Tuple[bool, Dict]:
        """
        Apply a recovery action to the current pipeline state.

        Returns:
            (success, next_state)
            - success: True if the pipeline recovered
            - next_state: updated state dict (attempt_num incremented, last_action set)
        """
        assert action in ACTIONS, f"Unknown action: {action}"

        prob = RECOVERY_PROBS.get(
            (state["stage"], state["error_type"], action),
            DEFAULT_RECOVERY_PROB,
        )
        success = self.rng.random() < prob

        next_state = {
            "stage": state["stage"],
            "error_type": state["error_type"],
            "attempt_num": state["attempt_num"] + 1,
            "last_action": action,
        }
        return success, next_state

    # ── State Encoding ──────────────────────────────────────────────────────

    @staticmethod
    def encode_state(state: Dict) -> int:
        """
        Encode a state dict into a single integer index for Q-table lookup.
        Index = stage * (N_ERRORS * N_ATTEMPTS * N_LAST_ACTIONS)
              + error * (N_ATTEMPTS * N_LAST_ACTIONS)
              + attempt * N_LAST_ACTIONS
              + last_action_plus1  (0 = no previous action)
        """
        stage_i = STAGE_IDX[state["stage"]]
        error_i = ERROR_IDX[state["error_type"]]
        attempt_i = min(state["attempt_num"], N_ATTEMPTS - 1)
        last_action = state.get("last_action")
        last_action_i = (ACTION_IDX[last_action] + 1) if last_action else 0

        idx = (
            stage_i * (N_ERRORS * N_ATTEMPTS * N_LAST_ACTIONS)
            + error_i * (N_ATTEMPTS * N_LAST_ACTIONS)
            + attempt_i * N_LAST_ACTIONS
            + last_action_i
        )
        return idx

    @staticmethod
    def decode_state(idx: int) -> Dict:
        """Reverse of encode_state — returns a state dict from an integer index."""
        last_action_i = idx % N_LAST_ACTIONS
        idx //= N_LAST_ACTIONS
        attempt_i = idx % N_ATTEMPTS
        idx //= N_ATTEMPTS
        error_i = idx % N_ERRORS
        stage_i = idx // N_ERRORS

        from config import STAGES, ALL_ERRORS, ACTIONS
        stage = STAGES[stage_i]
        error_type = ALL_ERRORS[error_i]
        last_action = ACTIONS[last_action_i - 1] if last_action_i > 0 else None
        return {
            "stage": stage,
            "error_type": error_type,
            "attempt_num": attempt_i,
            "last_action": last_action,
        }

    # ── Utility ─────────────────────────────────────────────────────────────

    def get_success_prob(self, stage: str, error_type: str, action: str) -> float:
        """Return the known recovery probability for a (stage, error, action) triple."""
        return RECOVERY_PROBS.get((stage, error_type, action), DEFAULT_RECOVERY_PROB)
