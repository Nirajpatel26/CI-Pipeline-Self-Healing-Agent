# tools/pipeline_inspector.py — PipelineStateInspector AutoGen custom tool

from typing import Dict, List, Tuple, Optional
import numpy as np
from config import (
    RECOVERY_PROBS, DEFAULT_RECOVERY_PROB,
    STAGES, ACTIONS, ACTION_IDX, STAGE_CRITICALITY
)


class PipelineStateInspector:
    """
    AutoGen custom tool called at every episode step.

    Provides structured intelligence to the RLRecoveryAgent:
    - recoverability_score  : probability the best action will succeed
    - last_safe_checkpoint  : stage to revert to if needed
    - top_2_recommended     : top-2 (action, q_value) from Q-table
    - risk_level            : LOW / MEDIUM / HIGH
    - safe_for_autonomous   : advisory hint (True unless risk is HIGH)

    Note: ``safe_for_autonomous_recovery`` is **advisory only**. Earlier
    versions of ``_select_action`` used it as a hard gate that forced
    ``escalate`` on HIGH-risk states; that starved the Q-table of signal
    and produced the 0% RecovRate% observed in prior experiments. The RL
    agent (and, in integrated mode, the MonitorLLM / RLRecoveryLLM) now
    makes the final call — this field is just one more feature they see.
    """

    def __init__(self, agent=None):
        """
        Args:
            agent: a QLearningAgent or UCBAgent instance (for Q-table queries).
                   Can be None in baseline/rule-based modes.
        """
        self.agent = agent

    def inspect(self, state: Dict, state_idx: int) -> Dict:
        """
        Main entry point. Returns a structured inspection report.

        Args:
            state     : current state dict {stage, error_type, attempt_num, last_action}
            state_idx : encoded integer state index
        Returns:
            dict with recoverability_score, last_safe_checkpoint,
                 top_2_recommended_actions, risk_level, safe_for_autonomous_recovery
        """
        stage = state["stage"]
        error_type = state["error_type"]
        attempt_num = state["attempt_num"]

        # ── Recoverability Score ────────────────────────────────────────────
        best_prob = max(
            RECOVERY_PROBS.get((stage, error_type, a), DEFAULT_RECOVERY_PROB)
            for a in ACTIONS
        )
        recoverability_score = round(best_prob, 3)

        # ── Last Safe Checkpoint ────────────────────────────────────────────
        stage_order = STAGES.index(stage)
        last_safe_checkpoint = STAGES[max(0, stage_order - 1)]

        # ── Top-2 Recommended Actions from Q-table ──────────────────────────
        if self.agent is not None:
            top_2 = self.agent.top_k_actions(state_idx, k=2)
            top_2_recommended = [
                {"action": a, "q_value": round(q, 4)} for a, q in top_2
            ]
        else:
            # Fall back to probability-based ranking when no agent is available
            ranked = sorted(
                ACTIONS,
                key=lambda a: RECOVERY_PROBS.get(
                    (stage, error_type, a), DEFAULT_RECOVERY_PROB
                ),
                reverse=True,
            )
            top_2_recommended = [
                {
                    "action": a,
                    "q_value": RECOVERY_PROBS.get(
                        (stage, error_type, a), DEFAULT_RECOVERY_PROB
                    ),
                }
                for a in ranked[:2]
            ]

        # ── Risk Level ──────────────────────────────────────────────────────
        risk_level = self._compute_risk(stage, error_type, attempt_num)

        # ── Autonomous Recovery Safety ──────────────────────────────────────
        safe_for_autonomous = risk_level != "HIGH"

        return {
            "recoverability_score": recoverability_score,
            "last_safe_checkpoint": last_safe_checkpoint,
            "top_2_recommended_actions": top_2_recommended,
            "risk_level": risk_level,
            "safe_for_autonomous_recovery": safe_for_autonomous,
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_risk(stage: str, error_type: str, attempt_num: int) -> str:
        """
        Risk escalation rules:
        - security_scan stage               → starts at MEDIUM
        - attempt_num >= 2                  → escalate to HIGH
        - error_type in {secret_exposed, vuln_detected} → HIGH
        - otherwise                         → LOW
        """
        if error_type in ("secret_exposed", "vuln_detected"):
            return "HIGH"
        if attempt_num >= 2:
            return "HIGH"
        if stage == "security_scan":
            return "MEDIUM"
        if STAGE_CRITICALITY.get(stage, 1.0) >= 1.3:
            return "MEDIUM"
        return "LOW"
