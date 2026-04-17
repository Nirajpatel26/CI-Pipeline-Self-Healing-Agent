# tests/test_simulator.py — Tests for PipelineSimulator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline_simulator import PipelineSimulator
from config import STAGES, STAGE_ERRORS, ACTIONS, RECOVERY_PROBS


class TestPipelineSimulator:

    def setup_method(self):
        self.sim = PipelineSimulator(seed=42)

    # ── generate_failure ────────────────────────────────────────────────

    def test_generate_failure_keys(self):
        state = self.sim.generate_failure()
        assert set(state.keys()) == {"stage", "error_type", "attempt_num", "last_action"}

    def test_generate_failure_valid_stage(self):
        for _ in range(50):
            state = self.sim.generate_failure()
            assert state["stage"] in STAGES

    def test_generate_failure_valid_error(self):
        for _ in range(50):
            state = self.sim.generate_failure()
            assert state["error_type"] in STAGE_ERRORS[state["stage"]]

    def test_generate_failure_initial_attempt(self):
        state = self.sim.generate_failure()
        assert state["attempt_num"] == 0
        assert state["last_action"] is None

    # ── apply_action ────────────────────────────────────────────────────

    def test_apply_action_increments_attempt(self):
        state = {"stage": "lint", "error_type": "syntax_error", "attempt_num": 0, "last_action": None}
        _, next_state = self.sim.apply_action(state, "retry")
        assert next_state["attempt_num"] == 1
        assert next_state["last_action"] == "retry"

    def test_apply_action_returns_bool(self):
        state = {"stage": "deploy", "error_type": "rollback_needed", "attempt_num": 0, "last_action": None}
        success, _ = self.sim.apply_action(state, "revert")
        assert isinstance(success, bool)

    def test_revert_on_rollback_needed_high_success(self):
        """deploy/rollback_needed + revert = 96% — should succeed most of the time."""
        state = {"stage": "deploy", "error_type": "rollback_needed", "attempt_num": 0, "last_action": None}
        successes = sum(
            self.sim.apply_action(state, "revert")[0] for _ in range(200)
        )
        assert successes > 150, f"Expected >150/200 successes, got {successes}"

    def test_retry_on_rollback_needed_low_success(self):
        """deploy/rollback_needed + retry = 10% — should fail most of the time."""
        state = {"stage": "deploy", "error_type": "rollback_needed", "attempt_num": 0, "last_action": None}
        successes = sum(
            self.sim.apply_action(state, "retry")[0] for _ in range(200)
        )
        assert successes < 50, f"Expected <50/200 successes, got {successes}"

    def test_unknown_action_raises(self):
        state = {"stage": "lint", "error_type": "syntax_error", "attempt_num": 0, "last_action": None}
        with pytest.raises(AssertionError):
            self.sim.apply_action(state, "invalid_action")

    # ── encode / decode ─────────────────────────────────────────────────

    def test_encode_decode_roundtrip(self):
        state = {"stage": "test", "error_type": "flaky_test", "attempt_num": 3, "last_action": "retry"}
        idx = PipelineSimulator.encode_state(state)
        decoded = PipelineSimulator.decode_state(idx)
        assert decoded["stage"] == state["stage"]
        assert decoded["error_type"] == state["error_type"]
        assert decoded["attempt_num"] == state["attempt_num"]
        assert decoded["last_action"] == state["last_action"]

    def test_encode_no_last_action(self):
        state = {"stage": "lint", "error_type": "syntax_error", "attempt_num": 0, "last_action": None}
        idx = PipelineSimulator.encode_state(state)
        assert idx >= 0

    def test_encode_unique(self):
        """Different states should map to different indices."""
        s1 = {"stage": "lint", "error_type": "syntax_error", "attempt_num": 0, "last_action": None}
        s2 = {"stage": "build", "error_type": "missing_deps", "attempt_num": 0, "last_action": None}
        assert PipelineSimulator.encode_state(s1) != PipelineSimulator.encode_state(s2)

    # ── get_success_prob ────────────────────────────────────────────────

    def test_get_success_prob_known(self):
        p = self.sim.get_success_prob("lint", "syntax_error", "auto_fix")
        assert p == pytest.approx(0.92)

    def test_get_success_prob_default(self):
        p = self.sim.get_success_prob("lint", "syntax_error", "skip_stage")
        # skip_stage/lint/syntax_error is listed in config
        assert 0.0 <= p <= 1.0
