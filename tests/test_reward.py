# tests/test_reward.py — Tests for compute_reward()

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from reward_function import compute_reward
from config import STAGE_CRITICALITY


def make_state(stage, error_type="syntax_error", attempt_num=0, last_action=None):
    return {
        "stage": stage,
        "error_type": error_type,
        "attempt_num": attempt_num,
        "last_action": last_action,
    }


class TestComputeReward:

    # ── Success rewards ─────────────────────────────────────────────────

    def test_success_attempt_0_gives_10(self):
        state = make_state("lint", attempt_num=0)
        r = compute_reward(state, "auto_fix", success=True)
        assert r == pytest.approx(10.0)

    def test_success_attempt_1_gives_7(self):
        state = make_state("lint", attempt_num=1)
        r = compute_reward(state, "retry", success=True)
        assert r == pytest.approx(7.0)

    def test_success_attempt_2_gives_5(self):
        state = make_state("build", attempt_num=2)
        r = compute_reward(state, "auto_fix", success=True)
        assert r == pytest.approx(5.0)

    def test_success_attempt_3_gives_3(self):
        state = make_state("test", attempt_num=3)
        r = compute_reward(state, "retry", success=True)
        assert r == pytest.approx(3.0)

    def test_success_attempt_5_still_gives_3(self):
        state = make_state("deploy", attempt_num=5)
        r = compute_reward(state, "revert", success=True)
        assert r == pytest.approx(3.0)

    # ── Failure rewards ─────────────────────────────────────────────────

    def test_failure_basic_penalty(self):
        state = make_state("lint", attempt_num=0)
        r = compute_reward(state, "retry", success=False)
        # -2 * criticality(lint=0.5) = -1.0
        assert r == pytest.approx(-1.0)

    def test_failure_high_criticality_stage(self):
        state = make_state("security_scan", error_type="vuln_detected", attempt_num=0)
        r = compute_reward(state, "retry", success=False)
        # -2 * 1.5 = -3.0
        assert r == pytest.approx(-3.0)

    def test_failure_deploy_criticality(self):
        state = make_state("deploy", error_type="rollback_needed", attempt_num=0)
        r = compute_reward(state, "retry", success=False)
        # -2 * 1.3 = -2.6
        assert r == pytest.approx(-2.6)

    # ── Escalate penalty ────────────────────────────────────────────────

    def test_escalate_penalty_on_failure(self):
        state = make_state("lint", attempt_num=0)
        r = compute_reward(state, "escalate", success=False)
        # base: -2 * 0.5 = -1; escalate: -8 * 0.5 = -4; total = -5 → clamped to -5
        expected = max(-10.0, -1.0 - 4.0)
        assert r == pytest.approx(expected)

    def test_escalate_high_criticality_clamped(self):
        """security_scan + escalate on failure should be clamped to -10."""
        state = make_state("security_scan", error_type="vuln_detected", attempt_num=2)
        r = compute_reward(state, "escalate", success=False)
        assert r == pytest.approx(-10.0)

    # ── skip_stage on security_scan ─────────────────────────────────────

    def test_skip_security_scan_penalty(self):
        """skip_stage on security_scan adds extra -5 penalty."""
        state = make_state("security_scan", error_type="vuln_detected", attempt_num=0)
        r_skip = compute_reward(state, "skip_stage", success=True)
        r_auto = compute_reward(state, "auto_fix", success=True)
        # skip_stage success: +10 - 5 = +5; auto_fix success: +10
        assert r_skip < r_auto
        assert r_skip == pytest.approx(5.0)

    def test_skip_non_security_no_extra_penalty(self):
        """skip_stage on a non-security stage should NOT add the -5 penalty."""
        state = make_state("test", error_type="flaky_test", attempt_num=0)
        r = compute_reward(state, "skip_stage", success=True)
        assert r == pytest.approx(10.0)

    # ── Clamping ────────────────────────────────────────────────────────

    def test_reward_clamped_to_minus_10(self):
        state = make_state("security_scan", error_type="secret_exposed", attempt_num=5)
        r = compute_reward(state, "escalate", success=False)
        assert r >= -10.0

    def test_reward_clamped_to_plus_10(self):
        state = make_state("lint", attempt_num=0)
        r = compute_reward(state, "auto_fix", success=True)
        assert r <= 10.0

    # ── Stage criticality multiplier ────────────────────────────────────

    def test_stage_criticality_affects_penalty(self):
        """Higher criticality stages should produce larger penalties."""
        s_low = make_state("lint", attempt_num=0)       # criticality 0.5
        s_high = make_state("security_scan", error_type="vuln_detected", attempt_num=0)  # criticality 1.5
        r_low = compute_reward(s_low, "retry", success=False)
        r_high = compute_reward(s_high, "retry", success=False)
        assert r_high < r_low
