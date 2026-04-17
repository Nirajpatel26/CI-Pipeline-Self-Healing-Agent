# tests/test_q_learning.py — Tests for QLearningAgent and UCBAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from q_learning_agent import QLearningAgent
from ucb_agent import UCBAgent
from config import N_STATES, N_ACTIONS, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, ACTIONS


class TestQLearningAgent:

    def setup_method(self):
        self.agent = QLearningAgent(seed=42)

    # ── Initialisation ──────────────────────────────────────────────────

    def test_q_table_shape(self):
        assert self.agent.q_table.shape == (N_STATES, N_ACTIONS)

    def test_q_table_zeros_init(self):
        assert np.all(self.agent.q_table == 0.0)

    def test_epsilon_init(self):
        assert self.agent.epsilon == pytest.approx(EPSILON_START)

    # ── Action Selection ────────────────────────────────────────────────

    def test_select_action_valid(self):
        for _ in range(20):
            idx = self.agent.select_action(0)
            assert 0 <= idx < N_ACTIONS

    def test_select_action_greedy_picks_max(self):
        self.agent.q_table[5, 3] = 99.0  # force max at action 3 for state 5
        assert self.agent.select_action_greedy(5) == 3

    def test_select_action_fully_random_when_epsilon_1(self):
        """With epsilon=1.0 all actions should appear over 500 samples."""
        self.agent.epsilon = 1.0
        seen = set()
        for _ in range(500):
            seen.add(self.agent.select_action(0))
        assert len(seen) > 1  # not always the same action

    def test_select_action_greedy_when_epsilon_0(self):
        """With epsilon=0.0 always pick the greedy action."""
        self.agent.epsilon = 0.0
        self.agent.q_table[0, 2] = 5.0
        actions = [self.agent.select_action(0) for _ in range(20)]
        assert all(a == 2 for a in actions)

    # ── Bellman Update ──────────────────────────────────────────────────

    def test_bellman_update_non_terminal(self):
        """Q(s,a) should increase after a positive reward on a non-terminal step."""
        self.agent.update(state_idx=0, action_idx=0, reward=10.0,
                          next_state_idx=1, done=False)
        assert self.agent.q_table[0, 0] > 0.0

    def test_bellman_update_terminal(self):
        """On terminal step, next-state value should not be used."""
        q_before = self.agent.q_table[0, 0]
        self.agent.update(state_idx=0, action_idx=0, reward=-5.0,
                          next_state_idx=1, done=True)
        # Q(0,0) = Q(0,0) + alpha * (-5 - Q(0,0))
        expected = q_before + 0.1 * (-5.0 - q_before)
        assert self.agent.q_table[0, 0] == pytest.approx(expected)

    def test_bellman_convergence_simple(self):
        """Repeated updates with reward=10 and done=True should converge Q to 10."""
        for _ in range(1000):
            self.agent.update(0, 0, 10.0, 0, done=True)
        assert self.agent.q_table[0, 0] == pytest.approx(10.0, abs=0.1)

    # ── Epsilon Decay ───────────────────────────────────────────────────

    def test_decay_epsilon_decreases(self):
        before = self.agent.epsilon
        self.agent.decay_epsilon()
        assert self.agent.epsilon < before

    def test_decay_epsilon_clamp(self):
        self.agent.epsilon = EPSILON_MIN
        self.agent.decay_epsilon()
        assert self.agent.epsilon == pytest.approx(EPSILON_MIN)

    def test_decay_epsilon_2000_episodes(self):
        for _ in range(2000):
            self.agent.decay_epsilon()
        assert self.agent.epsilon == pytest.approx(EPSILON_MIN)

    # ── Utilities ───────────────────────────────────────────────────────

    def test_get_q_table_copy(self):
        qt = self.agent.get_q_table()
        qt[0, 0] = 999.0
        assert self.agent.q_table[0, 0] != 999.0  # should be a copy

    def test_top_k_actions(self):
        self.agent.q_table[10, 4] = 5.0
        self.agent.q_table[10, 1] = 3.0
        top = self.agent.top_k_actions(10, k=2)
        assert top[0][0] == ACTIONS[4]
        assert top[1][0] == ACTIONS[1]

    def test_q_table_coverage_zero_initially(self):
        assert self.agent.get_q_table_coverage() == pytest.approx(0.0)

    def test_q_table_coverage_increases_after_update(self):
        self.agent.update(0, 0, 1.0, 1, done=True)
        assert self.agent.get_q_table_coverage() > 0.0


class TestUCBAgent:

    def setup_method(self):
        self.agent = UCBAgent(seed=42)

    def test_q_table_shape(self):
        assert self.agent.q_table.shape == (N_STATES, N_ACTIONS)

    def test_visit_counts_init(self):
        """All counts initialised to 1 to avoid division by zero."""
        assert np.all(self.agent.visit_counts == 1.0)

    def test_select_action_valid(self):
        for _ in range(20):
            idx = self.agent.select_action(0)
            assert 0 <= idx < N_ACTIONS

    def test_update_increments_count(self):
        before = self.agent.visit_counts[0, 2]
        self.agent.update(0, 2, 5.0, 1, done=False)
        assert self.agent.visit_counts[0, 2] == before + 1

    def test_bellman_update(self):
        self.agent.update(0, 0, 10.0, 1, done=True)
        assert self.agent.q_table[0, 0] > 0.0

    def test_top_k_actions(self):
        self.agent.q_table[5, 0] = 8.0
        self.agent.q_table[5, 3] = 4.0
        top = self.agent.top_k_actions(5, k=2)
        assert top[0][0] == ACTIONS[0]
