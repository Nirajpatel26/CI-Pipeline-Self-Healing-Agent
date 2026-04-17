# q_learning_agent.py — Tabular Q-Learning agent with epsilon-greedy exploration

import numpy as np
from config import (
    ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    N_STATES, N_ACTIONS, ACTIONS, RANDOM_SEED
)


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    State space: encoded integers 0..N_STATES-1 (via PipelineSimulator.encode_state)
    Action space: 6 discrete actions (indices 0..5)
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        n_actions: int = N_ACTIONS,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        epsilon_start: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        seed: int = RANDOM_SEED,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table initialised to zeros
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # ── Action Selection ────────────────────────────────────────────────────

    def select_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection. Returns action index."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state_idx]))

    def select_action_greedy(self, state_idx: int) -> int:
        """Pure greedy (for evaluation). Returns action index."""
        return int(np.argmax(self.q_table[state_idx]))

    # ── Q-Table Update ──────────────────────────────────────────────────────

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> None:
        """Bellman update: Q(s,a) ← Q(s,a) + α[r + γ max_a'Q(s',a') - Q(s,a)]"""
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])
        td_error = target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.alpha * td_error

    # ── Epsilon Decay ───────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Decay epsilon by EPSILON_DECAY, clamped to EPSILON_MIN."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Utilities ───────────────────────────────────────────────────────────

    def get_q_table(self) -> np.ndarray:
        """Return the full Q-table array (n_states × n_actions)."""
        return self.q_table.copy()

    def get_q_table_coverage(self) -> float:
        """Fraction of state-action pairs that have been visited (Q ≠ 0)."""
        visited = np.count_nonzero(self.q_table)
        return visited / self.q_table.size

    def best_action_name(self, state_idx: int) -> str:
        """Return the action name with the highest Q-value for a state."""
        return ACTIONS[self.select_action_greedy(state_idx)]

    def top_k_actions(self, state_idx: int, k: int = 2):
        """Return top-k (action_name, q_value) pairs for a state."""
        q_vals = self.q_table[state_idx]
        top_indices = np.argsort(q_vals)[::-1][:k]
        return [(ACTIONS[i], float(q_vals[i])) for i in top_indices]
