# ucb_agent.py — UCB1 exploration agent (alternative to epsilon-greedy)

import numpy as np
from config import (
    ALPHA, GAMMA, UCB_C,
    N_STATES, N_ACTIONS, ACTIONS, RANDOM_SEED
)


class UCBAgent:
    """
    Q-Learning agent using UCB1 (Upper Confidence Bound) exploration
    instead of epsilon-greedy.

    UCB action selection: a* = argmax_a [ Q(s,a) + c * sqrt(ln(N(s)) / N(s,a)) ]

    Visit counts are initialised to 1 to avoid division by zero.
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        n_actions: int = N_ACTIONS,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        c: float = UCB_C,
        seed: int = RANDOM_SEED,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        # Initialise counts to 1 so UCB is defined from episode 1
        self.visit_counts = np.ones((n_states, n_actions), dtype=np.float64)

    # ── Action Selection ────────────────────────────────────────────────────

    def select_action(self, state_idx: int) -> int:
        """UCB1 action selection. Returns action index."""
        n_state = self.visit_counts[state_idx].sum()
        ucb_values = (
            self.q_table[state_idx]
            + self.c * np.sqrt(np.log(n_state) / self.visit_counts[state_idx])
        )
        return int(np.argmax(ucb_values))

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
        """Bellman update identical to Q-Learning; also increments visit count."""
        self.visit_counts[state_idx, action_idx] += 1

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])
        td_error = target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.alpha * td_error

    # ── Utilities ───────────────────────────────────────────────────────────

    def get_q_table(self) -> np.ndarray:
        return self.q_table.copy()

    def get_q_table_coverage(self) -> float:
        """Fraction of state-action pairs with Q ≠ 0."""
        visited = np.count_nonzero(self.q_table)
        return visited / self.q_table.size

    def best_action_name(self, state_idx: int) -> str:
        return ACTIONS[self.select_action_greedy(state_idx)]

    def top_k_actions(self, state_idx: int, k: int = 2):
        q_vals = self.q_table[state_idx]
        top_indices = np.argsort(q_vals)[::-1][:k]
        return [(ACTIONS[i], float(q_vals[i])) for i in top_indices]
