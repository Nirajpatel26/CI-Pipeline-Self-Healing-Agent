# tests/test_recovrate_smoke.py — Verifies the 0% RecovRate fix without any LLM
#
# Pre-fix: _select_action forced "escalate" whenever the inspector said risk=HIGH,
# starving the Q-table on those states, and eval used epsilon=0.0, so the greedy
# policy collapsed to action index 0 (retry) and RecovRate% approached 0.
#
# Post-fix:
#   (a) _select_action lets the RL agent pick from the full action set.
#   (b) Eval epsilon is small (>0) to break ties.
#   (c) Q-update always runs on the action actually taken.
#
# This smoke test trains a small Q-Learning agent and asserts RecovRate > 0.

from experiment_runner import run_condition, compute_summary
from q_learning_agent import QLearningAgent


def test_q_learning_recovrate_above_zero():
    """A 200-train / 100-eval Q-Learning run should recover on >0% of eval episodes."""
    seed = 7
    ql = QLearningAgent(seed=seed)

    # Train (mode=off → no LLM calls, pure Python RL loop)
    run_condition("q_learning", 200, train=True, rl_agent=ql, seed=seed, mode="off")

    # Evaluate greedy with a small epsilon for tie-breaking (the fix)
    ql_eval = QLearningAgent(seed=seed + 1)
    ql_eval.q_table = ql.q_table.copy()
    ql_eval.epsilon = 0.02

    logs = run_condition(
        "q_learning", 100, train=False, rl_agent=ql_eval, seed=seed + 1, mode="off"
    )
    summary = compute_summary(logs)

    assert summary["recovery_rate_pct"] > 0.0, (
        f"RecovRate% is still 0 — expected the fix to lift it above zero. "
        f"summary={summary}"
    )


def test_integrated_mode_runs_without_llm_server():
    """
    Integrated mode must not crash when the llama.cpp server is unreachable:
    each LLM wrapper falls back to a deterministic default via safe_call.
    """
    seed = 11
    ql = QLearningAgent(seed=seed)
    logs = run_condition(
        "q_learning_llm", 10, train=True, rl_agent=ql, seed=seed, mode="integrated"
    )
    assert len(logs) == 10
    # All fallbacks → at least some attempts with some reward signal
    rewards = [l["total_reward"] for l in logs]
    assert any(r != 0.0 for r in rewards)
