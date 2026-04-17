# experiment_runner.py — Runs all 4 experimental conditions and collects metrics

import os
import csv
import json
import time
import numpy as np
from typing import Dict, List

from config import (
    N_EPISODES, EVAL_EPISODES, RANDOM_SEED, RESULTS_DIR
)
from pipeline_simulator import PipelineSimulator
from q_learning_agent import QLearningAgent
from ucb_agent import UCBAgent
from autogen_agents import CIHealingSystem


# ── Metric Collection ─────────────────────────────────────────────────────

def run_condition(
    condition: str,
    n_episodes: int,
    train: bool = False,
    rl_agent=None,
    seed: int = RANDOM_SEED,
    use_autogen: bool = False,
    mode: str = None,
) -> List[Dict]:
    """
    Run `n_episodes` episodes for a given condition and return per-episode logs.

    Args:
        condition  : "baseline" | "rule_based" | "q_learning" | "ucb" | "q_learning_llm"
        n_episodes : number of episodes
        train      : whether to update the RL agent's Q-table
        rl_agent   : QLearningAgent or UCBAgent (None for baseline/rule_based)
        seed       : random seed for the simulator
        use_autogen: legacy flag; kept for back-compat. Prefer `mode`.
        mode       : "off" | "narration" | "integrated". Overrides use_autogen.
    """
    simulator = PipelineSimulator(seed=seed)
    system = CIHealingSystem(
        simulator=simulator,
        rl_agent=rl_agent,
        condition=condition,
        use_autogen=use_autogen,
        max_steps=10,
        mode=mode,
    )

    logs = []
    for ep in range(n_episodes):
        result = system.run_episode(train=train)
        logs.append({
            "condition": condition,
            "episode": ep,
            "total_reward": result["total_reward"],
            "steps": result["steps"],
            "success": int(result["success"]),
            "escalated": int(result["escalated"]),
            "train": int(train),
            "epsilon": getattr(rl_agent, "epsilon", None),
        })

        if (ep + 1) % 200 == 0:
            recent = logs[-200:]
            avg_r = np.mean([x["total_reward"] for x in recent])
            rec_rate = np.mean([x["success"] for x in recent]) * 100
            print(
                f"  [{condition}] ep {ep+1:>4}/{n_episodes} | "
                f"avg_reward={avg_r:+.2f} | recovery={rec_rate:.1f}%"
            )

    return logs


def compute_summary(logs: List[Dict]) -> Dict:
    """Aggregate per-episode logs into a summary dict."""
    rewards = [x["total_reward"] for x in logs]
    steps = [x["steps"] for x in logs]
    successes = [x["success"] for x in logs]
    escalations = [x["escalated"] for x in logs]

    # Convergence episode: first window of 100 episodes with recovery_rate >= 85%
    convergence_ep = None
    for i in range(100, len(logs) + 1):
        window = successes[i - 100: i]
        if np.mean(window) >= 0.85:
            convergence_ep = logs[i - 100]["episode"]
            break

    return {
        "condition": logs[0]["condition"],
        "recovery_rate_pct": round(np.mean(successes) * 100, 2),
        "mean_attempts": round(np.mean(steps), 3),
        "escalation_rate_pct": round(np.mean(escalations) * 100, 2),
        "mean_episode_reward": round(np.mean(rewards), 4),
        "std_episode_reward": round(np.std(rewards), 4),
        "convergence_episode": convergence_ep,
        "n_episodes": len(logs),
    }


# ── ExperimentRunner ─────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Runs all 4 conditions sequentially and writes results to CSV files.

    Output files:
        results/episode_log.csv       — per-episode data for all conditions
        results/summary.json          — aggregate metrics per condition
        results/q_table_ql.npy        — Q-table from Q-Learning agent
        results/q_table_ucb.npy       — Q-table from UCB agent
    """

    def __init__(
        self,
        n_train: int = N_EPISODES,
        n_eval: int = EVAL_EPISODES,
        seed: int = RANDOM_SEED,
        use_autogen: bool = False,
        mode: str = "off",
        include_llm_condition: bool = False,
        epsilon_eval: float = 0.02,
    ):
        self.n_train = n_train
        self.n_eval = n_eval
        self.seed = seed
        self.use_autogen = use_autogen
        self.mode = mode
        self.include_llm_condition = include_llm_condition
        self.epsilon_eval = epsilon_eval
        self.all_logs: List[Dict] = []
        self.summaries: List[Dict] = []

    def run_all(self) -> Dict:
        """Run all 4 conditions. Returns summary dict."""
        t0 = time.time()

        # Baseline and rule-based never need LLM narration/integration.
        baseline_mode = "off"

        # ── Baseline ──────────────────────────────────────────────────────
        print("\n=== Condition 1: Baseline (always retry → escalate) ===")
        baseline_logs = run_condition(
            "baseline", self.n_eval, train=False,
            rl_agent=None, seed=self.seed, use_autogen=False,
            mode=baseline_mode,
        )
        self.all_logs.extend(baseline_logs)
        self.summaries.append(compute_summary(baseline_logs))

        # ── Rule-Based ────────────────────────────────────────────────────
        print("\n=== Condition 2: Rule-Based (hardcoded if-else) ===")
        rule_logs = run_condition(
            "rule_based", self.n_eval, train=False,
            rl_agent=None, seed=self.seed, use_autogen=False,
            mode=baseline_mode,
        )
        self.all_logs.extend(rule_logs)
        self.summaries.append(compute_summary(rule_logs))

        # ── Q-Learning ────────────────────────────────────────────────────
        print(f"\n=== Condition 3: Q-Learning (train={self.n_train}, eval={self.n_eval}) ===")
        ql_agent = QLearningAgent(seed=self.seed)

        print("  [Training]")
        ql_train_logs = run_condition(
            "q_learning", self.n_train, train=True,
            rl_agent=ql_agent, seed=self.seed, use_autogen=self.use_autogen,
            mode=self.mode,
        )

        print("  [Evaluation]")
        ql_agent_eval = QLearningAgent(seed=self.seed + 1)
        ql_agent_eval.q_table = ql_agent.q_table.copy()
        # epsilon_eval > 0 breaks ties on untrained Q-rows so the policy does
        # not collapse to action index 0 (retry). Fix for 0% RecovRate.
        ql_agent_eval.epsilon = self.epsilon_eval
        ql_eval_logs = run_condition(
            "q_learning", self.n_eval, train=False,
            rl_agent=ql_agent_eval, seed=self.seed + 1, use_autogen=self.use_autogen,
            mode=self.mode,
        )

        # tag train logs separately for learning curves
        for log in ql_train_logs:
            log["phase"] = "train"
        for log in ql_eval_logs:
            log["phase"] = "eval"

        self.all_logs.extend(ql_train_logs)
        self.all_logs.extend(ql_eval_logs)
        self.summaries.append(compute_summary(ql_eval_logs))
        np.save(os.path.join(RESULTS_DIR, "q_table_ql.npy"), ql_agent.get_q_table())

        # ── UCB ───────────────────────────────────────────────────────────
        print(f"\n=== Condition 4: UCB (train={self.n_train}, eval={self.n_eval}) ===")
        ucb_agent = UCBAgent(seed=self.seed)

        print("  [Training]")
        ucb_train_logs = run_condition(
            "ucb", self.n_train, train=True,
            rl_agent=ucb_agent, seed=self.seed, use_autogen=self.use_autogen,
            mode=self.mode,
        )

        print("  [Evaluation]")
        ucb_agent_eval = UCBAgent(seed=self.seed + 1)
        ucb_agent_eval.q_table = ucb_agent.q_table.copy()
        ucb_eval_logs = run_condition(
            "ucb", self.n_eval, train=False,
            rl_agent=ucb_agent_eval, seed=self.seed + 1, use_autogen=self.use_autogen,
            mode=self.mode,
        )

        for log in ucb_train_logs:
            log["phase"] = "train"
        for log in ucb_eval_logs:
            log["phase"] = "eval"

        self.all_logs.extend(ucb_train_logs)
        self.all_logs.extend(ucb_eval_logs)
        self.summaries.append(compute_summary(ucb_eval_logs))
        np.save(os.path.join(RESULTS_DIR, "q_table_ucb.npy"), ucb_agent.get_q_table())

        # ── Q-Learning + LLM (True Integration) ──────────────────────────
        if self.include_llm_condition:
            print(
                f"\n=== Condition 5: Q-Learning + LLM (mode=integrated, "
                f"train={self.n_train}, eval={self.n_eval}) ==="
            )
            ql_llm_agent = QLearningAgent(seed=self.seed)

            print("  [Training]")
            ql_llm_train_logs = run_condition(
                "q_learning_llm", self.n_train, train=True,
                rl_agent=ql_llm_agent, seed=self.seed, use_autogen=False,
                mode="integrated",
            )

            print("  [Evaluation]")
            ql_llm_eval = QLearningAgent(seed=self.seed + 1)
            ql_llm_eval.q_table = ql_llm_agent.q_table.copy()
            ql_llm_eval.epsilon = self.epsilon_eval
            ql_llm_eval_logs = run_condition(
                "q_learning_llm", self.n_eval, train=False,
                rl_agent=ql_llm_eval, seed=self.seed + 1, use_autogen=False,
                mode="integrated",
            )

            for log in ql_llm_train_logs:
                log["phase"] = "train"
            for log in ql_llm_eval_logs:
                log["phase"] = "eval"

            self.all_logs.extend(ql_llm_train_logs)
            self.all_logs.extend(ql_llm_eval_logs)
            self.summaries.append(compute_summary(ql_llm_eval_logs))
            np.save(
                os.path.join(RESULTS_DIR, "q_table_ql_llm.npy"),
                ql_llm_agent.get_q_table(),
            )
            self.ql_llm_train_logs = ql_llm_train_logs

        # ── Save outputs ──────────────────────────────────────────────────
        self._save_episode_log()
        self._save_summary()

        elapsed = time.time() - t0
        print(f"\n✓ All conditions complete in {elapsed:.1f}s")
        self._print_summary_table()

        # Store training logs for visualizations
        self.ql_train_logs = ql_train_logs
        self.ucb_train_logs = ucb_train_logs

        return {
            "summaries": self.summaries,
            "ql_agent": ql_agent,
            "ucb_agent": ucb_agent,
        }

    def _save_episode_log(self):
        path = os.path.join(RESULTS_DIR, "episode_log.csv")
        if not self.all_logs:
            return
        fieldnames = list(self.all_logs[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.all_logs)
        print(f"  Saved: {path}")

    def _save_summary(self):
        path = os.path.join(RESULTS_DIR, "summary.json")
        with open(path, "w") as f:
            json.dump(self.summaries, f, indent=2)
        print(f"  Saved: {path}")

    def _print_summary_table(self):
        print("\n" + "=" * 75)
        print(f"{'Condition':<15} {'RecovRate%':>10} {'MeanAttempts':>12} "
              f"{'EscRate%':>9} {'MeanReward':>11} {'ConvergEp':>10}")
        print("-" * 75)
        for s in self.summaries:
            conv = str(s["convergence_episode"]) if s["convergence_episode"] else "N/A"
            print(
                f"{s['condition']:<15} {s['recovery_rate_pct']:>10.1f} "
                f"{s['mean_attempts']:>12.2f} {s['escalation_rate_pct']:>9.1f} "
                f"{s['mean_episode_reward']:>11.4f} {conv:>10}"
            )
        print("=" * 75)


# ── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CI Healing Agent Experiment Runner")
    parser.add_argument("--train", type=int, default=N_EPISODES,
                        help="Number of training episodes (default 2000)")
    parser.add_argument("--eval", type=int, default=EVAL_EPISODES,
                        help="Number of eval episodes (default 1000)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--autogen", action="store_true",
                        help="Legacy: enable AutoGen narration (same as --mode narration)")
    parser.add_argument(
        "--mode",
        choices=["off", "narration", "integrated"],
        default=None,
        help=(
            "LLM mode: 'off' (pure RL), 'narration' (old post-hoc GroupChat), "
            "'integrated' (True Integration: LLM in the decision loop)."
        ),
    )
    parser.add_argument(
        "--with-llm-condition",
        action="store_true",
        help=(
            "Run an additional 5th condition (q_learning_llm, mode=integrated) "
            "alongside the 4 standard conditions for side-by-side comparison."
        ),
    )
    parser.add_argument(
        "--epsilon-eval", type=float, default=0.02,
        help=(
            "Eval-time epsilon for Q-Learning greedy policy. Small non-zero "
            "values break ties on untrained Q-rows (fix for 0%% RecovRate)."
        ),
    )
    args = parser.parse_args()

    # Resolve mode: explicit --mode wins; else fall back to --autogen.
    mode = args.mode if args.mode is not None else ("narration" if args.autogen else "off")

    runner = ExperimentRunner(
        n_train=args.train,
        n_eval=args.eval,
        seed=args.seed,
        use_autogen=args.autogen,
        mode=mode,
        include_llm_condition=args.with_llm_condition,
        epsilon_eval=args.epsilon_eval,
    )
    runner.run_all()
