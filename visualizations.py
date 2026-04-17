# visualizations.py — All plots for the CI Healing Agent experiment

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import RESULTS_DIR, ACTIONS, STAGES, N_EPISODES, EPSILON_START, EPSILON_DECAY, EPSILON_MIN

sns.set_theme(style="whitegrid", palette="muted")
FIGURE_DPI = 150


def _save(fig, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Learning Curve ─────────────────────────────────────────────────────

def plot_learning_curve(ql_train_logs, ucb_train_logs, window: int = 50):
    """
    Episode vs. rolling mean total reward for Q-Learning and UCB (training phase).
    """
    def rolling_mean(logs, w):
        rewards = [x["total_reward"] for x in logs]
        return pd.Series(rewards).rolling(w, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = range(1, len(ql_train_logs) + 1)

    ax.plot(episodes, rolling_mean(ql_train_logs, window),
            label="Q-Learning", color="steelblue", linewidth=1.5)
    ax.plot(range(1, len(ucb_train_logs) + 1), rolling_mean(ucb_train_logs, window),
            label="UCB", color="darkorange", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean Total Reward (rolling {window})")
    ax.set_title("Learning Curve — Q-Learning vs UCB")
    ax.legend()
    _save(fig, "learning_curve.png")


# ── 2. Recovery Rate Comparison ───────────────────────────────────────────

def plot_recovery_rate_comparison(summaries):
    """Bar chart comparing recovery rates across all 4 conditions."""
    labels = [s["condition"].replace("_", "\n") for s in summaries]
    values = [s["recovery_rate_pct"] for s in summaries]
    colors = ["#4878D0", "#6ACC65", "#D65F5F", "#B47CC7"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Recovery Rate (%)")
    ax.set_title("Pipeline Recovery Rate by Condition (Evaluation)")
    _save(fig, "recovery_rate_comparison.png")


# ── 3. Q-Table Heatmap ────────────────────────────────────────────────────

def plot_q_table_heatmap(q_table: np.ndarray, title: str = "Q-Learning", filename: str = "q_table_heatmap.png"):
    """
    Heatmap of mean Q-values aggregated over state clusters (one row per stage).
    Columns are actions.
    """
    from config import STAGES, N_ERRORS, N_ATTEMPTS, N_LAST_ACTIONS, N_ACTIONS

    n_per_stage = N_ERRORS * N_ATTEMPTS * N_LAST_ACTIONS
    stage_means = np.zeros((len(STAGES), N_ACTIONS))

    for i, stage in enumerate(STAGES):
        start = i * n_per_stage
        end = start + n_per_stage
        stage_means[i] = q_table[start:end].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        stage_means,
        ax=ax,
        xticklabels=ACTIONS,
        yticklabels=STAGES,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
    )
    ax.set_title(f"Q-Table Heatmap — {title} (mean Q-value per stage × action)")
    ax.set_xlabel("Action")
    ax.set_ylabel("Pipeline Stage")
    _save(fig, filename)


# ── 4. Escalation Rate Over Training ─────────────────────────────────────

def plot_escalation_rate(ql_train_logs, ucb_train_logs, window: int = 100):
    """Rolling escalation rate (%) over training episodes."""
    def rolling_esc(logs, w):
        esc = [x["escalated"] for x in logs]
        return pd.Series(esc).rolling(w, min_periods=1).mean().values * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(ql_train_logs) + 1), rolling_esc(ql_train_logs, window),
            label="Q-Learning", color="steelblue")
    ax.plot(range(1, len(ucb_train_logs) + 1), rolling_esc(ucb_train_logs, window),
            label="UCB", color="darkorange", linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Escalation Rate % (rolling {window})")
    ax.set_title("Escalation Rate During Training")
    ax.legend()
    _save(fig, "escalation_rate.png")


# ── 5. Epsilon Decay Curve ────────────────────────────────────────────────

def plot_epsilon_decay(n_episodes: int = N_EPISODES):
    """Theoretical epsilon decay curve."""
    eps = [EPSILON_START]
    for _ in range(n_episodes - 1):
        eps.append(max(EPSILON_MIN, eps[-1] * EPSILON_DECAY))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, n_episodes + 1), eps, color="mediumpurple", linewidth=1.5)
    ax.axhline(EPSILON_MIN, color="gray", linestyle=":", label=f"ε_min={EPSILON_MIN}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (exploration rate)")
    ax.set_title("Epsilon Decay Curve")
    ax.legend()
    _save(fig, "epsilon_decay.png")


# ── 6. Recovery Attempts Box Plot ─────────────────────────────────────────

def plot_recovery_attempts_boxplot(episode_log_path: str = None):
    """Box plot of recovery attempts per condition (eval phases only)."""
    if episode_log_path is None:
        episode_log_path = os.path.join(RESULTS_DIR, "episode_log.csv")

    df = pd.read_csv(episode_log_path)
    eval_df = df[df.get("phase", "eval") != "train"] if "phase" in df.columns else df

    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = eval_df["condition"].unique()
    data = [eval_df[eval_df["condition"] == c]["steps"].values for c in conditions]
    bplot = ax.boxplot(data, labels=[c.replace("_", "\n") for c in conditions],
                       patch_artist=True, notch=False)
    colors = ["#4878D0", "#6ACC65", "#D65F5F", "#B47CC7"]
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Number of Recovery Attempts")
    ax.set_title("Recovery Attempts Distribution by Condition")
    _save(fig, "recovery_attempts_boxplot.png")


# ── Master Plotting Function ──────────────────────────────────────────────

def generate_all_plots(
    ql_train_logs,
    ucb_train_logs,
    summaries,
    ql_q_table: np.ndarray = None,
    ucb_q_table: np.ndarray = None,
):
    """Generate all 6 required plots."""
    print("\n── Generating visualizations ──")

    plot_learning_curve(ql_train_logs, ucb_train_logs)
    plot_recovery_rate_comparison(summaries)

    if ql_q_table is not None:
        plot_q_table_heatmap(ql_q_table, title="Q-Learning", filename="q_table_heatmap.png")
    if ucb_q_table is not None:
        plot_q_table_heatmap(ucb_q_table, title="UCB", filename="q_table_heatmap_ucb.png")

    plot_escalation_rate(ql_train_logs, ucb_train_logs)
    plot_epsilon_decay()

    try:
        plot_recovery_attempts_boxplot()
    except Exception as e:
        print(f"  [warning] boxplot skipped: {e}")

    print("── All plots saved to results/ ──")


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    # Load from saved files (run experiment_runner.py first)
    import sys

    ql_qt = np.load(os.path.join(RESULTS_DIR, "q_table_ql.npy")) \
        if os.path.exists(os.path.join(RESULTS_DIR, "q_table_ql.npy")) else None
    ucb_qt = np.load(os.path.join(RESULTS_DIR, "q_table_ucb.npy")) \
        if os.path.exists(os.path.join(RESULTS_DIR, "q_table_ucb.npy")) else None

    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        summaries = json.load(f)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "episode_log.csv"))
    ql_train = df[(df["condition"] == "q_learning") & (df.get("phase", "train") == "train")].to_dict("records")
    ucb_train = df[(df["condition"] == "ucb") & (df.get("phase", "train") == "train")].to_dict("records")

    generate_all_plots(ql_train, ucb_train, summaries, ql_qt, ucb_qt)
