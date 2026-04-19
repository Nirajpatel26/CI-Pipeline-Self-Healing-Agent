"""
generate_sample_interactions.py
================================
Generates two artefacts:

1. results/sample_interactions.txt
   — Human-readable step-by-step episode traces for EARLY vs LATE training,
     and a side-by-side BEFORE/AFTER comparison across all 4 conditions.
     Used in the report and video demonstration.

2. results/sample_interactions.png
   — A 2-panel figure showing 3 annotated episode traces each for
     (a) Early Q-Learning (episodes 1-10) and
     (b) Late Q-Learning (episodes 1990-2000).
     Clearly visualises the learning progress.

Run:  python generate_sample_interactions.py
"""

import os, sys, random, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import (ACTIONS, ACTION_IDX, N_STATES, N_ACTIONS,
                    RANDOM_SEED, MAX_STEPS, RESULTS_DIR)
from pipeline_simulator import PipelineSimulator
from q_learning_agent import QLearningAgent
from ucb_agent import UCBAgent
from reward_function import compute_reward

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def _action_emoji(action: str) -> str:
    return {
        "retry":          "🔁",
        "revert":         "↩️ ",
        "auto_fix":       "🔧",
        "switch_version": "🔄",
        "skip_stage":     "⏭️ ",
        "escalate":       "🚨",
    }.get(action, "  ")


def run_traced_episode(simulator: PipelineSimulator,
                       agent,
                       epsilon_override: float | None = None,
                       seed_offset: int = 0) -> dict:
    """
    Run one episode and return a full step-by-step trace.
    epsilon_override: if set, forces this epsilon for action selection.
    """
    rng_backup = None
    if epsilon_override is not None and hasattr(agent, "epsilon"):
        rng_backup = agent.epsilon
        agent.epsilon = epsilon_override

    state = simulator.generate_failure()
    steps = []
    total_reward = 0.0
    escalated = False

    for attempt in range(MAX_STEPS):
        state_idx = PipelineSimulator.encode_state(state)
        action = agent.select_action(state_idx)
        action_name = ACTIONS[action]
        if action_name == "escalate":
            escalated = True

        success, next_state = simulator.apply_action(state, action_name)
        reward = compute_reward(state, action_name, success)
        total_reward += reward

        steps.append({
            "attempt":    attempt + 1,
            "stage":      state["stage"],
            "error":      state["error_type"],
            "action":     action_name,
            "success":    success,
            "reward":     reward,
        })

        done = success or action_name == "escalate" or attempt == MAX_STEPS - 1
        if done:
            break
        state = next_state

    if rng_backup is not None:
        agent.epsilon = rng_backup

    return {
        "steps":        steps,
        "total_reward": round(total_reward, 2),
        "escalated":    escalated,
        "resolved":     steps[-1]["success"] or escalated,
        "n_steps":      len(steps),
    }


def format_episode(episode_num: int, trace: dict, label: str = "") -> str:
    lines = []
    tag = f"  [{label}]" if label else ""
    lines.append(f"\n{'─'*60}")
    lines.append(f"  Episode {episode_num}{tag}")
    lines.append(f"  Failure: {trace['steps'][0]['stage']} / {trace['steps'][0]['error']}")
    lines.append(f"{'─'*60}")
    for s in trace["steps"]:
        outcome = "✅ FIXED" if s["success"] else ("🚨 ESCALATED" if s["action"] == "escalate" else "❌ failed")
        lines.append(
            f"  Step {s['attempt']}: {_action_emoji(s['action'])} {s['action']:<16}"
            f"  reward={s['reward']:+6.1f}   {outcome}"
        )
    lines.append(f"{'─'*60}")
    lines.append(f"  Total reward: {trace['total_reward']:+.2f}   "
                 f"Steps: {trace['n_steps']}   "
                 f"Escalated: {'YES' if trace['escalated'] else 'no'}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT FILE
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_report():
    lines = []
    lines.append("=" * 65)
    lines.append("  CI Pipeline Self-Healing Agent — Sample Interactions")
    lines.append("  INFO 7375 | Niraj Patel")
    lines.append("=" * 65)

    sim = PipelineSimulator(seed=RANDOM_SEED)

    # ── SECTION A: Early vs Late Q-Learning ──────────────────────────────────
    lines.append("\n\n" + "█" * 65)
    lines.append("  SECTION A  Learning Progress — Q-Learning Agent")
    lines.append("  Compare how the agent behaves at episode ~10 vs episode ~1990")
    lines.append("█" * 65)

    # Train a fresh agent all the way through, capturing early & late episodes
    agent_ql = QLearningAgent(n_states=N_STATES, n_actions=N_ACTIONS, seed=RANDOM_SEED)
    sim2 = PipelineSimulator(seed=RANDOM_SEED)

    early_traces, late_traces = [], []
    SCENARIO_ERRORS = [
        # cover diverse stages
        ("lint",          "syntax_error"),
        ("security_scan", "vuln_detected"),
        ("deploy",        "rollback_needed"),
    ]

    # Run 2000 training episodes; capture 3 early (ep 5,8,10) and 3 late (ep 1988,1994,1999)
    EARLY_EPS  = {5, 8, 10}
    LATE_EPS   = {1988, 1994, 1999}
    TRACE_SEED_OVERRIDE = {   # pin scenario so early vs late face the same failure
        5:    ("lint",          "syntax_error"),
        8:    ("security_scan", "vuln_detected"),
        10:   ("deploy",        "rollback_needed"),
        1988: ("lint",          "syntax_error"),
        1994: ("security_scan", "vuln_detected"),
        1999: ("deploy",        "rollback_needed"),
    }

    rng_trace = random.Random(RANDOM_SEED + 99)

    for ep in range(2000):
        if ep in TRACE_SEED_OVERRIDE:
            # Force a specific failure for a fair early/late comparison
            stage, error = TRACE_SEED_OVERRIDE[ep]
            forced_state = {"stage": stage, "error_type": error,
                            "attempt_num": 0, "last_action": None}
            # run traced episode without training (read-only Q-table peek)
            trace = run_traced_episode(sim2, agent_ql)
            # override state so trace starts from the fixed failure
            forced_trace = run_traced_episode(sim2, agent_ql)
            # manually replay with forced state
            state = forced_state.copy()
            steps = []
            total_reward = 0.0
            escalated = False
            for attempt in range(MAX_STEPS):
                state["attempt_num"] = attempt
                state_idx = PipelineSimulator.encode_state(state)
                action_idx = agent_ql.select_action(state_idx)
                action_name = ACTIONS[action_idx]
                if action_name == "escalate":
                    escalated = True
                success, next_state = sim2.apply_action(state, action_name)
                reward = compute_reward(state, action_name, success)
                total_reward += reward
                steps.append({"attempt": attempt + 1, "stage": state["stage"],
                               "error": state["error_type"], "action": action_name,
                               "success": success, "reward": reward})
                done = success or action_name == "escalate" or attempt == MAX_STEPS - 1
                if done:
                    break
                state = next_state
            t = {"steps": steps, "total_reward": round(total_reward, 2),
                 "escalated": escalated, "resolved": steps[-1]["success"] or escalated,
                 "n_steps": len(steps)}
            if ep in EARLY_EPS:
                early_traces.append((ep + 1, t, TRACE_SEED_OVERRIDE[ep]))
            else:
                late_traces.append((ep + 1, t, TRACE_SEED_OVERRIDE[ep]))

        # Normal training step
        state = sim2.generate_failure()
        for _ in range(MAX_STEPS):
            state_idx = PipelineSimulator.encode_state(state)
            a = agent_ql.select_action(state_idx)
            action_name = ACTIONS[a]
            success, next_state = sim2.apply_action(state, action_name)
            reward = compute_reward(state, action_name, success)
            next_idx = PipelineSimulator.encode_state(next_state)
            done = success or action_name == "escalate"
            agent_ql.update(state_idx, a, reward, next_idx, done)
            if done:
                break
            state = next_state
        agent_ql.decay_epsilon()

    lines.append("\n  ── EARLY TRAINING (episodes 5, 8, 10) — agent is still exploring ──")
    lines.append("  At this stage ε ≈ 0.97, so most actions are random.\n")
    for ep_num, trace, (stage, error) in early_traces:
        lines.append(format_episode(ep_num, trace, label="EARLY"))

    lines.append("\n\n  ── LATE TRAINING (episodes 1988, 1994, 1999) — agent has converged ──")
    lines.append("  At this stage ε ≈ 0.05, agent acts on learned Q-values.\n")
    for ep_num, trace, (stage, error) in late_traces:
        lines.append(format_episode(ep_num, trace, label="LATE"))

    # ── SECTION B: Before / After — all 4 conditions on the same 5 failures ──
    lines.append("\n\n" + "█" * 65)
    lines.append("  SECTION B  Before / After — Same Failures, 4 Policies")
    lines.append("  Each row is the SAME failure; columns are the 4 policies.")
    lines.append("█" * 65)

    SHOWCASE_FAILURES = [
        ("lint",          "syntax_error"),
        ("build",         "version_conflict"),
        ("test",          "flaky_test"),
        ("security_scan", "vuln_detected"),
        ("deploy",        "rollback_needed"),
    ]

    # Prepare trained UCB agent
    agent_ucb = UCBAgent(n_states=N_STATES, n_actions=N_ACTIONS, seed=RANDOM_SEED)
    sim3 = PipelineSimulator(seed=RANDOM_SEED)
    for _ in range(2000):
        state = sim3.generate_failure()
        for _ in range(MAX_STEPS):
            state_idx = PipelineSimulator.encode_state(state)
            a = agent_ucb.select_action(state_idx)
            action_name = ACTIONS[a]
            success, next_state = sim3.apply_action(state, action_name)
            reward = compute_reward(state, action_name, success)
            next_idx = PipelineSimulator.encode_state(next_state)
            done = success or action_name == "escalate"
            agent_ucb.update(state_idx, a, reward, next_idx, done)
            if done:
                break
            state = next_state

    def baseline_action(state, attempt):
        return "escalate" if attempt >= 3 else "retry"

    RULE_MAP = {
        "syntax_error": "auto_fix",    "style_violation": "auto_fix",
        "import_error": "switch_version",
        "missing_deps": "switch_version", "compile_error": "revert",
        "version_conflict": "switch_version",
        "flaky_test": "retry",          "assertion_error": "auto_fix",
        "timeout": "retry",
        "vuln_detected": "revert",      "license_violation": "auto_fix",
        "secret_exposed": "revert",
        "rollback_needed": "revert",    "resource_unavailable": "retry",
        "config_error": "auto_fix",
    }

    # Freeze QL and UCB to greedy
    ql_eps_save = agent_ql.epsilon
    agent_ql.epsilon = 0.0

    header = f"\n  {'Failure':<32} {'Baseline':^14} {'Rule-Based':^14} {'Q-Learning':^14} {'UCB':^14}"
    lines.append(header)
    lines.append("  " + "─" * 90)

    sim_cmp = PipelineSimulator(seed=RANDOM_SEED + 7)

    for stage, error in SHOWCASE_FAILURES:
        row_results = {}
        for policy_name in ["baseline", "rule_based", "q_learning", "ucb"]:
            state = {"stage": stage, "error_type": error,
                     "attempt_num": 0, "last_action": None}
            steps_taken = []
            total_r = 0.0
            esc = False
            for attempt in range(MAX_STEPS):
                state["attempt_num"] = attempt
                if policy_name == "baseline":
                    action_name = baseline_action(state, attempt)
                elif policy_name == "rule_based":
                    action_name = RULE_MAP.get(error, "escalate")
                elif policy_name == "q_learning":
                    state_idx = PipelineSimulator.encode_state(state)
                    a = agent_ql.select_action(state_idx)
                    action_name = ACTIONS[a]
                else:  # ucb
                    state_idx = PipelineSimulator.encode_state(state)
                    a = agent_ucb.select_action(state_idx)
                    action_name = ACTIONS[a]

                if action_name == "escalate":
                    esc = True
                success, next_state = sim_cmp.apply_action(state, action_name)
                reward = compute_reward(state, action_name, success)
                total_r += reward
                steps_taken.append(action_name)
                if success or action_name == "escalate":
                    break
                state = next_state

            row_results[policy_name] = {
                "actions": steps_taken,
                "reward":  round(total_r, 1),
                "esc":     esc,
                "steps":   len(steps_taken),
            }

        def cell(r):
            esc_tag = "🚨" if r["esc"] else "✅"
            acts = "→".join(r["actions"])
            return f"{esc_tag} {acts[:10]:<10} r={r['reward']:+.0f}"

        failure_label = f"{stage}/{error}"
        lines.append(f"  {failure_label:<32} "
                     f"{cell(row_results['baseline']):^14}  "
                     f"{cell(row_results['rule_based']):^14}  "
                     f"{cell(row_results['q_learning']):^14}  "
                     f"{cell(row_results['ucb']):^14}")

    lines.append("\n  Legend: 🚨 = escalated to human   ✅ = auto-resolved")
    lines.append("  r = total episode reward\n")

    agent_ql.epsilon = ql_eps_save

    # write text file
    out_txt = os.path.join(RESULTS_DIR, "sample_interactions.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_txt}")
    return early_traces, late_traces


# ─────────────────────────────────────────────────────────────────────────────
# 2. PNG FIGURE — learning-progress panel
# ─────────────────────────────────────────────────────────────────────────────

def generate_interactions_figure(early_traces, late_traces):
    """
    Two-panel bar figure.
    Left  panel: 3 early-training episodes — actions taken, coloured by outcome.
    Right panel: same 3 failures, late-training — shows convergence to good actions.
    """
    COLOR_MAP = {
        "retry":          "#5B9BD5",
        "revert":         "#ED7D31",
        "auto_fix":       "#70AD47",
        "switch_version": "#FFC000",
        "skip_stage":     "#A9D18E",
        "escalate":       "#FF0000",
    }
    OUTCOME_ALPHA = {"success": 1.0, "fail": 0.45, "escalate": 0.8}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Learning Progress: Early vs Late Training Episodes\n(same failure type, same agent)",
                 fontsize=13, fontweight="bold")

    for ax, traces, title_suffix, phase in [
        (axes[0], early_traces, "Early Training (ep 5, 8, 10)\nε ≈ 0.97 — mostly random", "early"),
        (axes[1], late_traces,  "Late Training  (ep 1988, 1994, 1999)\nε ≈ 0.05 — greedy policy", "late"),
    ]:
        y_ticks, y_labels = [], []
        y = 0
        for ep_num, trace, (stage, error) in traces:
            label = f"Ep {ep_num}\n{stage[:4]}/{error[:8]}"
            y_labels.append(label)
            y_ticks.append(y + 0.4)
            x_start = 0
            for step in trace["steps"]:
                action = step["action"]
                width = 1.0
                if step["success"]:
                    alpha = OUTCOME_ALPHA["success"]
                    edgecolor = "green"
                    lw = 2
                elif action == "escalate":
                    alpha = OUTCOME_ALPHA["escalate"]
                    edgecolor = "darkred"
                    lw = 2
                else:
                    alpha = OUTCOME_ALPHA["fail"]
                    edgecolor = "grey"
                    lw = 0.5

                bar = ax.barh(y, width, left=x_start, height=0.7,
                              color=COLOR_MAP.get(action, "#999"),
                              alpha=alpha, edgecolor=edgecolor, linewidth=lw)
                ax.text(x_start + 0.05, y + 0.35,
                        action[:5], va="center", ha="left",
                        fontsize=7.5, color="black", fontweight="bold")
                x_start += width

            # reward badge
            rew_color = "#2E7D32" if trace["total_reward"] > 0 else "#C62828"
            ax.text(x_start + 0.1, y + 0.35,
                    f"r={trace['total_reward']:+.0f}",
                    va="center", ha="left", fontsize=8,
                    color=rew_color, fontweight="bold")
            y += 1.1

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8.5)
        ax.set_xlabel("Recovery Steps", fontsize=9)
        ax.set_xlim(0, MAX_STEPS + 1.5)
        ax.set_title(title_suffix, fontsize=9.5, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # legend
    legend_patches = [mpatches.Patch(color=c, label=a)
                      for a, c in COLOR_MAP.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    # annotation boxes
    axes[0].annotate("Random exploration\nleads to escalations",
                     xy=(0.5, 0.02), xycoords="axes fraction",
                     ha="center", fontsize=8, color="#C62828",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFE0E0", alpha=0.8))
    axes[1].annotate("Learned policy:\ndirect auto_fix / revert",
                     xy=(0.5, 0.02), xycoords="axes fraction",
                     ha="center", fontsize=8, color="#1B5E20",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_png = os.path.join(RESULTS_DIR, "sample_interactions.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating sample interactions...")
    early, late = generate_text_report()
    generate_interactions_figure(early, late)
    print("Done.")
