"""
Generates architecture_diagram.png for the CI Pipeline Self-Healing Agent.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("#12192C")
ax.set_facecolor("#12192C")

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#12192C"
C_SIM    = "#1B4F8A"   # steel blue  — CI Environment
C_INSP   = "#1A6B5A"   # teal        — Inspector
C_RL     = "#5B2D8E"   # purple      — RL Agents
C_AGENTS = "#8B4A00"   # amber       — AutoGen agents
C_REWARD = "#1A6130"   # green       — Reward
C_LLM    = "#7A1F1F"   # dark red    — LLM backend
C_EXP    = "#2C3E6B"   # slate blue  — ExperimentRunner
C_BORDER = "#FFFFFF"
C_TEXT   = "#FFFFFF"
C_MUTED  = "#B0BEC5"
C_ARROW  = "#90CAF9"
C_TITLE  = "#E3F2FD"

def box(ax, x, y, w, h, color, label, sublabels=(), radius=0.25, alpha=0.92, fontsize=9.5):
    """Draw a rounded rectangle with a title and optional sub-labels."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={radius}",
                          linewidth=1.5, edgecolor=C_BORDER,
                          facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    # Title bar band
    band = FancyBboxPatch((x, y + h - 0.38), w, 0.38,
                          boxstyle=f"round,pad=0.01,rounding_size={radius}",
                          linewidth=0, facecolor="#FFFFFF22", zorder=4)
    ax.add_patch(band)
    ax.text(x + w/2, y + h - 0.19, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=C_TEXT, zorder=5)
    step = (h - 0.42) / (len(sublabels) + 1) if sublabels else 0
    for i, sl in enumerate(sublabels):
        ay = y + h - 0.42 - step * (i + 1)
        ax.text(x + 0.18, ay, sl,
                ha="left", va="center", fontsize=7.5,
                color=C_MUTED, zorder=5)

def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=14),
                zorder=6)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.13, label, ha="center", va="bottom",
                fontsize=7, color=C_ARROW, zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none", alpha=0.7))

def bidir(ax, x0, y0, x1, y1, label="", color=C_ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<|-|>", color=color,
                                lw=1.6, mutation_scale=14),
                zorder=6)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.13, label, ha="center", va="bottom",
                fontsize=7, color=C_ARROW, zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none", alpha=0.7))

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — CI Environment / Pipeline Simulator  (far left)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 0.3, 4.0, 2.8, 5.5, C_SIM, "PipelineSimulator",
    sublabels=[
        "5 Stages:",
        "  lint → build → test",
        "  security_scan → deploy",
        "",
        "15 error types (3/stage)",
        "Stochastic RECOVERY_PROBS",
        "",
        "generate_failure()",
        "apply_action(state, action)",
        "encode_state()  /  decode_state()",
    ])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — PipelineStateInspector  (below centre-left)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 0.3, 1.0, 2.8, 2.6, C_INSP, "PipelineStateInspector",
    sublabels=[
        "AutoGen Custom Tool",
        "risk_level: LOW|MEDIUM|HIGH",
        "recoverability_score ∈ [0,1]",
        "top_2_recommended_actions",
    ])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — RL Agents  (centre-left)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 3.6, 6.5, 3.2, 3.0, C_RL, "QLearningAgent",
    sublabels=[
        "Q-table: 5,775 × 6  (float64)",
        "ε-greedy: ε = 1.0 → 0.05",
        "  decay = 0.995/episode",
        "Bellman update:",
        "  Q(s,a)←Q(s,a)+α[r+γ·max Q(s′,a′)−Q(s,a)]",
        "α = 0.1   γ = 0.95",
    ])

box(ax, 3.6, 3.2, 3.2, 2.9, C_RL, "UCBAgent",
    sublabels=[
        "Q-table: 5,775 × 6  (float64)",
        "visit_counts: 5,775×6  (init=1)",
        "UCB1 formula:",
        "  a*=argmax[Q(s,a)+c·√(lnN(s)/N(s,a))]",
        "c = 2.0  (no ε schedule)",
    ])

box(ax, 3.6, 1.0, 3.2, 1.85, "#3D3D3D", "Baseline / Rule-Based",
    sublabels=[
        "Baseline: retry×3 → escalate",
        "Rule-Based: error_type→action",
        "  (hardcoded if-else map)",
    ])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — CIHealingSystem / AutoGen agents  (centre)
# ─────────────────────────────────────────────────────────────────────────────
# Outer orchestrator frame
outer = FancyBboxPatch((7.3, 0.5), 4.8, 10.0,
                       boxstyle="round,pad=0.05,rounding_size=0.3",
                       linewidth=2, edgecolor="#FFA726",
                       facecolor="#1A1200", alpha=0.85, zorder=2)
ax.add_patch(outer)
ax.text(9.7, 10.25, "CIHealingSystem  (AutoGen GroupChat Orchestrator)",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#FFA726", zorder=5)

# Four agent boxes inside
agent_data = [
    (7.55, 8.0, "MonitorAgent",
     ["Classifies failure severity",
      "Gate: step==0 | HIGH/MED risk | attempt≥2",
      "Output: severity label, urgency"]),
    (7.55, 5.8, "RLRecoveryAgent",
     ["Selects action via RL argmax",
      "LLM tie-break when top-2 Q-vals",
      "  within 10% OR state unvisited",
      "Actions: retry|revert|auto_fix|",
      "  switch_version|skip_stage|escalate"]),
    (7.55, 3.5, "ExecutorAgent",
     ["Applies action to simulator",
      "LLM triage: fires on fail + attempt≥2",
      "Output: should_continue flag"]),
    (7.55, 1.2, "ValidatorAgent",
     ["Computes & shapes reward",
      "LLM fires every 20th train episode",
      "Output: reward_adjustment ∈[−2,+2]"]),
]
for (bx, by, lbl, subs) in agent_data:
    box(ax, bx, by, 4.3, 1.95, C_AGENTS, lbl, sublabels=subs, fontsize=8.5)

# Vertical flow arrows between agents
for ya, yb in [(9.95, 9.92), (7.98, 7.77), (5.78, 5.47), (3.48, 3.17)]:
    ax.annotate("", xy=(9.7, yb), xytext=(9.7, ya),
                arrowprops=dict(arrowstyle="-|>", color="#FFA726", lw=1.4, mutation_scale=12),
                zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — RewardFunction  (centre-right)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 12.6, 5.5, 2.8, 5.0, C_REWARD, "RewardFunction",
    sublabels=[
        "r ∈ [−10, +10]",
        "",
        "Success:",
        "  attempt 0 → +10",
        "  attempt 1 → +7",
        "  attempt 2 → +5",
        "  attempt 3+ → +3",
        "Failure: −2 × criticality",
        "Escalate: −8 × criticality",
        "skip security_scan: −5",
    ])

# Criticality mini-table
ax.text(13.35, 5.75, "Criticality: lint 0.5 | build 0.7 | test 0.8\n"
        "security_scan 1.5 | deploy 1.3",
        ha="center", va="center", fontsize=6.8,
        color="#A5D6A7", zorder=5,
        bbox=dict(boxstyle="round,pad=0.2", fc="#0D2B0D", ec="#1A6130", alpha=0.9))

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — LLM Backend  (bottom centre-right)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 12.6, 1.0, 2.8, 4.0, C_LLM, "LLM Backend",
    sublabels=[
        "Llama 3.1 8B (4-bit GGUF)",
        "llama.cpp server · port 8081",
        "",
        "≤ 6 LLM calls / episode",
        "Fallback: deterministic default",
        "",
        "MonitorLLM",
        "RLRecoveryLLM",
        "ExecutorLLM",
        "ValidatorLLM",
    ])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — ExperimentRunner  (far right)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 15.9, 5.5, 1.85, 5.0, C_EXP, "ExperimentRunner",
    sublabels=[
        "4 conditions:",
        " baseline",
        " rule_based",
        " q_learning",
        " ucb",
        "",
        "Train: 2,000 ep",
        "Eval:  1,000 ep",
        "seed = 42",
    ], fontsize=8)

box(ax, 15.9, 1.0, 1.85, 4.0, C_EXP, "Results",
    sublabels=[
        "episode_log.csv",
        "summary.json",
        "q_table_ql.npy",
        "q_table_ucb.npy",
        "",
        "6 plots",
    ], fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# ARROWS — data flows
# ─────────────────────────────────────────────────────────────────────────────
# Simulator → Inspector (downward)
arrow(ax, 1.7, 4.0, 1.7, 3.6, "encoded state (int)")

# Simulator → CIHealingSystem (failure event)
arrow(ax, 3.1, 7.0, 7.3, 8.5, "failure event\n(stage, error, attempt)")

# Inspector → CIHealingSystem
arrow(ax, 3.1, 2.2, 7.3, 3.8, "inspection report\n(risk, recoverability, top-2)")

# CIHealingSystem → Simulator (action)
arrow(ax, 7.3, 6.5, 3.1, 6.2, "action string")

# Simulator → CIHealingSystem (outcome)
arrow(ax, 3.1, 5.5, 7.3, 5.0, "(success: bool,\nnext_state)")

# QL/UCB → Healing system (action selection)
arrow(ax, 6.8, 7.6, 7.55, 8.6, "action_idx")
arrow(ax, 6.8, 4.6, 7.55, 6.2, "action_idx")

# CIHealingSystem → RewardFunction
arrow(ax, 12.1, 4.5, 12.6, 6.2, "(state, action, success)")

# RewardFunction → QL agent (reward)
arrow(ax, 12.6, 7.5, 6.8, 7.9, "scalar reward →\nBellman update")

# CIHealingSystem ↔ LLM
bidir(ax, 12.1, 2.5, 12.6, 2.5, "HTTP POST\n/v1/chat/completions")

# ExperimentRunner → CIHealingSystem
arrow(ax, 15.9, 7.5, 12.15, 7.0, "orchestrates\nconditions")

# CIHealingSystem → Results
arrow(ax, 12.15, 2.0, 15.9, 2.5, "episode logs")

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND — Operating Modes
# ─────────────────────────────────────────────────────────────────────────────
legend_x, legend_y = 0.3, 0.02
ax.text(legend_x, legend_y + 0.72, "Operating Modes:",
        fontsize=7.5, fontweight="bold", color=C_TEXT, zorder=8)
modes = [
    ("mode=\"off\"",        "#888888", "Pure RL — no LLM"),
    ("mode=\"narration\"",  "#FFA726", "GroupChat post-hoc every 10th ep"),
    ("mode=\"integrated\"", "#EF5350", "LLM in decision loop (4 hooks)"),
]
for i, (lbl, col, desc) in enumerate(modes):
    ax.plot(legend_x + 0.15, legend_y + 0.48 - i * 0.22, "s",
            color=col, markersize=7, zorder=8)
    ax.text(legend_x + 0.35, legend_y + 0.48 - i * 0.22,
            f"{lbl}  —  {desc}",
            fontsize=6.8, color=C_MUTED, va="center", zorder=8)

# ─────────────────────────────────────────────────────────────────────────────
# MDP SUMMARY BOX
# ─────────────────────────────────────────────────────────────────────────────
mdp_txt = (
    "MDP:  States = 5,775  (5 stages × 15 errors × 11 attempts × 7 last-actions)\n"
    "Actions = 6  {retry, revert, auto_fix, switch_version, skip_stage, escalate}\n"
    "Reward r ∈ [−10, +10]  |  γ = 0.95   α = 0.1   ε: 1.0 → 0.05"
)
ax.text(9.0, 0.3, mdp_txt, ha="center", va="center", fontsize=7.5,
        color="#B3E5FC", zorder=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="#0A1929", ec="#1565C0", lw=1.2, alpha=0.95))

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(9.0, 10.7, "CI Pipeline Self-Healing Agent — System Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=C_TITLE, zorder=8)
ax.text(9.0, 10.4, "INFO 7375  |  RL + AutoGen GroupChat + LLM Integration",
        ha="center", va="center", fontsize=9, color=C_MUTED, zorder=8)

plt.tight_layout(pad=0.1)
OUT = r"D:\NEU\SEM4\Prompt\final exam\CI_Healing_Agent\architecture_diagram.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {OUT}")
