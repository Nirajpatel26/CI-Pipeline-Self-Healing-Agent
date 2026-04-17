# config.py — Centralized hyperparameters and constants

import os

# ── RL Hyperparameters ──────────────────────────────────────────────────────
ALPHA = 0.1           # Q-learning rate
GAMMA = 0.95          # discount factor
EPSILON_START = 1.0   # initial exploration rate
EPSILON_MIN = 0.05    # minimum exploration floor
EPSILON_DECAY = 0.995 # multiplicative decay per episode
UCB_C = 2.0           # UCB exploration constant

# ── Training / Evaluation ───────────────────────────────────────────────────
N_EPISODES = 2000     # Q-Learning / UCB training episodes
EVAL_EPISODES = 1000  # evaluation episodes per condition
MAX_STEPS = 10        # max recovery attempts per episode
RANDOM_SEED = 42

# ── Pipeline Definition ─────────────────────────────────────────────────────
STAGES = ["lint", "build", "test", "security_scan", "deploy"]

# Error types per stage
STAGE_ERRORS = {
    "lint":          ["syntax_error", "style_violation", "import_error"],
    "build":         ["missing_deps", "compile_error", "version_conflict"],
    "test":          ["flaky_test", "assertion_error", "timeout"],
    "security_scan": ["vuln_detected", "license_violation", "secret_exposed"],
    "deploy":        ["rollback_needed", "resource_unavailable", "config_error"],
}

# Flat ordered list of all error types (for state encoding)
ALL_ERRORS = [e for errors in STAGE_ERRORS.values() for e in errors]

# Actions
ACTIONS = ["retry", "revert", "auto_fix", "switch_version", "skip_stage", "escalate"]

# Indices
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}
ERROR_IDX = {e: i for i, e in enumerate(ALL_ERRORS)}
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── MDP State Space ─────────────────────────────────────────────────────────
# State = (stage_idx, error_type_idx, attempt_num, last_action_idx)
N_STAGES = len(STAGES)                   # 5
N_ERRORS = len(ALL_ERRORS)              # 15
N_ATTEMPTS = MAX_STEPS + 1              # 0..10 → 11 values
N_LAST_ACTIONS = len(ACTIONS) + 1       # +1 for "no previous action" (idx=0)
N_ACTIONS = len(ACTIONS)               # 6

# Total states = 5 * 15 * 11 * 7 = 5775 (≥1050 from spec; spec used subset)
N_STATES = N_STAGES * N_ERRORS * N_ATTEMPTS * N_LAST_ACTIONS

# ── Stage Criticality (penalty multiplier) ──────────────────────────────────
STAGE_CRITICALITY = {
    "lint":          0.5,
    "build":         0.7,
    "test":          0.8,
    "security_scan": 1.5,  # highest risk
    "deploy":        1.3,
}

# ── Recovery Probability Table ──────────────────────────────────────────────
# {(stage, error_type, action): success_probability}
RECOVERY_PROBS = {
    # lint
    ("lint", "syntax_error",     "auto_fix"):       0.92,
    ("lint", "syntax_error",     "retry"):          0.30,
    ("lint", "syntax_error",     "revert"):         0.60,
    ("lint", "syntax_error",     "switch_version"): 0.20,
    ("lint", "syntax_error",     "skip_stage"):     0.95,
    ("lint", "syntax_error",     "escalate"):       0.99,
    ("lint", "style_violation",  "auto_fix"):       0.85,
    ("lint", "style_violation",  "retry"):          0.20,
    ("lint", "style_violation",  "skip_stage"):     0.90,
    ("lint", "style_violation",  "escalate"):       0.99,
    ("lint", "import_error",     "auto_fix"):       0.70,
    ("lint", "import_error",     "switch_version"): 0.65,
    ("lint", "import_error",     "retry"):          0.25,
    ("lint", "import_error",     "escalate"):       0.99,
    # build
    ("build", "missing_deps",     "switch_version"): 0.82,
    ("build", "missing_deps",     "auto_fix"):       0.75,
    ("build", "missing_deps",     "retry"):          0.40,
    ("build", "missing_deps",     "escalate"):       0.99,
    ("build", "compile_error",    "auto_fix"):       0.68,
    ("build", "compile_error",    "revert"):         0.72,
    ("build", "compile_error",    "retry"):          0.25,
    ("build", "compile_error",    "escalate"):       0.99,
    ("build", "version_conflict", "switch_version"): 0.88,
    ("build", "version_conflict", "revert"):         0.70,
    ("build", "version_conflict", "retry"):          0.15,
    ("build", "version_conflict", "escalate"):       0.99,
    # test
    ("test", "flaky_test",       "retry"):          0.65,
    ("test", "flaky_test",       "auto_fix"):       0.55,
    ("test", "flaky_test",       "skip_stage"):     0.90,
    ("test", "flaky_test",       "escalate"):       0.99,
    ("test", "assertion_error",  "auto_fix"):       0.72,
    ("test", "assertion_error",  "revert"):         0.68,
    ("test", "assertion_error",  "retry"):          0.20,
    ("test", "assertion_error",  "escalate"):       0.99,
    ("test", "timeout",          "retry"):          0.50,
    ("test", "timeout",          "switch_version"): 0.60,
    ("test", "timeout",          "skip_stage"):     0.88,
    ("test", "timeout",          "escalate"):       0.99,
    # security_scan
    ("security_scan", "vuln_detected",    "auto_fix"):       0.60,
    ("security_scan", "vuln_detected",    "switch_version"): 0.70,
    ("security_scan", "vuln_detected",    "revert"):         0.75,
    ("security_scan", "vuln_detected",    "retry"):          0.10,
    ("security_scan", "vuln_detected",    "skip_stage"):     0.80,  # HIGH risk
    ("security_scan", "vuln_detected",    "escalate"):       0.99,
    ("security_scan", "license_violation","auto_fix"):       0.50,
    ("security_scan", "license_violation","skip_stage"):     0.80,
    ("security_scan", "license_violation","escalate"):       0.99,
    ("security_scan", "secret_exposed",   "revert"):         0.85,
    ("security_scan", "secret_exposed",   "auto_fix"):       0.55,
    ("security_scan", "secret_exposed",   "escalate"):       0.99,
    # deploy
    ("deploy", "rollback_needed",      "revert"):         0.96,
    ("deploy", "rollback_needed",      "retry"):          0.10,
    ("deploy", "rollback_needed",      "auto_fix"):       0.40,
    ("deploy", "rollback_needed",      "escalate"):       0.99,
    ("deploy", "resource_unavailable", "retry"):          0.55,
    ("deploy", "resource_unavailable", "switch_version"): 0.65,
    ("deploy", "resource_unavailable", "escalate"):       0.99,
    ("deploy", "config_error",         "auto_fix"):       0.78,
    ("deploy", "config_error",         "revert"):         0.82,
    ("deploy", "config_error",         "retry"):          0.30,
    ("deploy", "config_error",         "escalate"):       0.99,
}
# Default probability for unlisted (stage, error, action) combos
DEFAULT_RECOVERY_PROB = 0.15

# ── Llama.cpp Server Config ─────────────────────────────────────────────────
LLM_PORT = int(os.environ.get("LLAMA_PORT", 8081))
LLM_BASE_URL = f"http://localhost:{LLM_PORT}/v1"
LLM_MODEL = "llama-3.1-8b"
LLM_API_KEY = "none"
LLM_TEMPERATURE = 0.1
LLM_SEED = 42

# ── Results Directory ───────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
