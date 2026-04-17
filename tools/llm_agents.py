# tools/llm_agents.py — 4 LLM wrappers for True Integration
#
# Each wrapper makes a single HTTP call to the local llama.cpp OpenAI-compatible
# endpoint (LLM_BASE_URL) and returns a parsed, validated dataclass from
# tools/llm_parse.py. All failures fall back to deterministic defaults so the
# RL loop never stalls.

import json
from typing import Dict, List, Optional

import requests

from config import (
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE, ACTIONS
)
from tools.llm_parse import (
    extract_json, safe_call,
    MonitorOutput, RecoveryOutput, ExecutorOutput, ValidatorOutput,
)


# ── HTTP helper ─────────────────────────────────────────────────────────────

_CHAT_URL = f"{LLM_BASE_URL}/chat/completions"
_HEADERS = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}


def _chat(system: str, user: str, max_tokens: int = 140, timeout: float = 8.0) -> str:
    """One-shot chat completion. Returns the assistant text or raises."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(_CHAT_URL, headers=_HEADERS, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ── MonitorLLM ──────────────────────────────────────────────────────────────

_MONITOR_SYS = (
    "You are the MonitorAgent in a CI/CD self-healing system. "
    "Given a failed pipeline state and an inspector report, classify the "
    "severity and suggest a recovery category. "
    "Respond with ONLY a JSON object, no prose, no code fences. "
    "Schema: "
    '{"severity": "low"|"med"|"high", '
    '"suggested_category": "retry_class"|"revert_class"|"code_fix"|"switch_env"|"skip"|"escalate"}'
)


class MonitorLLM:
    @staticmethod
    def analyze(state: Dict, inspector: Dict) -> MonitorOutput:
        user = json.dumps({
            "stage": state["stage"],
            "error_type": state["error_type"],
            "attempt_num": state["attempt_num"],
            "last_action": state.get("last_action"),
            "inspector_risk": inspector.get("risk_level"),
            "recoverability": inspector.get("recoverability_score"),
        })
        fallback = MonitorOutput()  # low / retry_class

        def run() -> MonitorOutput:
            text = _chat(_MONITOR_SYS, user, max_tokens=80)
            return MonitorOutput.from_dict(extract_json(text), raw=text)

        return safe_call(run, fallback)


# ── RLRecoveryLLM ───────────────────────────────────────────────────────────

_RECOVERY_SYS = (
    "You are the RLRecoveryAgent. You see Q-values for all candidate "
    "recovery actions plus the MonitorAgent's severity classification. "
    "Pick the best action. Prefer the action with the highest Q-value "
    "unless severity is 'high' and a safer alternative exists (e.g., revert, escalate). "
    "Respond with ONLY a JSON object: "
    f'{{"action": one of {ACTIONS}, "reasoning": "<1 short sentence>"}}'
)


class RLRecoveryLLM:
    @staticmethod
    def choose(
        state: Dict,
        q_values: Dict[str, float],
        monitor: MonitorOutput,
        inspector: Dict,
    ) -> RecoveryOutput:
        # Deterministic fallback: argmax over Q-values
        argmax_action = max(q_values, key=lambda k: q_values[k]) if q_values else "retry"
        fallback = RecoveryOutput(action=argmax_action, reasoning="fallback: argmax Q")

        user = json.dumps({
            "stage": state["stage"],
            "error_type": state["error_type"],
            "attempt_num": state["attempt_num"],
            "q_values": {a: round(v, 4) for a, v in q_values.items()},
            "monitor": {
                "severity": monitor.severity,
                "suggested_category": monitor.suggested_category,
            },
            "inspector_top_actions": inspector.get("top_2_recommended_actions", []),
        })

        def run() -> RecoveryOutput:
            text = _chat(_RECOVERY_SYS, user, max_tokens=100)
            return RecoveryOutput.from_dict(extract_json(text), raw=text)

        return safe_call(run, fallback)


# ── ExecutorLLM ─────────────────────────────────────────────────────────────

_EXECUTOR_SYS = (
    "You are the ExecutorAgent. You receive the outcome of a recovery "
    "action applied to the pipeline. Classify the outcome and decide "
    "whether recovery should continue. "
    "Respond with ONLY JSON: "
    '{"outcome_class": "recovered"|"partial"|"failed", '
    '"should_continue": true|false}'
)


class ExecutorLLM:
    @staticmethod
    def assess(action: str, success: bool, next_state: Dict, attempts: int) -> ExecutorOutput:
        fallback = ExecutorOutput(
            outcome_class="recovered" if success else "failed",
            should_continue=not success,
        )
        user = json.dumps({
            "action": action,
            "success": bool(success),
            "attempts_so_far": attempts,
            "next_state": {
                "stage": next_state["stage"],
                "error_type": next_state["error_type"],
                "attempt_num": next_state["attempt_num"],
            },
        })

        def run() -> ExecutorOutput:
            text = _chat(_EXECUTOR_SYS, user, max_tokens=60)
            return ExecutorOutput.from_dict(extract_json(text), raw=text)

        return safe_call(run, fallback)


# ── ValidatorLLM ────────────────────────────────────────────────────────────

_VALIDATOR_SYS = (
    "You are the ValidatorAgent. Given the base reward computed by a "
    "reward function, suggest a small adjustment (-2 to +2) that reflects "
    "whether the recovery was efficient, excessive, or risky. Positive = "
    "reward efficient recovery; negative = discourage risky or wasteful actions. "
    "Respond with ONLY JSON: "
    '{"reward_adjustment": <float in [-2, 2]>, "assessment": "<1 short sentence>"}'
)


class ValidatorLLM:
    @staticmethod
    def score(
        state: Dict,
        action: str,
        success: bool,
        base_reward: float,
        total_reward: float,
    ) -> ValidatorOutput:
        fallback = ValidatorOutput()  # adjustment=0.0
        user = json.dumps({
            "stage": state["stage"],
            "error_type": state["error_type"],
            "attempt_num": state["attempt_num"],
            "action": action,
            "success": bool(success),
            "base_reward": round(float(base_reward), 3),
            "total_reward_so_far": round(float(total_reward), 3),
        })

        def run() -> ValidatorOutput:
            text = _chat(_VALIDATOR_SYS, user, max_tokens=100)
            return ValidatorOutput.from_dict(extract_json(text), raw=text)

        return safe_call(run, fallback)
