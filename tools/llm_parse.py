# tools/llm_parse.py — Robust JSON extraction + validation for LLM outputs
#
# Lightweight, dependency-free schemas (no pydantic) since requirements.txt
# lists only numpy/pandas/matplotlib/seaborn/pytest/requests/pyautogen.

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List

from config import ACTIONS


# ── JSON extraction ─────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict:
    """
    Best-effort JSON object extraction from a free-form LLM response.

    Tries, in order:
      1. Fenced ```json ... ``` block
      2. Direct json.loads on the whole text
      3. Stack-based brace matching to find the first balanced {...}

    Raises:
        ValueError if no parseable JSON object is found.
    """
    if not text:
        raise ValueError("empty text")

    # 1. Fenced block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Whole thing
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. Stack-based brace matcher for the first balanced {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise ValueError("no parseable JSON object in text")


# ── Enum coercion helper ────────────────────────────────────────────────────

def _coerce_enum(value: Any, allowed: List[str], default: str) -> str:
    """Coerce value to a member of allowed; case-insensitive substring fallback."""
    if not isinstance(value, str):
        return default
    v = value.strip().lower()
    for a in allowed:
        if v == a.lower():
            return a
    # substring match
    for a in allowed:
        if a.lower() in v or v in a.lower():
            return a
    return default


def _coerce_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


# ── Schema dataclasses ──────────────────────────────────────────────────────

_SEVERITIES = ["low", "med", "high"]
_CATEGORIES = [
    "retry_class", "revert_class", "code_fix", "switch_env", "skip", "escalate"
]
_OUTCOMES = ["recovered", "partial", "failed"]


@dataclass
class MonitorOutput:
    severity: str = "low"
    suggested_category: str = "retry_class"
    raw: str = ""

    @classmethod
    def from_dict(cls, d: dict, raw: str = "") -> "MonitorOutput":
        return cls(
            severity=_coerce_enum(d.get("severity"), _SEVERITIES, "low"),
            suggested_category=_coerce_enum(
                d.get("suggested_category"), _CATEGORIES, "retry_class"
            ),
            raw=raw,
        )


@dataclass
class RecoveryOutput:
    action: str = "retry"
    reasoning: str = ""
    raw: str = ""

    @classmethod
    def from_dict(cls, d: dict, raw: str = "") -> "RecoveryOutput":
        return cls(
            action=_coerce_enum(d.get("action"), ACTIONS, "retry"),
            reasoning=str(d.get("reasoning", ""))[:300],
            raw=raw,
        )


@dataclass
class ExecutorOutput:
    outcome_class: str = "failed"
    should_continue: bool = True
    raw: str = ""

    @classmethod
    def from_dict(cls, d: dict, raw: str = "") -> "ExecutorOutput":
        cont = d.get("should_continue", True)
        if isinstance(cont, str):
            cont = cont.strip().lower() in ("true", "yes", "1")
        return cls(
            outcome_class=_coerce_enum(d.get("outcome_class"), _OUTCOMES, "failed"),
            should_continue=bool(cont),
            raw=raw,
        )


@dataclass
class ValidatorOutput:
    reward_adjustment: float = 0.0
    assessment: str = ""
    raw: str = ""

    @classmethod
    def from_dict(cls, d: dict, raw: str = "") -> "ValidatorOutput":
        return cls(
            reward_adjustment=_coerce_float(d.get("reward_adjustment"), -2.0, 2.0, 0.0),
            assessment=str(d.get("assessment", ""))[:300],
            raw=raw,
        )


# ── safe_call wrapper ───────────────────────────────────────────────────────

def safe_call(fn: Callable[[], Any], fallback: Any, on_error: Optional[Callable[[Exception], None]] = None) -> Any:
    """
    Run fn() with blanket exception handling. On any error, return fallback.
    Used to keep LLM failures from ever crashing the training loop.
    """
    try:
        return fn()
    except Exception as e:  # broad by design: network, JSON, validation, timeout
        if on_error is not None:
            try:
                on_error(e)
            except Exception:
                pass
        return fallback
