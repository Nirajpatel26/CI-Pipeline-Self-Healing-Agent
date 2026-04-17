# tests/test_llm_parse.py — JSON extraction + schema validation

import pytest

from tools.llm_parse import (
    extract_json, safe_call,
    MonitorOutput, RecoveryOutput, ExecutorOutput, ValidatorOutput,
)


# ── extract_json ─────────────────────────────────────────────────────────

def test_extract_json_bare_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_prose():
    text = 'Sure! Here is the answer: {"severity": "high"} — hope this helps.'
    assert extract_json(text) == {"severity": "high"}


def test_extract_json_fenced():
    text = 'Analysis:\n```json\n{"action": "retry"}\n```\nDone.'
    assert extract_json(text) == {"action": "retry"}


def test_extract_json_nested():
    text = 'prefix {"outer": {"inner": 42}} suffix'
    assert extract_json(text) == {"outer": {"inner": 42}}


def test_extract_json_empty_raises():
    with pytest.raises(ValueError):
        extract_json("")


def test_extract_json_no_braces_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


# ── MonitorOutput ────────────────────────────────────────────────────────

def test_monitor_valid():
    out = MonitorOutput.from_dict({"severity": "high", "suggested_category": "escalate"})
    assert out.severity == "high"
    assert out.suggested_category == "escalate"


def test_monitor_missing_fields_uses_defaults():
    out = MonitorOutput.from_dict({})
    assert out.severity == "low"
    assert out.suggested_category == "retry_class"


def test_monitor_unknown_enum_coerced():
    out = MonitorOutput.from_dict({"severity": "CRITICAL", "suggested_category": "revert"})
    # "CRITICAL" does not match → default low; "revert" substring-matches revert_class
    assert out.severity == "low"
    assert out.suggested_category == "revert_class"


# ── RecoveryOutput ───────────────────────────────────────────────────────

def test_recovery_valid():
    out = RecoveryOutput.from_dict({"action": "auto_fix", "reasoning": "best Q"})
    assert out.action == "auto_fix"
    assert out.reasoning == "best Q"


def test_recovery_unknown_action_defaults_to_retry():
    out = RecoveryOutput.from_dict({"action": "nuke_everything"})
    assert out.action == "retry"


# ── ExecutorOutput ───────────────────────────────────────────────────────

def test_executor_valid():
    out = ExecutorOutput.from_dict({"outcome_class": "recovered", "should_continue": False})
    assert out.outcome_class == "recovered"
    assert out.should_continue is False


def test_executor_string_bool_coerced():
    out = ExecutorOutput.from_dict({"outcome_class": "failed", "should_continue": "yes"})
    assert out.should_continue is True


# ── ValidatorOutput ──────────────────────────────────────────────────────

def test_validator_clamped():
    out = ValidatorOutput.from_dict({"reward_adjustment": 99.0})
    assert out.reward_adjustment == 2.0


def test_validator_negative_clamped():
    out = ValidatorOutput.from_dict({"reward_adjustment": -50.0})
    assert out.reward_adjustment == -2.0


def test_validator_invalid_string_defaults_zero():
    out = ValidatorOutput.from_dict({"reward_adjustment": "not a number"})
    assert out.reward_adjustment == 0.0


# ── safe_call ────────────────────────────────────────────────────────────

def test_safe_call_success():
    assert safe_call(lambda: 42, fallback=0) == 42


def test_safe_call_on_exception_returns_fallback():
    def boom():
        raise RuntimeError("nope")
    assert safe_call(boom, fallback="fb") == "fb"


def test_safe_call_on_error_hook():
    seen = []
    def boom():
        raise ValueError("x")
    safe_call(boom, fallback=None, on_error=lambda e: seen.append(type(e).__name__))
    assert seen == ["ValueError"]
