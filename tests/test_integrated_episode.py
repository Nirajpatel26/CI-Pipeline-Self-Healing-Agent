# tests/test_integrated_episode.py — End-to-end integrated episode with stubbed LLMs

import numpy as np
import pytest

from autogen_agents import CIHealingSystem
from pipeline_simulator import PipelineSimulator
from q_learning_agent import QLearningAgent
from tools.llm_parse import (
    MonitorOutput, RecoveryOutput, ExecutorOutput, ValidatorOutput,
)
import tools.llm_agents as llm_agents


@pytest.fixture
def stub_llms(monkeypatch):
    """Replace all 4 LLM wrappers with deterministic stubs — no network."""

    class StubMonitor:
        @staticmethod
        def analyze(state, inspector):
            return MonitorOutput(severity="med", suggested_category="code_fix")

    class StubRecovery:
        @staticmethod
        def choose(state, q_values, monitor, inspector):
            # Always pick auto_fix regardless of Q-values so we can verify that
            # the integrated loop actually consults the LLM and uses its output.
            return RecoveryOutput(action="auto_fix", reasoning="stub")

    class StubExecutor:
        @staticmethod
        def assess(action, success, next_state, attempts):
            return ExecutorOutput(outcome_class="failed", should_continue=True)

    class StubValidator:
        @staticmethod
        def score(state, action, success, base_reward, total_reward):
            return ValidatorOutput(reward_adjustment=0.5, assessment="stub")

    monkeypatch.setattr(llm_agents, "MonitorLLM", StubMonitor)
    monkeypatch.setattr(llm_agents, "RLRecoveryLLM", StubRecovery)
    monkeypatch.setattr(llm_agents, "ExecutorLLM", StubExecutor)
    monkeypatch.setattr(llm_agents, "ValidatorLLM", StubValidator)

    # Also patch the names already imported into autogen_agents
    import autogen_agents as ag
    monkeypatch.setattr(ag, "MonitorLLM", StubMonitor)
    monkeypatch.setattr(ag, "RLRecoveryLLM", StubRecovery)
    monkeypatch.setattr(ag, "ExecutorLLM", StubExecutor)
    monkeypatch.setattr(ag, "ValidatorLLM", StubValidator)


def test_integrated_episode_runs_and_updates_q_table(stub_llms):
    sim = PipelineSimulator(seed=3)
    ql = QLearningAgent(seed=3)
    system = CIHealingSystem(
        simulator=sim,
        rl_agent=ql,
        condition="q_learning_llm",
        use_autogen=False,
        mode="integrated",
        max_steps=5,
    )

    # Force ε=0 so the LLM actually drives action selection via the exploit
    # branch; with ε=1 we'd only see random exploration.
    ql.epsilon = 0.0
    before = ql.q_table.copy()

    result = system.run_episode(train=True)

    assert isinstance(result, dict)
    assert "success" in result and "total_reward" in result
    # Q-table should have at least one updated cell
    assert not np.array_equal(ql.q_table, before), "Q-table did not update"
    # Reward bounds preserved even after validator shaping (+0.5 per step)
    assert -50.0 <= result["total_reward"] <= 50.0


def test_integrated_respects_llm_budget(stub_llms):
    """LLM call count must not exceed llm_budget_per_episode."""
    sim = PipelineSimulator(seed=5)
    ql = QLearningAgent(seed=5)
    ql.epsilon = 0.0  # exploit → triggers RLRecoveryLLM at every step
    system = CIHealingSystem(
        simulator=sim,
        rl_agent=ql,
        condition="q_learning_llm",
        use_autogen=False,
        mode="integrated",
        max_steps=10,
        llm_budget_per_episode=3,
    )

    system.run_episode(train=True)
    assert system._llm_call_count <= 3


def test_off_mode_skips_all_llm_hooks(stub_llms):
    """mode='off' must never call any LLM wrapper."""
    calls = {"n": 0}

    class CountingRecovery:
        @staticmethod
        def choose(*args, **kwargs):
            calls["n"] += 1
            return RecoveryOutput(action="retry")

    import autogen_agents as ag
    ag.RLRecoveryLLM = CountingRecovery

    sim = PipelineSimulator(seed=9)
    ql = QLearningAgent(seed=9)
    system = CIHealingSystem(
        simulator=sim,
        rl_agent=ql,
        condition="q_learning",
        use_autogen=False,
        mode="off",
        max_steps=5,
    )
    system.run_episode(train=True)
    assert calls["n"] == 0
