# autogen_agents.py — AutoGen GroupChat multi-agent setup for CI healing

import json
import numpy as np
from typing import Dict, Optional, Union

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

from config import (
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE, LLM_SEED,
    ACTIONS, ACTION_IDX
)
from pipeline_simulator import PipelineSimulator
from reward_function import compute_reward
from tools.pipeline_inspector import PipelineStateInspector
from tools.llm_agents import MonitorLLM, RLRecoveryLLM, ExecutorLLM, ValidatorLLM
from tools.llm_parse import MonitorOutput


# ── LLM Configuration ─────────────────────────────────────────────────────

LLM_CONFIG = {
    "config_list": [
        {
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "api_key": LLM_API_KEY,
            "price": [0, 0],   # suppress "model not found" cost warning for local LLM
        }
    ],
    "temperature": LLM_TEMPERATURE,
    "seed": LLM_SEED,
}

# ── AutoGen Narration Frequency ──────────────────────────────────────────
# Run the full 4-agent GroupChat every Nth episode.
# Episodes in between use the fast Python loop only.
# Set to 1 to narrate every episode (slow), 10 for every 10th (recommended).
AUTOGEN_CHAT_EVERY_N = 10


# ── CI Healing Multi-Agent System ─────────────────────────────────────────

class CIHealingSystem:
    """
    Orchestrates 4 AutoGen agents via GroupChat to handle one pipeline episode:

    1. MonitorAgent    — parses raw failure → structured state JSON
    2. RLRecoveryAgent — uses inspector + RL agent to pick an action
    3. ExecutorAgent   — applies the action to the simulator
    4. ValidatorAgent  — computes reward, logs, decides termination

    Falls back to a lightweight Python loop when AutoGen is not installed
    (useful for local unit testing without the full AutoGen dependency).
    """

    def __init__(
        self,
        simulator: PipelineSimulator,
        rl_agent=None,        # QLearningAgent or UCBAgent (None for baseline/rule)
        condition: str = "q_learning",   # "baseline" | "rule_based" | "q_learning" | "ucb" | "q_learning_llm"
        use_autogen: bool = True,
        max_steps: int = 10,
        mode: Optional[str] = None,      # "off" | "narration" | "integrated"
        llm_budget_per_episode: int = 6,
    ):
        self.simulator = simulator
        self.rl_agent = rl_agent
        self.condition = condition
        self.max_steps = max_steps
        self.llm_budget_per_episode = llm_budget_per_episode

        # Resolve mode. Back-compat: if caller set use_autogen=True and no explicit
        # mode, keep the old narration behavior.
        if mode is None:
            mode = "narration" if use_autogen else "off"
        assert mode in ("off", "narration", "integrated"), f"Unknown mode: {mode}"
        self.mode = mode

        # "narration" still requires the pyautogen package; "integrated" does NOT
        # (it uses plain HTTP via tools/llm_agents.py).
        self.use_autogen = (mode == "narration") and AUTOGEN_AVAILABLE
        self.inspector = PipelineStateInspector(agent=rl_agent)

        if self.use_autogen:
            self._build_agents()

    # ── Agent Construction ────────────────────────────────────────────────

    def _build_agents(self):
        self.monitor_agent = AssistantAgent(
            name="MonitorAgent",
            system_message=(
                "You are the MonitorAgent in a CI pipeline self-healing system. "
                "You receive a CI failure state and its inspector report with real data "
                "from the pipeline simulator. Analyze the failure: identify the stage, "
                "error type, risk level, and recoverability. Summarize in 2-3 sentences. "
                "Focus on what went wrong and how critical it is."
            ),
            llm_config=LLM_CONFIG,
        )

        self.rl_recovery_agent = AssistantAgent(
            name="RLRecoveryAgent",
            system_message=(
                "You are the RLRecoveryAgent. You receive the chosen recovery action "
                "selected by the RL policy (Q-Learning or UCB) along with the inspector's "
                f"top recommendations. Available actions: {ACTIONS}. "
                "Comment on the action choice in 2-3 sentences: explain whether it aligns "
                "with the inspector's recommendations, and why the RL agent may have "
                "preferred it given the Q-values and exploration strategy."
            ),
            llm_config=LLM_CONFIG,
        )

        self.executor_agent = AssistantAgent(
            name="ExecutorAgent",
            system_message=(
                "You are the ExecutorAgent. You receive the real outcome of applying a "
                "recovery action to the CI pipeline simulator (success/failure and the "
                "resulting next state). Report what happened in 2-3 sentences: was the "
                "action effective? Did the pipeline recover or does it need further attempts?"
            ),
            llm_config=LLM_CONFIG,
        )

        self.validator_agent = AssistantAgent(
            name="ValidatorAgent",
            system_message=(
                "You are the ValidatorAgent. You receive the complete episode summary "
                "with real rewards computed by the reward function. Assess the overall "
                "recovery quality in 2-3 sentences: was the recovery efficient? How does "
                "the total reward reflect the agent's performance? Should the strategy "
                "be adjusted for similar failures in the future?"
            ),
            llm_config=LLM_CONFIG,
        )

    # ── Episode Execution ─────────────────────────────────────────────────

    def run_episode(self, train: bool = True) -> Dict:
        """
        Run one full pipeline failure + recovery episode.

        Returns:
            {
                "total_reward": float,
                "steps": int,
                "success": bool,
                "escalated": bool,
                "actions_taken": [str],
            }
        """
        state = self.simulator.generate_failure()

        if self.mode == "integrated":
            return self._run_episode_integrated(state, train)
        if self.use_autogen:
            return self._run_episode_autogen(state, train)
        return self._run_episode_python(state, train)

    # ── Python fallback loop (no AutoGen dependency) ──────────────────────

    def _run_episode_python(self, initial_state: Dict, train: bool) -> Dict:
        state = initial_state
        total_reward = 0.0
        actions_taken = []
        success = False
        escalated = False

        for step in range(self.max_steps):
            state_idx = PipelineSimulator.encode_state(state)
            inspection = self.inspector.inspect(state, state_idx)

            # Action selection
            action = self._select_action(state, state_idx, inspection)
            actions_taken.append(action)

            if action == "escalate":
                escalated = True

            # Apply action
            outcome_success, next_state = self.simulator.apply_action(state, action)

            # Reward
            reward = compute_reward(state, action, outcome_success)
            total_reward += reward

            next_state_idx = PipelineSimulator.encode_state(next_state)
            done = outcome_success or action == "escalate" or step == self.max_steps - 1

            # RL update
            if train and self.rl_agent is not None:
                action_idx = ACTION_IDX[action]
                self.rl_agent.update(state_idx, action_idx, reward, next_state_idx, done)

            state = next_state
            if done:
                success = outcome_success
                break

        if train and self.rl_agent is not None and hasattr(self.rl_agent, "decay_epsilon"):
            self.rl_agent.decay_epsilon()

        return {
            "total_reward": total_reward,
            "steps": len(actions_taken),
            "success": success,
            "escalated": escalated,
            "actions_taken": actions_taken,
        }

    def _select_action(self, state: Dict, state_idx: int, inspection: Dict) -> str:
        """Route action selection to the appropriate condition policy.

        Note: the previous implementation forced ``escalate`` whenever the
        inspector reported ``safe_for_autonomous_recovery == False``. That
        starved the Q-table of signal on HIGH-risk states and produced the
        0% RecovRate% observed in prior runs. The inspector's risk is now
        advisory only — the RL agent (and, in integrated mode, the LLM)
        makes the final call.
        """
        if self.condition == "baseline":
            return self._baseline_policy(state)
        if self.condition == "rule_based":
            return self._rule_based_policy(state)
        # q_learning or ucb: let the RL agent choose from the full action set.
        if self.rl_agent is not None:
            action_idx = self.rl_agent.select_action(state_idx)
            return ACTIONS[action_idx]
        return "escalate"

    @staticmethod
    def _baseline_policy(state: Dict) -> str:
        """Always retry up to 3 times, then escalate."""
        return "retry" if state["attempt_num"] < 3 else "escalate"

    @staticmethod
    def _rule_based_policy(state: Dict) -> str:
        """Hardcoded decision tree."""
        error = state["error_type"]
        stage = state["stage"]
        rules = {
            "syntax_error":      "auto_fix",
            "style_violation":   "auto_fix",
            "import_error":      "switch_version",
            "missing_deps":      "switch_version",
            "compile_error":     "revert",
            "version_conflict":  "switch_version",
            "flaky_test":        "retry",
            "assertion_error":   "auto_fix",
            "timeout":           "retry",
            "vuln_detected":     "revert",
            "license_violation": "escalate",
            "secret_exposed":    "revert",
            "rollback_needed":   "revert",
            "resource_unavailable": "retry",
            "config_error":      "auto_fix",
        }
        return rules.get(error, "escalate")

    # ── AutoGen Hybrid Loop (real computation + GroupChat narration) ─────

    def _run_episode_autogen(self, initial_state: Dict, train: bool) -> Dict:
        """
        Hybrid approach: Python loop for correct computation, AutoGen
        GroupChat for multi-agent narration every Nth episode.

        The simulation loop is identical to _run_episode_python — it calls
        the real simulator, reward function, and Q-table updates.  Every
        AUTOGEN_CHAT_EVERY_N episodes the 4 AutoGen agents discuss the
        real episode data via GroupChat (visible in notebook output).
        Results always come from Python, never from LLM text parsing.
        """
        state = initial_state
        total_reward = 0.0
        actions_taken = []
        success = False
        escalated = False
        step_log = []  # collect per-step data for GroupChat narration

        for step in range(self.max_steps):
            state_idx = PipelineSimulator.encode_state(state)
            inspection = self.inspector.inspect(state, state_idx)

            # Action selection (same policy routing as Python path)
            action = self._select_action(state, state_idx, inspection)
            actions_taken.append(action)

            if action == "escalate":
                escalated = True

            # Apply action via the REAL simulator
            outcome_success, next_state = self.simulator.apply_action(state, action)

            # Compute reward via the REAL reward function
            reward = compute_reward(state, action, outcome_success)
            total_reward += reward

            next_state_idx = PipelineSimulator.encode_state(next_state)
            done = outcome_success or action == "escalate" or step == self.max_steps - 1

            # RL Q-table update (fixes Bug 3: Q-table now actually learns)
            if train and self.rl_agent is not None:
                action_idx = ACTION_IDX[action]
                self.rl_agent.update(state_idx, action_idx, reward, next_state_idx, done)

            # Log step data for narration
            step_log.append({
                "step": step + 1,
                "state": state,
                "inspection_risk": inspection.get("risk_level", "UNKNOWN"),
                "recoverability": inspection.get("recoverability_score", 0.0),
                "top_actions": inspection.get("top_2_recommended_actions", []),
                "action": action,
                "outcome": "SUCCESS" if outcome_success else "FAILED",
                "reward": reward,
                "done": done,
            })

            state = next_state
            if done:
                success = outcome_success
                break

        if train and self.rl_agent is not None and hasattr(self.rl_agent, "decay_epsilon"):
            self.rl_agent.decay_epsilon()

        # ── AutoGen GroupChat narration (every Nth episode) ──────────────
        self._episode_count = getattr(self, "_episode_count", 0) + 1
        if self._episode_count % AUTOGEN_CHAT_EVERY_N == 0:
            self._narrate_episode(
                initial_state, step_log, total_reward, success, escalated
            )

        return {
            "total_reward": total_reward,
            "steps": len(actions_taken),
            "success": success,
            "escalated": escalated,
            "actions_taken": actions_taken,
        }

    # ── True Integration: LLM in the Decision Loop ───────────────────────

    def _run_episode_integrated(self, initial_state: Dict, train: bool) -> Dict:
        """
        LLM-in-the-loop episode:

            MonitorLLM  → classifies severity
            RLRecoveryLLM → picks action using Q-values + Monitor output
            Simulator   → applies action
            ExecutorLLM → triages outcome (optional early exit)
            ValidatorLLM → shapes reward before Q-update

        All 4 hooks are *gated* to keep total LLM calls manageable on a
        local llama.cpp server. Each hook falls back to a deterministic
        default on any error, so a dead LLM never breaks training.
        """
        state = initial_state
        total_reward = 0.0
        actions_taken = []
        success = False
        escalated = False
        self._llm_call_count = 0
        self._episode_count = getattr(self, "_episode_count", 0) + 1

        for step in range(self.max_steps):
            state_idx = PipelineSimulator.encode_state(state)
            inspection = self.inspector.inspect(state, state_idx)

            # ── Hook 1: MonitorLLM (gated) ─────────────────────────────────
            monitor_out = self._maybe_monitor(state, inspection, step)

            # ── Hook 2: Action selection (RL + optional RLRecoveryLLM) ────
            action = self._integrated_select_action(
                state, state_idx, inspection, monitor_out, train, step
            )
            actions_taken.append(action)
            if action == "escalate":
                escalated = True

            # ── Apply action via the real simulator ───────────────────────
            outcome_success, next_state = self.simulator.apply_action(state, action)

            # ── Hook 3: ExecutorLLM (gated) ───────────────────────────────
            executor_out = self._maybe_executor(
                action, outcome_success, next_state, state["attempt_num"]
            )

            # ── Base reward ───────────────────────────────────────────────
            base_reward = compute_reward(state, action, outcome_success)

            # ── Hook 4: ValidatorLLM (gated) ──────────────────────────────
            validator_out = self._maybe_validator(
                state, action, outcome_success, base_reward, total_reward, train
            )
            adjustment = validator_out.reward_adjustment if validator_out else 0.0
            reward = max(-10.0, min(10.0, base_reward + adjustment))
            total_reward += reward

            next_state_idx = PipelineSimulator.encode_state(next_state)
            done = (
                outcome_success
                or action == "escalate"
                or step == self.max_steps - 1
            )
            # ExecutorLLM may also hint at an early exit after a success.
            if executor_out is not None and outcome_success and not executor_out.should_continue:
                done = True

            # ── Q-update: ALWAYS on the action actually taken ─────────────
            if train and self.rl_agent is not None:
                action_idx = ACTION_IDX[action]
                self.rl_agent.update(
                    state_idx, action_idx, reward, next_state_idx, done
                )

            state = next_state
            if done:
                success = outcome_success
                break

        if train and self.rl_agent is not None and hasattr(self.rl_agent, "decay_epsilon"):
            self.rl_agent.decay_epsilon()

        return {
            "total_reward": total_reward,
            "steps": len(actions_taken),
            "success": success,
            "escalated": escalated,
            "actions_taken": actions_taken,
        }

    def _integrated_select_action(
        self,
        state: Dict,
        state_idx: int,
        inspection: Dict,
        monitor_out: Optional[MonitorOutput],
        train: bool,
        step: int,
    ) -> str:
        """Action selection for integrated mode.

        - baseline / rule_based: unchanged.
        - q_learning / ucb / q_learning_llm during training: ε-greedy exploration
          still drives discovery; the LLM only enters the exploit branch.
        - On exploit: if the top-2 Q-values are within 10% of each other or the
          state is unvisited (all zeros), ask RLRecoveryLLM to break the tie.
          Otherwise fall back to plain argmax.
        """
        if self.condition == "baseline":
            return self._baseline_policy(state)
        if self.condition == "rule_based":
            return self._rule_based_policy(state)
        if self.rl_agent is None:
            return "escalate"

        # ε-random exploration during training
        epsilon = getattr(self.rl_agent, "epsilon", 0.0)
        if train and epsilon > 0.0:
            rng = getattr(self.rl_agent, "rng", None)
            if rng is not None and rng.random() < epsilon:
                idx = int(rng.integers(0, len(ACTIONS)))
                return ACTIONS[idx]

        # Exploit: look at Q-values and decide whether to ask the LLM
        q_row = self.rl_agent.q_table[state_idx]
        q_values = {ACTIONS[i]: float(q_row[i]) for i in range(len(ACTIONS))}
        sorted_q = sorted(q_values.values(), reverse=True)
        unvisited = bool(np.all(q_row == 0.0))
        close_call = (
            len(sorted_q) >= 2
            and abs(sorted_q[0]) > 1e-9
            and (sorted_q[0] - sorted_q[1]) / (abs(sorted_q[0]) + 1e-9) < 0.10
        )

        if self._llm_enabled() and (unvisited or close_call):
            if self._llm_call_count < self.llm_budget_per_episode:
                self._llm_call_count += 1
                monitor_in = monitor_out or MonitorOutput()
                rec = RLRecoveryLLM.choose(state, q_values, monitor_in, inspection)
                return rec.action

        # Fallback: pure argmax
        action_idx = int(np.argmax(q_row))
        return ACTIONS[action_idx]

    # ── Gated hook helpers ───────────────────────────────────────────────

    def _llm_enabled(self) -> bool:
        return self.mode == "integrated"

    def _maybe_monitor(self, state: Dict, inspection: Dict, step: int):
        if not self._llm_enabled():
            return None
        risk = inspection.get("risk_level", "LOW")
        gate = (step == 0) or (risk in ("MEDIUM", "HIGH")) or (state["attempt_num"] >= 2)
        if not gate or self._llm_call_count >= self.llm_budget_per_episode:
            return None
        self._llm_call_count += 1
        return MonitorLLM.analyze(state, inspection)

    def _maybe_executor(self, action: str, success: bool, next_state: Dict, attempts: int):
        if not self._llm_enabled():
            return None
        # Triage moment: failed action with some attempts behind us.
        if success or attempts < 2:
            return None
        if self._llm_call_count >= self.llm_budget_per_episode:
            return None
        self._llm_call_count += 1
        return ExecutorLLM.assess(action, success, next_state, attempts)

    def _maybe_validator(
        self,
        state: Dict,
        action: str,
        success: bool,
        base_reward: float,
        total_reward: float,
        train: bool,
    ):
        if not self._llm_enabled() or not train:
            return None
        # Fire only every 20th training episode to keep LLM volume low.
        if (self._episode_count % 20) != 0:
            return None
        if self._llm_call_count >= self.llm_budget_per_episode:
            return None
        self._llm_call_count += 1
        return ValidatorLLM.score(state, action, success, base_reward, total_reward)

    # ── GroupChat Narration ───────────────────────────────────────────────

    def _narrate_episode(
        self,
        initial_state: Dict,
        step_log: list,
        total_reward: float,
        success: bool,
        escalated: bool,
    ):
        """
        Run a single AutoGen GroupChat round where all 4 agents discuss
        the completed episode using real simulator data.

        Creates a fresh GroupChat with max_round=5 (orchestrator + 4 agents).
        Wrapped in try/except so LLM errors never crash the training loop.
        """
        try:
            user_proxy = UserProxyAgent(
                name="Orchestrator",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )

            group_chat = GroupChat(
                agents=[
                    user_proxy,
                    self.monitor_agent,
                    self.rl_recovery_agent,
                    self.executor_agent,
                    self.validator_agent,
                ],
                messages=[],
                max_round=5,
                speaker_selection_method="round_robin",
            )
            manager = GroupChatManager(
                groupchat=group_chat, llm_config=LLM_CONFIG
            )

            # Build a single message with all real step data
            steps_text = []
            for s in step_log:
                steps_text.append(
                    f"  Step {s['step']}: "
                    f"state={{stage={s['state']['stage']}, "
                    f"error={s['state']['error_type']}, "
                    f"attempt={s['state']['attempt_num']}}}, "
                    f"risk={s['inspection_risk']}, "
                    f"recoverability={s['recoverability']:.2f}, "
                    f"top_recommended={[a.get('action','?') for a in s['top_actions']]}, "
                    f"action_taken={s['action']}, "
                    f"outcome={s['outcome']}, "
                    f"reward={s['reward']:+.1f}"
                )

            message = (
                f"=== Episode Report (real simulator data) ===\n"
                f"Initial failure: stage={initial_state['stage']}, "
                f"error={initial_state['error_type']}\n"
                f"Recovery steps:\n" + "\n".join(steps_text) + "\n"
                f"Result: total_reward={total_reward:+.2f}, "
                f"success={success}, escalated={escalated}\n\n"
                f"MonitorAgent: analyze the failure and risk. "
                f"RLRecoveryAgent: comment on the action choices. "
                f"ExecutorAgent: summarize the execution outcomes. "
                f"ValidatorAgent: assess overall recovery quality."
            )

            user_proxy.initiate_chat(manager, message=message)

        except Exception as e:
            # LLM errors must never crash the training loop
            print(f"  [AutoGen narration skipped: {e}]")
