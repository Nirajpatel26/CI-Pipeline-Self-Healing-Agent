# CI Pipeline Self-Healing Agent

**INFO 7375 — Prompt Engineering for Generative AI**  
Multi-agent AutoGen system that monitors a simulated CI/CD pipeline, detects failures, and learns optimal recovery actions using Reinforcement Learning.

---

## Architecture

```
Monitor Agent → RL Recovery Agent → Executor Agent → Validator Agent
                     ↑                                      ↓
              PipelineStateInspector              reward / episode log
```

| Component | Role |
|-----------|------|
| `PipelineSimulator` | Stochastic 5-stage CI failure generator |
| `QLearningAgent` | Tabular Q-Learning with epsilon-greedy |
| `UCBAgent` | Same Q-table; UCB1 exploration |
| `PipelineStateInspector` | AutoGen tool: risk, recoverability, top-2 actions |
| `CIHealingSystem` | AutoGen GroupChat orchestrator (4 agents) |
| `ExperimentRunner` | Runs baseline / rule-based / Q-Learning / UCB |
| `visualizations.py` | 6 publication-ready plots |

---

## Quick Start (local, no LLM)

```bash
pip install -r requirements.txt
python experiment_runner.py --train 2000 --eval 1000
python visualizations.py
```

Outputs written to `results/`.

## Google Colab (with Llama 3.1 8B)

Open `CI_Healing_Agent_Colab.ipynb` in Google Colab Pro with **L4 GPU**.  
The notebook handles:
1. Installing dependencies
2. Downloading llama.cpp + Llama 3.1 8B GGUF (4-bit, ~4.7 GB)
3. Starting the llama.cpp server on port 8080
4. Running all 4 experimental conditions (≈15–25 min on L4)
5. Generating all 6 plots inline

Set `USE_AUTOGEN = True` in Cell 5 to enable the full GroupChat.

---

## Experimental Conditions

| Condition | Policy | Episodes |
|-----------|--------|----------|
| Baseline | Always retry → escalate after 3 fails | 1000 eval |
| Rule-Based | Hardcoded if-else decision tree | 1000 eval |
| Q-Learning | ε-greedy, 2000 train + 1000 eval | 3000 total |
| UCB | UCB1 exploration, 2000 train + 1000 eval | 3000 total |

---

## Metrics

- Pipeline Recovery Rate (%)
- Mean Recovery Attempts
- Escalation Rate (%)
- Mean Episode Reward
- Convergence Episode (first 100-ep window ≥ 85% recovery)
- Q-Table Coverage (%)

---

## MDP Formulation

- **States:** `(stage × error_type × attempt_num × last_action)` → 5,775 states
- **Actions:** `{retry, revert, auto_fix, switch_version, skip_stage, escalate}`
- **Reward:** scalar ∈ [−10, +10], stage-criticality weighted
- **γ = 0.95**, **α = 0.1**, **ε: 1.0 → 0.05 over 2000 episodes**

---

## Tests

```bash
pytest tests/ -v
```

- `tests/test_simulator.py` — failure generation, state encoding, action probabilities
- `tests/test_q_learning.py` — Bellman update correctness, epsilon decay, UCB counts
- `tests/test_reward.py` — edge cases: security skip penalty, escalation, clamping

---

## Ethical Considerations

- Agent **never** autonomously recovers in HIGH-risk states (requires escalation)
- `skip_stage` on `security_scan` incurs a −5 penalty and sets `safe_for_autonomous_recovery=False`
- Q-tables trained on synthetic data — **retrain before deploying on real pipelines**
- Reward function penalises escalation to discourage over-escalation in production

---

## File Structure

```
CI_Healing_Agent/
├── config.py                  # hyperparameters & constants
├── pipeline_simulator.py      # stochastic failure generator
├── reward_function.py         # scalar reward computation
├── q_learning_agent.py        # ε-greedy Q-Learning
├── ucb_agent.py               # UCB1 Q-Learning
├── autogen_agents.py          # 4-agent GroupChat system
├── experiment_runner.py       # run all 4 conditions
├── visualizations.py          # 6 plots
├── tools/
│   └── pipeline_inspector.py  # AutoGen custom tool
├── tests/
│   ├── test_simulator.py
│   ├── test_q_learning.py
│   └── test_reward.py
├── results/                   # auto-created output dir
├── requirements.txt
└── CI_Healing_Agent_Colab.ipynb
```
