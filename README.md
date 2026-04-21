<div align="center">

# 🤖 CI Pipeline Self-Healing Agent

### *An Autonomous Multi-Agent System for CI/CD Recovery via Reinforcement Learning*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AutoGen](https://img.shields.io/badge/AutoGen-0.2.35-00A67E?style=for-the-badge)
![Llama](https://img.shields.io/badge/Llama-3.1--8B-FF6F00?style=for-the-badge&logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**INFO 7375 — Prompt Engineering for Generative AI · Final Project**
*Northeastern University · Spring 2026*

[Overview](#-overview) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Experiments](#-experimental-conditions) • [Results](#-metrics) • [Ethics](#-ethical-considerations)

</div>

---

## 🎯 Overview

**CI Pipeline Self-Healing Agent** is a research-grade system that fuses **multi-agent LLM orchestration** (AutoGen) with **tabular Reinforcement Learning** (Q-Learning & UCB1) to autonomously monitor, diagnose, and recover a simulated CI/CD pipeline. Four specialized agents collaborate through a GroupChat to detect failures, select optimal recovery actions from a learned policy, execute them safely, and validate the outcome — all while respecting human-in-the-loop guardrails for high-risk states.

> 💡 *Why?* Modern CI/CD pipelines fail constantly — flaky tests, dependency drift, infrastructure hiccups. Engineers burn hours triaging the same patterns. This project explores whether a **learned policy** can replace brittle if-else runbooks.

---

## 🏗️ Architecture

```
 ┌─────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
 │   Monitor   │──▶ │  RL Recovery     │──▶ │   Executor   │──▶ │  Validator   │
 │   Agent     │    │  Agent (Q/UCB)   │    │   Agent      │    │   Agent      │
 └─────────────┘    └──────────────────┘    └──────────────┘    └──────────────┘
                            ▲                                            │
                            │                                            ▼
                  ┌─────────────────────┐                      ┌─────────────────┐
                  │ PipelineStateInsp.  │                      │ reward + log    │
                  │ (AutoGen Tool)      │                      │ (episode trace) │
                  └─────────────────────┘                      └─────────────────┘
```

| Component | Role |
|-----------|------|
| `PipelineSimulator` | Stochastic 5-stage CI failure generator |
| `QLearningAgent` | Tabular Q-Learning with ε-greedy exploration |
| `UCBAgent` | Same Q-table, UCB1 confidence-bound exploration |
| `PipelineStateInspector` | Custom AutoGen tool: risk, recoverability, top-2 actions |
| `CIHealingSystem` | 4-agent AutoGen GroupChat orchestrator |
| `ExperimentRunner` | Runs baseline / rule-based / Q-Learning / UCB |
| `visualizations.py` | 6 publication-ready plots |

---

## 🚀 Quick Start

### Local Run (no LLM required)

```bash
pip install -r requirements.txt
python experiment_runner.py --train 2000 --eval 1000
python visualizations.py
```

All outputs land in `results/`.

### Google Colab (with Llama 3.1 8B)

Open [`CI_Healing_Agent_Colab.ipynb`](CI_Healing_Agent_Colab.ipynb) in **Colab Pro with L4 GPU**. The notebook:

1. Installs dependencies
2. Downloads llama.cpp + Llama 3.1 8B GGUF (4-bit, ~4.7 GB)
3. Starts the llama.cpp server on port 8080
4. Runs all 4 experimental conditions (~15–25 min on L4)
5. Renders all 6 plots inline

Set `USE_AUTOGEN = True` in Cell 5 to enable the full GroupChat.

---

## 🧪 Experimental Conditions

| Condition | Policy | Episodes |
|-----------|--------|----------|
| **Baseline** | Always retry → escalate after 3 fails | 1,000 eval |
| **Rule-Based** | Hardcoded if-else decision tree | 1,000 eval |
| **Q-Learning** | ε-greedy, 2,000 train + 1,000 eval | 3,000 total |
| **UCB** | UCB1 exploration, 2,000 train + 1,000 eval | 3,000 total |

---

## 📊 Metrics

- ✅ **Pipeline Recovery Rate** (%)
- 🔁 **Mean Recovery Attempts**
- 🚨 **Escalation Rate** (%)
- 💰 **Mean Episode Reward**
- 📈 **Convergence Episode** (first 100-ep window ≥ 85% recovery)
- 🧠 **Q-Table Coverage** (%)

---

## 🧮 MDP Formulation

| Element | Definition |
|--------|-----------|
| **States** | `(stage × error_type × attempt_num × last_action)` → **5,775 states** |
| **Actions** | `{retry, revert, auto_fix, switch_version, skip_stage, escalate}` |
| **Reward** | scalar ∈ `[−10, +10]`, stage-criticality weighted |
| **γ** | 0.95 |
| **α** | 0.1 |
| **ε** | 1.0 → 0.05 decayed over 2,000 episodes |

---

## 🧬 Tests

```bash
pytest tests/ -v
```

- `tests/test_simulator.py` — failure generation, state encoding, action probabilities
- `tests/test_q_learning.py` — Bellman update correctness, ε-decay, UCB counts
- `tests/test_reward.py` — edge cases: security-skip penalty, escalation, clamping

---

## ⚖️ Ethical Considerations

- 🛑 Agent **never** autonomously recovers in HIGH-risk states — escalation is mandatory.
- 🔒 `skip_stage` on `security_scan` incurs a −5 penalty and sets `safe_for_autonomous_recovery=False`.
- 🧪 Q-tables are trained on **synthetic data** — retrain before deploying on real pipelines.
- ⚠️ Reward function penalises escalation to discourage over-escalation in production.

---

## 📂 File Structure

```
CI_Healing_Agent/
├── config.py                    # hyperparameters & constants
├── pipeline_simulator.py        # stochastic failure generator
├── reward_function.py           # scalar reward computation
├── q_learning_agent.py          # ε-greedy Q-Learning
├── ucb_agent.py                 # UCB1 Q-Learning
├── autogen_agents.py            # 4-agent GroupChat system
├── experiment_runner.py         # run all 4 conditions
├── visualizations.py            # 6 plots
├── tools/
│   └── pipeline_inspector.py    # AutoGen custom tool
├── tests/
│   ├── test_simulator.py
│   ├── test_q_learning.py
│   └── test_reward.py
├── results/                     # auto-created output dir
├── requirements.txt
└── CI_Healing_Agent_Colab.ipynb
```

---

## 📜 Citation

If you reference this project in academic work:

```bibtex
@misc{ci_healing_agent_2026,
  title  = {CI Pipeline Self-Healing Agent: Multi-Agent Reinforcement Learning for Autonomous CI/CD Recovery},
  author = {Patel, Niraj},
  year   = {2026},
  note   = {INFO 7375 Final Project, Northeastern University}
}
```

---

<div align="center">

**Built with 🐍 Python · 🤝 AutoGen · 🦙 Llama 3.1 · 🧠 Reinforcement Learning**

*⭐ If this project helped you, consider starring the repo.*

</div>
