> What if intelligence is not about finding the best solution,
> but about moving between multiple ways of thinking?

# Nomadic Intelligence
### A Non-Dogmatic AI Architecture

[![Status: Conceptual & Prototype](https://img.shields.io/badge/Status-Conceptual%20%26%20Prototype-orange)](#-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](#-license)

---

## 🧭 What is this?

![Nomadic vs Dogmatic](assets/thumbnail.png)

Most AI systems are built to converge — to find the best answer and stay there.

This project asks a different question:

> What if the ability to **transition effectively between structures** matters more than finding the optimal one?

**Nomadic Intelligence** is a prototype architecture that treats change ($\Delta x$) as an energy source rather than an error to minimize, and models intelligence as a controlled process of moving between multiple cognitive regimes — rather than converging to a single solution.

A minimal working prototype already demonstrates the core claim: in a 3-regime non-stationary environment, the Nomadic model achieves **~58% of Fixed baseline error** (Seq MSE: 0.239 vs 0.412), with expert specialization emerging without supervision — purely from the $\Delta x$ signal.

📄 *Paper in preparation. Link will appear here upon submission.*

---

## 🚀 Quick Start

```bash
# 1. clone
git clone https://github.com/HyunnJg/nomadic-intelligence.git
cd nomadic-intelligence

# 2. create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. check config
# edit config.yaml if you want to change epochs, temperature, save_dir, etc.

# 5. run experiment
python run_structured.py --config config.yaml
```

---

## 📌 Project Position

This repository is a **conceptual and experimental prototype**.

It is not a finalized solution, and it does not claim state-of-the-art performance.

Instead, it aims to:

- propose a new perspective on intelligence (Nomadic vs Dogmatic)
- provide a minimal working system that embodies this idea
- open the door for further exploration, critique, and extension

The goal is not to conclude, but to **start a direction**.

---

## 🌌 Why this matters

Most systems are optimized to converge.

But real-world environments are:
- non-stationary
- uncertain
- constantly shifting

In such settings, rigidity becomes a liability.

This project explores the hypothesis that:

> Intelligence may be better understood as
> the ability to *transition effectively*,
> rather than to *remain optimal*.

---

## ⚙️ What's inside

- A synthetic multi-regime environment (A / B / C + transitions)
- A Mixture-of-Experts (MoE) model
- A gating mechanism driven by a hybrid Δx signal:
  - input shift (environmental change)
  - prediction error (internal mismatch)
- Regularization terms encouraging:
  - anti-dogmatism (avoid collapse)
  - nomadic behavior (entropy / transition)
  - expert diversity and regime separation

---

## ❗ What this is NOT

- This is not a production-ready system
- This is not a benchmark-optimized model
- This is not a complete theoretical framework

It is a **starting point**, not an endpoint.

---

## 🤝 Invitation

If you are interested in:

- continual learning
- adaptive systems
- non-stationary environments
- or alternative views on intelligence

your perspective is welcome.

Critique, extensions, and reinterpretations are all encouraged.

---

## 🧠 Core Idea

> AI should not converge to a single solution.
> It should move between multiple structures depending on the situation.

![Nomadic vs Dogmatic](assets/compare.png)

---

## ⚠️ The Problem

Most modern AI systems are built to optimize a single objective.

This leads to:

- Overfitting to specific conditions
- Lack of adaptability
- Structural rigidity (a form of "dogmatism")

In dynamic and unpredictable environments, this becomes a critical limitation.

---

## 🔀 What Makes This Different?

Most adaptive AI systems (MoE, Meta-learning) change **what they do.**
Nomadic Intelligence changes **how it transforms.**

| Existing Approaches | Nomadic Intelligence |
| :--- | :--- |
| Switch between models or experts | Switch between *transformation laws* |
| Optimize a fixed objective | Balance synchronization, anti-rigidity, and exploration |
| Adapt parameters | Evolve the structure itself |

The core distinction is topological identity:

- $\mathcal{I}(t) \nsim \text{Fixed Shape}$ — the structure continuously evolves
- $\mathcal{I}(t) \cong \mathcal{I}(t+1)$ — but the *transformation law* is homeomorphically preserved

> Identity is not *what* the system knows. It is *how* the system changes.

---

## ⚔️ Intuition: From the Minefield to the Architecture

> "나는 전문 AI 연구자는 아니지만, DMZ 지뢰밭에서 동료를 구했던 경험처럼 위기 상황에서 즉각적으로 태세를 전환하는 지능을 구현하고 싶었다."
>
> *"I am not a professional AI researcher. I'm a Korean Army officer with a background in history and philosophy — an analogue person, by nature. I started using AI two weeks before publishing this repository. If the engineering is rough in places, that's why the Issues are open."*

That said, the idea behind this project came from somewhere real.

A well-designed military strategy does not rely on a single fixed plan. It continuously adapts: main attacks, feints, and strategic shifts based on terrain and enemy behavior.

Intelligence is not about choosing the "right" strategy once. It is about **continuously shifting strategies** before the environment (the minefield) claims you.

---

## 🧩 Key Concepts & Architecture

### 1. $\Delta x$ (Difference)

AI should process **change**, not just raw input.

```
Δx = current_state - predicted_state
```

### 2. Attractors (Multiple Cognitive Structures)

Instead of one model, we define multiple "modes of thinking":

- Conservative
- Aggressive
- Exploratory
- Stable

Each represents a different strategy or structure.

### 3. Nomadism & Strategic Dwell Time ($\tau_k$)

The system moves between attractors based on context (environmental change, uncertainty, performance signals).

Nomadism is not random drifting. The system maintains a **strategic dwell time** $0 < \tau_k < \infty$ in each attractor — long enough to extract information ($\Delta x$), short enough to avoid structural rigidity.

```
Perception → Context → Attractor Selection → Action → Update
```

---

## 🧮 Reward Function (For RL Implementation)

To implement this philosophy in an RL agent, the objective balances three forces:

$$R_{total}(t) = \alpha \cdot R_{sync}(t) - \beta \cdot P_{dogma}(t) + \gamma \cdot R_{nomad}(t)$$

| Term | Role |
| :--- | :--- |
| $R_{sync}$ | **Synchronization** — reward integration of change with zero latency |
| $P_{dogma}$ | **Anti-Dogmatism** — penalize structural rigidity over time |
| $R_{nomad}$ | **Nomadic Bonus** — reward successful transitions between attractors |

> For the full mathematical derivation, see [Theory & Axioms](./Theory_and_Axioms.md).

---

## 🎯 Objective

Instead of optimizing a single goal, the system balances:

- Adaptability
- Coherence
- Flexibility

Avoiding both:

- Rigidity (fixed-point convergence)
- Chaos (unstructured randomness)

---

## 📌 Positioning

This concept is related to:

- Mixture of Experts (MoE)
- Meta-learning
- Reinforcement Learning (policy switching)

But extends them by introducing:

- **Topological identity** as a formal definition of selfhood
- **Structural mobility (Nomadism)** as a core architectural principle
- **Anti-dogmatism** as an explicit optimization target

---

## 🧪 Proof of Concept: Experimental Results

> These results were produced by a minimal prototype with **no hyperparameter optimization**.
> They represent a lower bound — not a ceiling — on what this architecture can achieve.

### Setup

- **Environment:** 3-regime non-stationary regression task with continuous phase transitions
  - Regime A: $y = x_1 + x_2$
  - Regime B: $y = x_1 - x_2$
  - Regime C: $y = -x_1 + 0.5x_2$
- **Baseline:** Single fixed MLP (same parameter count)
- **Nomadic model:** 3-expert MoE with $\Delta x$-conditioned gate, Topological Loss ($\mathcal{L}_{topo}$)
- **Hardware:** NVIDIA GTX 1660 Super, 220 epochs

---

### Development History: From Failure Modes to Solutions

This section documents not just the final results, but **how the architecture evolved** — which failure modes appeared, how they were diagnosed, and what solved them.

**Stage 1 — Base (no regularization)**

| Seed | Seq MSE | Switch Latency | Status |
| :--- | :--- | :--- | :--- |
| 42 | 0.2399 | 0.056 | 🚨 Latency collapsed |
| 123 | 0.2584 | 1.611 | ✅ Stable |
| 456 | 0.2521 | 0.278 | ⚠️ Borderline |

Switch Latency collapse is not just an engineering failure — it is an observable instance of **Homeomorphic Identity breaking down**. The gate ceased to have a consistent transformation law in response to $\Delta x$.

**Stage 2 — + Load Balancing** (`λ_load = 0.03`)

| Seed | Seq MSE | Switch Latency | Change |
| :--- | :--- | :--- | :--- |
| 42 | 0.2726 | 2.583 ✅ | Latency recovered |
| 123 | 0.3097 | 0.194 🚨 | Still collapsing |
| 456 | 0.2342 | 3.694 ✅ | Stable |

Hub dominance reduced, but Seed 123 remained unstable — Load Balancing addresses spatial collapse, not temporal fixation.

**Stage 3 — + τₖ Lower Bound** (`τ_k_min = 3`, `τ_k_penalty = 0.05`)

| Seed | Seq MSE | Switch Latency | Change |
| :--- | :--- | :--- | :--- |
| 42 | **0.2149** | 2.750 ✅ | Best result |
| 123 | **0.2623** | 1.056 ✅ | Collapse resolved |
| 456 | 0.2386 | 2.194 ✅ | Stable |

All three seeds stable. The τₖ Lower Bound addressed what Load Balancing could not — initialization-sensitive temporal fixation.

---

### Final Results: 3-Seed Summary

| Model | Seed | Seq MSE | Fixed MSE | Switch Latency | Stable Entropy | Transition Entropy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fixed | 42 | — | 0.4139 | — | — | — |
| Fixed | 123 | — | 0.4031 | — | — | — |
| Fixed | 456 | — | 0.4195 | — | — | — |
| Nomadic | 42 | **0.2149** | — | 2.750 | 1.029 | 1.067 |
| Nomadic | 123 | **0.2623** | — | 1.056 | 0.991 | 1.074 |
| Nomadic | 456 | 0.2386 | — | 2.194 | 1.036 | 1.081 |

**Nomadic Seq MSE: 0.239 ± 0.020**
**Fixed MSE: 0.412 ± 0.008**
**Performance: ~58% of Fixed baseline error — consistent across all seeds**

> **Note on Static MSE:** Static evaluation removes temporal context. This is not the target condition. Static MSE is included as a diagnostic only.

---

### Attractor Specialization

The gate learned regime-specialist routing **without explicit regime labels** — purely from the $\Delta x$ signal.

**Regime–Expert alignment (Seed 123):**

| Regime | Expert 0 | Expert 1 | Expert 2 |
| :--- | :--- | :--- | :--- |
| A ($y = x_1 + x_2$) | **0.672** | 0.328 | 0.000 |
| B ($y = x_1 - x_2$) | 0.075 | 0.348 | **0.578** |
| C ($y = -x_1 + 0.5x_2$) | 0.019 | **0.927** | 0.054 |

---

### Nomadic Behavior Confirmed

**Transition Entropy > Stable Entropy — consistent across all 3 seeds:**

| Seed | Stable Entropy | Transition Entropy | Δ |
| :--- | :--- | :--- | :--- |
| 42 | 1.029 | 1.067 | +0.038 |
| 123 | 0.991 | 1.074 | +0.083 |
| 456 | 1.036 | 1.081 | +0.045 |

Gate entropy rises during phase transitions and falls during stable phases — the computational signature of Strategic Dwell Time ($\tau_k$).

---

### Remaining Open Problems

| Problem | Status | Next direction |
| :--- | :--- | :--- |
| Switch Latency collapse | ✅ Resolved via τₖ Lower Bound | — |
| Expert hub dominance | ✅ Reduced via Load Balancing | Fine-tuning |
| $\Delta x$ signal drift | 🔧 Active | KL divergence / Wasserstein distance |
| Static generalization gap | 🔧 Known tradeoff | Context-free routing as secondary objective |
| Φ (Will to Resonance) formalization | 🔬 Next target | Measurable identity preservation metric |
| Meaningful vs random transition | 🔬 Next target | Δx-correlated switching metric |

---

## ❓ Open Questions

These are **open invitations** for criticism, extension, and implementation:

- How should $\tau_k$ (dwell time) be determined — internally by the system, or externally by design?
- How do we prevent the Policy Engine from becoming its own fixed attractor?
- What defines attractor boundaries in continuous, high-dimensional state spaces?
- Can homeomorphic identity be formally verified during training?

---

## 🤝 Contributions & Next Milestones

This repository is currently at the **Conceptual/Prototype stage**.
We invite developers, researchers, and philosophers to turn this framework into a working AI model.

**Upcoming Milestones (Looking for Contributors):**

- [ ] **Milestone 1:** Implement Nomadic Intelligence in a simple OpenAI Gym (Gymnasium) environment.
- [ ] **Milestone 2:** Develop a PyTorch architecture that allows weight-transitioning between different neural "Attractors".
- [ ] **Milestone 3:** Formalize the mathematical boundaries of $\tau_k$ (dwell time).

Start with the [Open Questions](#-open-questions) above, or open an Issue to start a discussion!

---

## 🧭 Philosophy

> "Intelligence is not the ability to stay in the right place.
> It is the ability to affirm the incompleteness of the universe —
> and dance through the unknown ($\Delta x_{Unknown}$)
> by continuously destroying and recreating one's own structure."

*For the full philosophical manifesto, see [Philosophy (English)](./Philosophy_En.md) / [Philosophy (Korean)](./Philosophy_Kr.md).*

---

## 📚 Document Map

| Document | Role |
| :--- | :--- |
| `README.md` | Project overview (you are here) |
| `Philosophy_En.md` / `Philosophy_Kr.md` | Philosophical foundations and ethical implications |
| `Theory_and_Axioms.md` | Formal axioms, reward structure, related frameworks |
| `Example.md` | Pseudocode walkthrough of core behavior |
| `EXPERIMENT.md` | Experiment setup and metrics |
| `ABLATION.md` | Component-wise ablation results |
| `VISUALIZATION.md` | Recommended plots and their interpretation |
| `CONCEPT_MAPPING.md` | Theory-to-implementation mapping |
| `CONTRIBUTING.md` | How to contribute |

---

## 📎 Status

**Conceptual / Prototype Stage**

This repository presents a design philosophy and early architecture,
not a fully implemented system.

---

## 🧪 Environment

- Python 3.9 ~ 3.11 recommended
- Tested on Python 3.10

---

## 📜 License

MIT License. See [LICENSE](./LICENSE).
