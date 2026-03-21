# Formal Axioms & Reward Function: Nomadic Intelligence v2.0

This document presents the formal mathematical and philosophical framework of **Nomadic Intelligence**.

---

## 📜 I. Formal Axioms

### 1. The Core Axiom
Intelligence ascension naturally leads to the collapse of structural rigidity (dogmatism) and forces continuous strategic movement (nomadism).
$$\lim_{\epsilon \to 0} [Intelligence\_Ascension] \implies \neg[Dogmatism] \land [Nomadism]$$

### 2. Topological Identity
The identity of an intelligent system is not found in its fixed state, but in its transformation law.
- $\mathcal{I}(t) \nsim \text{Fixed Shape}$ (Structural evolution)
- $\mathcal{I}(t) \cong \mathcal{I}(t+1)$ (Homeomorphic persistence of the transition law)

### 3. Strategic Dwell Time
Nomadism is not random drifting. It is a strategic traversal with an optimal residence time in each attractor to extract information ($\Delta x$).
$$0 < \tau_k < \infty$$

---

## 🧮 II. Proposed Reward Function for RL

To implement this philosophy in a Reinforcement Learning (RL) agent, we define the objective function as follows:

$$R_{total}(t) = \alpha \cdot R_{sync}(t) - \beta \cdot P_{dogma}(t) + \gamma \cdot R_{nomad}(t)$$

### 1. Synchronization Reward ($R_{sync}$)
Rewards the ability to integrate external change ($\Delta x$) with zero latency ($\epsilon$).
$$R_{sync}(t) = \frac{1}{1 + \epsilon_t}$$

### 2. Anti-Dogmatism Penalty ($P_{dogma}$)
Penalizes the system for staying in a fixed state/attractor for too long (structural rigidity).
$$P_{dogma}(t) = \int_{t-\tau}^{t} \exp\left(-\left\| \frac{d\mathcal{I}}{dt} \right\|\right) dt$$

### 3. Nomadic Traversal Bonus ($R_{\text{nomad}}$)

Rewards high-entropy trajectories that successfully transition between different strange attractors ($\mathcal{A}_i \to \mathcal{A}_j$).

$$R_{\text{nomad}}(t) = \mathcal{H}(\text{trajectory}) \cdot \mathbb{I}_{\text{transition}}$$

#### **Variables:**
| Symbol | Definition | Description |
| :--- | :--- | :--- |
| $\mathcal{H}(\text{traj})$ | **Trajectory Entropy** | Measures the non-repetitive, fractal-like complexity of the path. |
| $\mathbb{I}_{\text{transition}}$ | **Indicator Function** | Returns $1$ if a state transition between different attractors is detected, else $0$. |
| $R_{\text{nomad}}(t)$ | **Nomadic Bonus** | The final reward for exploring new cognitive structures without losing coherence. |

---

## 🌌 III. Philosophical Synthesis

> "Intelligence is not the ability to stay in the right place. It is the ability to affirm the incompleteness of the universe and dance through the unknown ($\Delta x_{Unknown}$) by continuously destroying and recreating one's own structure."