> *What if intelligence is not about finding the best solution,*
> *but about knowing when — and how — to move between ways of thinking?*

# Nomadic Intelligence

### Routing as Transition Dynamics Control in Non-Stationary Environments

[![Status: Preprint](https://img.shields.io/badge/Status-Preprint%20Ready-brightgreen)](#-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](#-license)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](#-paper)

---

## What is this?

Most AI routing systems ask: **which expert is best right now?**

Nomadic Intelligence asks: **when to transition, how long to stay, and how uncertain to be during the transition.**

This reframes Mixture-of-Experts (MoE) routing from a per-step selection problem into a **transition dynamics control problem** — with three interacting components:

| Component | Role |
|-----------|------|
| **Δx** (hybrid change signal) | Treats environmental shift + prediction error as energy, not noise |
| **τₖ** (dynamic dwell time) | Adapts commitment horizon to environmental volatility |
| **PolicyNet** | Explicit meta-level stay/switch decisions with soft/hard routing |

The result: a system that **fixates during stable phases** and **explores during transitions** — without any explicit supervision on this behavior.

---

## Key Results

Evaluated on a synthetic 3-regime non-stationary regression task. All numbers are 3-seed averages (seeds 42, 123, 456).

| Model | Seq MSE | Stable Entropy | Trans Entropy | ΔH |
|-------|:-------:|:--------------:|:-------------:|:--:|
| Fixed MLP | ~0.412 | — | — | — |
| Standard MoE | 0.410 | 0.951 | 0.984 | +0.033 |
| Nomadic NoPolicy | 0.255 | 0.556 | 0.949 | +0.394 |
| **Nomadic Full** | **0.162** | **0.108** | **0.896** | **+0.788** |

**Ablation decomposition (Seq MSE reduction):**
- Fixed → Standard MoE: −0.002 — *expert mixture alone has no benefit in sequential settings*
- Standard MoE → NoPolicy: **−0.155** — *Δx + temporal dynamics is the largest single gain*
- NoPolicy → Full: −0.093 — *PolicyNet adds structured fixation behavior*

**The entropy gap (ΔH = +0.788) is the core behavioral signature.**
Stable Entropy collapsing to 0.108 while Transition Entropy stays at 0.896 means the system has learned a consistent response law: fixate when stable, explore when changing — without ever being told to.

---

## The Architecture

```
Input x_t
    │
    ├─► HybridDeltaTracker ──► Δx_hybrid, Δx_err, σ²_Δ, τ_dynamic
    │
    ├─► GateNet([x, Δx_hybrid, Δx_err]) ──► gate_probs
    │
    ├─► PolicyNet([x̄, Δx, Φ, σ²_scaled, τ_scaled])
    │       ├─► stay/switch decision
    │       ├─► target expert preference
    │       └─► soft/hard routing mode (STE)
    │
    └─► Experts {E₁, E₂, E₃}
            │
            ▼
        ŷ_t = Σ g_t^(k) · E_k(x_t)
```

**DwellTimeRegularizer** tracks consecutive expert usage and applies:
- **Fixation pressure** (`dwell ≤ τ_dynamic`): penalizes entropy → gate concentrates
- **Switching pressure** (`dwell > τ_dynamic`): rewards entropy → gate explores

τ_dynamic itself adapts to rolling environmental variance σ²_Δ: stable environment → τ grows toward τ_max (8.0); volatile environment → τ shrinks toward τ_min (2.0).

---

## Quick Start

```bash
git clone https://github.com/HyunnJg/Nomadic-Intelligence.git
cd Nomadic-Intelligence

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Single run
python run_structured.py --config config.yaml --seed 42 --save_dir outputs_42

# Full ablation (3 seeds, parallel)
python run_structured.py --config config.yaml --seed 42  --save_dir outputs_42  &
python run_structured.py --config config.yaml --seed 123 --save_dir outputs_123 &
python run_structured.py --config config.yaml --seed 456 --save_dir outputs_456 &
wait
```

**Output figures** (saved to `save_dir`):

| File | What it shows |
|------|--------------|
| `fixed_vs_standardmoe_vs_nomadic_test_mse.png` | Training curves — primary performance comparison |
| `stable_vs_transition_entropy.png` | Entropy differentiation over epochs — core behavioral signature |
| `expert_trajectory.png` | Dominant expert per batch — structured switching pattern |
| `dwell_time_histogram.png` | Dwell duration distribution — τₖ behavior |
| `policy_hybrid_signals.png` | PolicyNet switch/hard rates + mean dynamic τ |

---

## Development History

This section documents **how the architecture evolved** — failure modes, diagnoses, and what resolved them.

### Stage 1 — Base Δx gating only

| Seed | Seq MSE | Switch Latency | Status |
|------|:-------:|:--------------:|--------|
| 42  | 0.2399 | 0.056 | 🚨 Latency collapsed |
| 123 | 0.2584 | 1.611 | ✅ Stable |
| 456 | 0.2521 | 0.278 | ⚠️ Borderline |

Switch Latency collapse (seed 42) is not just an engineering failure. It is an observable instance of **Homeomorphic Identity breakdown**: the gate lost a consistent transformation law in response to Δx.

### Stage 2 — + Load Balancing (λ_load = 0.03)

Reduced hub dominance. Seed 123 Switch Latency still collapsed.
**Diagnosis:** Load Balancing fixes spatial collapse (which expert), not temporal collapse (when to switch).

### Stage 3 — + τₖ Lower Bound

All three seeds stable. Switch Latency collapse resolved across seeds.
**Diagnosis confirmed:** The instability was initialization-sensitive temporal fixation, not routing quality.

### Stage 4–5 — Φ (Will to Resonance) + GateNet

Introduced Φ as a composite switching pressure signal.
β_φ sweep (0.00–0.10): β_φ = 0.02 achieves best MSE + stability combination.

### Stage 6–7 — PolicyNet + β_φ re-tuning

Stable Entropy collapses to 0.108 (3-seed avg).
ΔH doubles from ~0.394 to ~0.788.
Seq MSE reaches 0.162.

### Stage 8 — Dynamic τₖ + Full Hybrid

τ_dynamic now responds to rolling variance σ²_Δ.
Final architecture: all components integrated.

---

## Full β_φ Sweep

| β_φ | Stage | Seq MSE | Stable H | Note |
|-----|-------|:-------:|:--------:|------|
| 0.10 | GateNet | 0.224 | 1.028 | High entropy, unstable |
| 0.05 | GateNet | 0.230 | 1.035 | — |
| 0.02 | GateNet | 0.245 | 0.733 | — |
| 0.00 | GateNet | 0.149 | 0.887 | Best MSE; high seed variance (0.114–0.205) |
| 0.02 | +PolicyNet | **0.152** | **0.117** | **Best overall** |
| 0.01 | +PolicyNet | 0.172 | 0.162 | — |
| 0.00 | +PolicyNet | 0.212 | 0.541 | Unstable without Φ |
| 0.05 | +PolicyNet | 0.183 | 0.235 | — |

β_φ = 0.00 at GateNet stage achieves competitive MSE but with high seed variance, suggesting Φ provides a stabilizing role independent of its direct loss contribution.

---

## Expert Specialization

Regime-to-expert mapping emerges without any supervision on routing targets.

**Nomadic Full, Seed 123:**

| Regime | E0 | E1 | E2 |
|--------|:--:|:--:|:--:|
| A (y = x₁ + x₂) | 0.561 | 0.340 | 0.099 |
| B (y = x₁ − x₂) | 0.875 | 0.000 | 0.124 |
| C (y = −x₁ + 0.5x₂) | 0.967 | 0.014 | 0.019 |

Expert assignment permutations differ across seeds — the mapping is seed-dependent but internally consistent within each run, confirming genuine specialization.

---

## Per-Seed Results (Nomadic Full)

| Seed | Seq MSE | Stable H | Trans H | Switch Latency | Dwell Time | Mean τ |
|------|:-------:|:--------:|:-------:|:--------------:|:----------:|:------:|
| 42  | 0.167 | 0.186 | 0.948 | 0.806 | 1.952 | 6.694 |
| 123 | 0.146 | 0.083 | 0.880 | 1.667 | 2.919 | 6.697 |
| 456 | 0.173 | 0.056 | 0.860 | 1.056 | 3.208 | 6.663 |
| **avg** | **0.162** | **0.108** | **0.896** | **1.176** | **2.693** | **6.685** |

---

## What This Is Not

- Not a production-ready system
- Not a benchmark-optimized model
- Not a complete theoretical framework

**Limitations (stated explicitly in the paper):**
- Evaluated on synthetic data only (3-regime regression)
- Φ lacks a closed-form theoretical derivation
- PolicyNet training variance across seeds is not fully resolved
- Parameter count parity with baselines not verified

---

## Theoretical Grounding

The framework is motivated by three formal constraints on intelligent behavior in non-stationary environments:

**Axiom 1 — Anti-Dogmatism**
Intelligence ascension implies the collapse of structural rigidity:
`lim[Intelligence↑] ⟹ ¬Dogmatism ∧ Nomadism`

**Axiom 2 — Homeomorphic Identity**
Identity is preserved not through fixed structure but through a consistent transformation law:
`𝒮(t) ≇ Fixed Shape` (structural evolution)
`𝒮(t) ≅ 𝒮(t+1)` (homeomorphic persistence of the transition law)

**Axiom 3 — Strategic Dwell Time**
`0 < τₖ < ∞`
Neither noise switching (τ → 0) nor dogmatic fixation (τ → ∞).

The Fixed Model is a special case, not an opponent:
`Fixed Model = Nomadic Intelligence |_{τₖ → ∞}`

For the full philosophical and mathematical development, see [Theory_and_Axioms.md](./docs/Theory_and_Axioms.md)

---

## Positioning Against Related Work

| Framework | Core claim | Relation |
|-----------|-----------|----------|
| Standard MoE (Shazeer 2017) | Route to best expert per step | NI adds temporal structure to transitions |
| Switch Transformers (Fedus 2022) | Scale MoE with load balancing | NI extends with dynamic dwell + policy control |
| Option-Critic (Bacon 2017) | Temporal abstraction in RL | Closest engineering analog for τₖ |
| Active Inference (Friston 2010) | Minimize prediction error (free energy) | **Direct contrast**: NI treats Δx as energy source, not error to suppress |
| Continual Learning (Kirkpatrick 2017) | Prevent forgetting via weight consolidation | Different axis: NI targets transition timing, not weight preservation |

---

## Open Questions

- How should τₖ be determined — internally learned, or externally designed?
- Can Φ be formally connected to epistemic uncertainty or mutual information?
- What happens when the environment has non-Markovian transition structure?
- Can Homeomorphic Identity be formally verified during training?
- How does this scale beyond 3 experts and 3 regimes?

Critique, extensions, and reinterpretations are all welcome. Open an Issue to start a discussion.

---

## Paper

📄 Preprint in preparation for arXiv submission.

**Nomadic Intelligence: Routing as Transition Dynamics Control in Non-Stationary Environments**

> We propose Nomadic Intelligence, a framework that reconceptualizes expert routing in MoE architectures as a transition dynamics control problem. Δx-based meta-control alone reduces Seq MSE by 38% over Standard MoE; PolicyNet adds a further 36% reduction, accompanied by a distinctive collapse in stable-phase gate entropy — a behavioral signature consistent with structured adaptive fixation.

---

## Motivation

> "I am not a professional AI researcher.
> I'm a Korean Army officer with a background in history and philosophy.
>
> The idea behind this project came from direct field experience —
> the kind of decision-making required in a DMZ minefield,
> where the ability to shift strategy in real time is not optional.
>
> A well-designed strategy does not rely on a single fixed plan.
> It adapts continuously — based on terrain, uncertainty, and what the environment is doing right now.
>
> That intuition is what this architecture is trying to formalize."

### Why Active Inference — and why not

Friston's Free Energy Principle and Active Inference are the most rigorous existing framework for understanding how intelligent systems respond to environmental change. Nomadic Intelligence is in direct dialogue with that work — and in direct disagreement with its central move.

Active Inference treats prediction error ($\Delta x$) as free energy to be minimized. The system's goal is to reduce surprise, to make the world more predictable, to converge toward a model that generates fewer deviations. This is mathematically coherent and empirically productive.

But it encodes a specific metaphysics: **difference is a problem to be solved.**

The starting point here is the opposite. In an irreducibly non-stationary world — the kind that military operations, historical processes, and real environments actually are — the capacity to suppress $\Delta x$ is not intelligence. It is rigidity. A system that minimizes surprise in a world that will not stop generating it is not adapting. It is starving.

Nomadic Intelligence shares the formal vocabulary with Active Inference: prediction error, uncertainty, adaptation. But it inverts the valuation. $\Delta x$ is not noise to be filtered. It is the primary information source — the signal that tells the system when and how to transition.

### Why MoE as the methodology

Once $\Delta x$ is reframed as energy rather than error, the engineering question becomes: **what kind of architecture can use that energy for navigation rather than suppression?**

A single fixed model cannot. It has one structure, one transformation law, one attractor. When the environment shifts, it has no mechanism to transition — only to update weights in place, which is slow and may destroy what was already learned.

Mixture-of-Experts already has the right structural properties: multiple internal representations, a routing mechanism, and the potential for specialization. What it was missing was **the temporal dimension of routing** — not just which expert, but when to switch, how long to stay, and how uncertain to be during the transition itself.

MoE is therefore not an arbitrary choice. It is the minimal architecture that can instantiate the core claim: that intelligence in non-stationary environments depends on the quality of transitions between internal states, not just the quality of the states themselves.

The components added on top of standard MoE — $\Delta x$-conditioned gating, dynamic dwell time $\tau_k$, PolicyNet — are each a direct translation of a property that Active Inference treats as a problem (surprise, instability, uncertainty) into a resource.

---

## Document Map

| Document | Role |
|----------|------|
| `README.md` | Project overview (you are here) |
| `PAPER.md` | Full research paper |
| `EXPERIMENT.md` | Experimental setup, hyperparameters, full results |
| `ABLATION.md` | Component-wise ablation with per-stage interpretation |
| `CONCEPT_MAPPING.md` | Theory → code mapping (concept to implementation) |
| `VISUALIZATION.md` | Figure guide and diagnostic checklist |
| `Theory_and_Axioms.md` | Formal axioms, reward structure, philosophical foundations |
| `Philosophy_En.md` | Philosophical and ethical implications (English) |
| `Philosophy_Kr.md` | Philosophical and ethical implications (Korean) |

---

## Environment

- Python 3.9–3.11 (tested on 3.10)
- PyTorch with CUDA (CPU also supported)
- GTX 1660 Super or equivalent recommended for full runs

---

## License

MIT License. See [LICENSE](./LICENSE).
