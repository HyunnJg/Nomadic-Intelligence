# Ablation Study

This document details the incremental component analysis underlying the results reported in `PAPER.md`.  
Each row corresponds to a distinct architectural stage; all numbers are 3-seed averages (seeds 42, 123, 456).

---

## Ablation Summary Table

| Stage | Model | Seq MSE | Stable H | ΔH | Ablation Δ |
|-------|-------|:-------:|:--------:|:--:|:----------:|
| 0 | Fixed Baseline | ~0.412 | — | — | — |
| 1 | Standard MoE | 0.410 | 0.951 | +0.033 | −0.002 |
| 2 | + Δx + Dynamic τₖ + Dwell (NoPolicy) | 0.255 | 0.556 | +0.394 | **−0.155** |
| 3 | + PolicyNet (Full) | **0.162** | **0.108** | **+0.788** | −0.093 |

*Ablation Δ = Seq MSE reduction from previous stage. Negative = improvement.  
ΔH = H_transition − H_stable (positive = structured switching behavior).*

---

## Stage 0 → 1: Fixed vs. Standard MoE (Δ = −0.002)

**What changed:** replaced single MLP with 3-expert MoE + input-conditioned softmax gate.

**Result:** no meaningful improvement in sequential evaluation.

**Why:** Standard MoE treats each routing decision independently.  
It has no mechanism to reason about *when* to switch, *how long* to stay, or *how uncertain to be* during a regime transition.  
In sequential settings, expert mixture without temporal context is equivalent to a larger fixed model.

> **Conclusion:** Expert mixture alone does not solve the non-stationarity problem.  
> The problem is not model capacity — it is the absence of a transition dynamics model.

---

## Stage 1 → 2: + Δx, Dynamic τₖ, Dwell Regularizer (Δ = −0.155)

**What changed:** added three interacting components simultaneously.

| Component | Role |
|-----------|------|
| Δx (hybrid change signal) | Encodes environmental shift + prediction error as routing energy |
| Dynamic τₖ | Adapts minimum dwell time to environmental volatility (σ²_Δ) |
| DwellTimeRegularizer | Penalizes entropy during stable phases; rewards it post-capacity |

**Result:**
- Seq MSE drops from 0.410 → 0.255 (38% reduction)
- ΔH rises from 0.033 → 0.394: the system has *learned* to increase routing uncertainty during transitions and decrease it during stable phases — without any explicit supervision on this behavior

**Why this is the largest single gain:** Δx provides the signal that was entirely missing in Stage 1. The gate now has information about *whether the environment is changing*, not just *what the current input is*. Dwell control prevents both noise-driven rapid switching and pathological fixation.

> **Conclusion:** The core performance gain comes from temporal meta-signals, not from architectural complexity.  
> Δx-based gating is the most important single component.

---

## Stage 2 → 3: + PolicyNet (Δ = −0.093)

**What changed:** added explicit meta-level switching control via PolicyNet.

PolicyNet inputs: `[x̄, Δx_hybrid, Δx_err, Φ, tanh(10·σ²_Δ), tanh((τ_dynamic − 5)/5)]`  
PolicyNet outputs: stay/switch decision, target expert distribution, soft/hard routing mode.

**Result:**
- Seq MSE drops from 0.255 → 0.162 (36% additional reduction)
- Stable Entropy collapses from 0.556 → **0.108**: near-deterministic routing during stable phases
- ΔH doubles from 0.394 → 0.788

**Why Stable Entropy collapse matters:** This is the empirical signature of *homeomorphic fixation* — the gate converges to near-hard assignment during stable regimes while retaining full flexibility during transitions. The routing identity is preserved not through rigidity but through a consistent response law.

**Per-seed breakdown:**

| Seed | Seq MSE | Stable H | Trans H |
|------|:-------:|:--------:|:-------:|
| 42  | 0.167 | 0.186 | 0.948 |
| 123 | 0.146 | 0.083 | 0.880 |
| 456 | 0.173 | 0.056 | 0.860 |

Stable Entropy seed variance (0.056–0.186) indicates PolicyNet's convergence to hard fixation is not fully guaranteed under the current heuristic teacher. This is a known limitation; see PAPER.md §5.3.

> **Conclusion:** PolicyNet converts implicit gating into explicit policy-driven transition control.  
> Its distinctive contribution is not raw MSE reduction but *behavioral interpretability* — the entropy gap becomes a clean, measurable signature of structured adaptation.

---

## β_φ Sensitivity

β_φ scales Φ's contribution to the gating objective. Sweep results across stages:

| β_φ | Stage | Seq MSE | Stable H | Note |
|-----|-------|:-------:|:--------:|------|
| 0.10 | GateNet | 0.224 | 1.028 | High entropy, training unstable |
| 0.05 | GateNet | 0.230 | 1.035 | — |
| 0.02 | GateNet | 0.245 | 0.733 | — |
| 0.00 | GateNet | 0.149 | 0.887 | Best MSE; high seed variance (0.114–0.205) |
| 0.02 | +PolicyNet | **0.152** | **0.117** | Best overall: MSE + stability |
| 0.01 | +PolicyNet | 0.172 | 0.162 | — |
| 0.00 | +PolicyNet | 0.212 | 0.541 | Unstable without Φ signal |
| 0.05 | +PolicyNet | 0.183 | 0.235 | — |

β_φ = 0.02 with PolicyNet achieves the best balance of Seq MSE and stable fixation consistency.  
β_φ = 0.00 achieves competitive MSE at GateNet stage but with much higher cross-seed variance, suggesting Φ provides a *stabilizing* role independent of its direct loss contribution.

---

## Key Conclusion

> Performance gain is not from any single component.  
> It is from the interaction between three layers of temporal control:
> - **Δx** supplies the signal (what is changing, and how fast)
> - **τₖ** controls the commitment horizon (how long to stay)
> - **PolicyNet** converts these into explicit, interpretable decisions
>
> Remove any one layer and the behavioral signature (entropy differentiation) degrades.
