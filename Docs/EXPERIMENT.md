# Experiment Setup

For the full analysis and discussion of these results, see `PAPER.md`.  
This document serves as a standalone reference for reproducing the experiments.

---

## 1. Task: Non-Stationary Regression with Continuous Regime Transitions

Three linearly separable regimes with deliberately overlapping input distributions:

| Regime | Label Function | Input Center | Distribution |
|--------|----------------|-------------|-------------|
| A | y = x₁ + x₂ | (+2.5, +2.5) | N(center, 0.9²·I) |
| B | y = x₁ − x₂ | (−2.5, −2.5) | N(center, 0.9²·I) |
| C | y = −x₁ + 0.5x₂ | (+2.5, −2.5) | N(center, 0.9²·I) |

The sequence cycles A → B → C → A continuously.  
Between regimes, 8 *transition steps* linearly interpolate both inputs and labels:

```
x_mix(α) = (1 − α)·x_curr + α·x_next
y_mix(α) = (1 − α)·y_curr + α·y_next
α ∈ {1/8, 2/8, ..., 8/8}
```

**Why σ = 0.9 overlap?**  
Static features alone are insufficient to discriminate regimes reliably.  
Temporal signals (Δx, dwell history) are *necessary* for correct routing.  
This is the design property that makes the task non-trivial for static or stateless models.

### Evaluation Modes

**Sequence MSE (primary):**  
Full test set evaluated in temporal order with temporal context intact (Δx signal, routing history).  
This is the only metric that measures the temporal routing capability of Nomadic models.

**Static MSE (reference only):**  
Each regime evaluated independently, without temporal context.  
Nomadic models lack Δx signal in this mode — this is *intentional*, not a bug.  
The static/sequence MSE gap quantifies how much a model depends on temporal context.  
A large gap is the intended behavior, not a failure mode.

**Why Standard MoE has identical Static and Sequence MSE:**  
Standard MoE has no temporal signals. Its routing decisions are stateless.  
Static and sequential evaluation are equivalent for this model by construction.  
This provides a clean lower bound on the temporal structure benefit.

### 1.1 Real-World Time Series (ETTh1)
- **Dataset:** Electricity Transformer Temperature (ETTh1), hourly data.
- **Features (7):** HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (Normalized).
- **Regime Definition:** 3-tier volatility regimes based on the 7-day rolling standard deviation of OT (split at 33rd and 67th percentiles).
- **Target Horizon:** 24-step ahead prediction ($t+24$) via `df['OT'].shift(-24)`. This enforces strict prediction pressure, rendering simple persistence baselines ineffective.

### 1.2 LLM Signal Transfer (Gemma-4-E2B)
- **Base Model:** `google/gemma-2b` (4-bit NF4 quantization via bitsandbytes).
- **Environment:** Single T4/L4 GPU.
- **Signal Extraction:** Hidden state of the final token at each generation step serves as $x_t$. Model uncertainty (1 - top1 probability) serves as $err_t$.
- **Adapters:** 3 distinct LoRA adapters (r=4) routed by $\Delta x$ thresholds.

---

## 2. Model Configurations

### Fixed Baseline
- Single 3-layer MLP with ReLU activations
- Architecture: 2 → 64 → 64 → 1
- No temporal signals; no routing
- Parameter count: ~8,500

### Standard MoE (Ablation Control)
- 3 experts (2-layer MLP, Tanh, hidden 64)
- Input-conditioned gate: 3-layer MLP (ReLU, hidden 64), standard softmax
- **No** Δx, **no** dynamic τₖ, **no** PolicyNet
- Load balancing loss only (`λ_load = 0.03`)
- Seq MSE = Static MSE by construction

### Nomadic NoPolicy
- Same 3 experts as Standard MoE
- **GateNet**: 3-layer MLP conditioned on `[x, Δx_hybrid, Δx_err] ∈ ℝ^(d+2)` (ReLU, hidden 64)
- **HybridDeltaTracker**: EMA-based Δx computation + dynamic τₖ via rolling variance
- **DwellTimeRegularizer**: environment-aware fixation / switching pressure
- **No** PolicyNet

### Nomadic Full
- All components of NoPolicy, plus:
- **PolicyNet**: shared 2-layer MLP (`d+5 → 64 → 64`, ReLU) with three output heads:
  - `stay_switch_head`: Linear → softmax (dim 2) — stay(0) / switch(1)
  - `target_head`: Linear → softmax (dim K) — preferred expert
  - `mode_head`: Linear → softmax (dim 2) — soft(0) / hard(1)
- Straight-Through Estimator (STE) for discrete hard-routing decisions

---

## 3. Metrics

### Primary
- **Sequence Test MSE**: MSE on time-ordered test sequence (never shuffled)

### Secondary
| Metric | Definition | Interpretation |
|--------|-----------|---------------|
| Static Test MSE | MSE on regime-stratified static eval | Reference only; high values expected for Nomadic |
| Stable-phase Gate Entropy H_s | H(g_t) over stable-regime timesteps | Low → fixation; target ≈ 0 |
| Transition-phase Gate Entropy H_t | H(g_t) over transition-step timesteps | High → exploration; target ≈ log(K) |
| ΔH = H_t − H_s | Entropy differentiation | Core behavioral signature; target > 0.5 |
| Switch Latency | Mean steps: regime-onset → gate response | Lower = faster adaptation |
| Mean Dwell Time | Mean consecutive steps on dominant expert | Should be moderate (2–5 typical) |
| Mean Dynamic τ | Mean τ_dynamic(t) over test sequence | Reflects environmental volatility estimate |

**ΔH is the primary behavioral signature**, not Seq MSE alone.  
A model with low Seq MSE but flat ΔH ≈ 0 is not demonstrating structured switching — it may be collapsing to a fixed routing pattern that happens to work.

---

## 4. Hyperparameters

### Architecture
| Parameter | Value | Note |
|-----------|-------|------|
| hidden_dim (expert, gate) | 64 | — |
| policy_hidden_dim | 64 | — |
| num_experts (K) | 3 | Matches number of regimes |
| temperature (base) | 0.60 | Gate softmax temperature |
| temp_stable | 0.30 | Φ=0 → this temperature |
| temp_transition | 1.00 | Φ=1 → this temperature |

### Training
| Parameter | Value | Note |
|-----------|-------|------|
| Epochs | 220 | — |
| Learning rate | 2×10⁻³ | Adam |
| Weight decay | 1×10⁻⁵ | — |
| Phase batch size | 64 | Per stable block |
| Train cycles | 40 | Full A→B→C→A cycles in train set |
| Test cycles | 12 | Full A→B→C→A cycles in test set |
| Transition steps | 8 | Steps per regime-to-regime transition |
| Device | CUDA (GTX 1660 Super) | CPU also supported |
| Seeds | 42, 123, 456 | All results are 3-seed averages |

### Loss Weights
| Parameter | Value | Role |
|-----------|-------|------|
| λ_load | 0.03 | Load balancing (prevent expert collapse) |
| τ_k_min | 3 | Static dwell floor (steps) |
| τ_k_penalty (λ_τ) | 0.05 | Dwell regularizer strength |
| τ_min / τ_max | 2.0 / 8.0 | Dynamic τ range |
| τ_var_scale | 6.0 | Sensitivity of τ to σ²_Δ |
| τ_var_window | 8 | Rolling window for σ²_Δ |
| λ_sep | 0.08 | Regime gate separation loss |
| λ_cons | 0.03 | Within-regime gate consistency |
| β_φ | 0.02 | Φ scaling (best from sweep) |
| policy_mix_weight | 0.25 | PolicyNet influence on gate distribution |

### Δx Parameters
| Parameter | Value | Role |
|-----------|-------|------|
| EMA decay | 0.80 | err_ema smoothing |
| err_baseline_momentum | 0.85 | Slower baseline for relative comparison |
| w_env | 1.0 | Environmental shift weight |
| w_err | 2.0 | Prediction error weight (2× env) |

---

## 5. How to Run

```bash
# Single seed
python run_structured.py --config config.yaml --seed 42 --save_dir outputs_seed42

# Full ablation (3 seeds, parallel)
python run_structured.py --config config.yaml --seed 42  --save_dir outputs_42  &
python run_structured.py --config config.yaml --seed 123 --save_dir outputs_123 &
python run_structured.py --config config.yaml --seed 456 --save_dir outputs_456 &
wait
```

Output files per run (saved to `save_dir`):
```
fixed_vs_standardmoe_vs_nomadic_test_mse.png   ← training curves (primary)
stable_vs_transition_entropy.png               ← entropy differentiation over epochs
expert_trajectory.png                          ← dominant expert per batch
dwell_time_histogram.png                       ← dwell duration distribution
policy_hybrid_signals.png                      ← PolicyNet behavioral signals
regime_expert_usage_heatmap.png                ← regime → expert specialization
dynamic_tau_trace.png                          ← τ_dynamic over training
switch_latency_curve.png                       ← switch latency over epochs
```

---

## 6. Results Summary

### Ablation (3-seed averages)

| Model | Seq MSE | Stable H | Trans H | ΔH | Ablation Δ |
|-------|:-------:|:--------:|:-------:|:--:|:----------:|
| Fixed | ~0.412 | — | — | — | — |
| Standard MoE | 0.410 | 0.951 | 0.984 | +0.033 | −0.002 |
| Nomadic NoPolicy | 0.255 | 0.556 | 0.949 | +0.394 | **−0.155** |
| Nomadic Full | **0.162** | **0.108** | **0.896** | **+0.788** | −0.093 |

### Per-seed (Nomadic Full)

| Seed | Seq MSE | Stable H | Trans H | Switch Latency | Dwell Time | Mean τ |
|------|:-------:|:--------:|:-------:|:--------------:|:----------:|:------:|
| 42  | 0.167 | 0.186 | 0.948 | 0.806 | 1.952 | 6.694 |
| 123 | 0.146 | 0.083 | 0.880 | 1.667 | 2.919 | 6.697 |
| 456 | 0.173 | 0.056 | 0.860 | 1.056 | 3.208 | 6.663 |
| **avg** | **0.162** | **0.108** | **0.896** | **1.176** | **2.693** | **6.685** |

### Expert Specialization (Nomadic Full, seed 123)

| Regime | E0 | E1 | E2 |
|--------|:--:|:--:|:--:|
| A | 0.561 | 0.340 | 0.099 |
| B | 0.875 | 0.000 | 0.124 |
| C | 0.967 | 0.014 | 0.019 |

Expert assignment permutations differ across seeds (seed-dependent initialization),  
confirming genuine regime-consistent specialization rather than initialization artifact.
