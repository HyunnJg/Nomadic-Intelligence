# Concept to Implementation Mapping

This document traces the path from theoretical concepts in `Theory_and_Axioms.md`  
to concrete code components in `run_structured.py`.

For experimental results grounded in these components, see `PAPER.md`.

---

## Core Mapping Table

| Theoretical Concept | Implementation | Code Location |
|---------------------|---------------|---------------|
| Δx as energy (not error) | `HybridDeltaTracker.compute()` | `run_structured.py` |
| Environmental shift component | `delta_env = ‖x̄_t − x̄_{t−1}‖₂` | `HybridDeltaTracker` |
| Prediction error component | `delta_err = ReLU(EMA_err − baseline_err)` | `HybridDeltaTracker` |
| Hybrid signal | `delta_hybrid = tanh(w_env·δ_env + w_err·δ_err)` | `HybridDeltaTracker` |
| Φ (Will to Resonance) | `compute_phi_signal()` | `run_structured.py` |
| β_φ scaling | `cfg.beta_phi` (default 0.02) | `config.yaml` |
| Dwell time τₖ (static floor) | `DwellTimeRegularizer`, `tau_k_min=3` | `run_structured.py` |
| Dynamic τₖ (environment-aware) | `compute_dynamic_tau(sigma2_delta)` | `HybridDeltaTracker` |
| σ²_Δ (rolling variance) | `deque(maxlen=tau_var_window)` | `HybridDeltaTracker` |
| Anti-dogmatism (entropy reward) | `dwell_bonus > tau_capacity` branch | `DwellTimeRegularizer` |
| Fixation pressure | `dwell_count ≤ tau_capacity` → `−λ·H(g)` | `DwellTimeRegularizer` |
| Expert diversity | `compute_load_balancing_loss()` | `run_structured.py` |
| Regime separation | `compute_regime_gate_stats()` → `L_sep` | `run_structured.py` |
| Within-regime consistency | `L_cons` in `compute_regime_gate_stats()` | `run_structured.py` |
| Switching policy (explicit) | `PolicyNet` | `run_structured.py` |
| Stay/switch decision | `stay_switch_head` (dim 2, softmax) | `PolicyNet` |
| Target expert preference | `target_head` (dim K, softmax) | `PolicyNet` |
| Hard/soft routing mode | `mode_head` (dim 2, softmax) | `PolicyNet` |
| Discrete gradient (STE) | Straight-Through Estimator in training loop | `run_structured.py` |
| Adaptive temperature | `compute_adaptive_temperature(phi)` | `run_structured.py` |
| GateNet input enrichment | `[x, Δx_hybrid, Δx_err] ∈ ℝ^(d+2)` | `GateNet.forward()` |
| PolicyNet input enrichment | `[x̄, Δx_hybrid, Δx_err, Φ, σ²_scaled, τ_scaled]` | `build_policy_input()` |
| Homeomorphic Identity (proxy) | Stable H → 0, Transition H stays high | Metrics in eval loop |

---

## Component Deep Dives

### 1. Δx — Change Signal as Energy

**Theory:** Δx is not prediction error to be suppressed.  
It is the primary information source about environmental change — the energy that drives routing decisions.

**Implementation:**
```python
# delta_env: how much the input distribution shifted
delta_env = torch.norm(x_mean - prev_x_mean, p=2)

# delta_err: relative error increase (above own recent baseline)
delta_err = ReLU(err_ema - err_baseline)

# hybrid: bounded combination
delta_hybrid = tanh(w_env * delta_env + w_err * delta_err)
```

Both components are fed as **separate channels** to GateNet (`[x, delta_hybrid, delta_err]`), allowing the gate to learn distinct responses to input drift vs. prediction degradation.

**Key distinction from Active Inference (Friston 2010):**  
Active Inference minimizes Δx (prediction error = free energy to reduce).  
Nomadic Intelligence uses Δx as a routing trigger — high Δx → increased switching pressure.  
Suppressing Δx would eliminate the signal the system needs to adapt.

---

### 2. Φ — Will to Resonance

**Theory:** Φ is the system's orientation toward integrating Δx rather than resisting it.  
High Φ = readiness to switch. Low Φ = readiness to fixate.

**Implementation:**
```python
phi_signal = tanh(
    phi_scale_env  * delta_env      +   # environmental shift
    phi_scale_err  * delta_err      +   # prediction error excess
    phi_scale_exp  * explanation_err +  # current routing inadequacy
    phi_scale_gap  * best_expert_gap    # potential gain from switching
)
```

Φ then modulates the routing temperature:
```python
temp = temp_stable + (temp_transition - temp_stable) * phi_val
# High Φ → high temp → high entropy (exploration/transition)
# Low  Φ → low  temp → low  entropy (fixation)
```

**Operationalization note:** Φ is a heuristic composite signal.  
Its relationship to formal uncertainty measures (epistemic uncertainty, mutual information)  
has not been established. See PAPER.md §5.3 for the corresponding limitation statement.

---

### 3. τₖ — Strategic Dwell Time

**Theory:**  
`0 < τₖ < ∞` is the formal condition for nomadic intelligence.  
- τₖ → 0: noise switching, no information extracted from current attractor  
- τₖ → ∞: dogmatic fixation (Fixed Model limit case)

**Two-layer implementation:**

**Static floor** (`tau_k_min = 3`): minimum dwell count before switching pressure activates.

**Dynamic ceiling** (`tau_dynamic`): adapts to environmental volatility.
```python
sigma2_delta = Var(recent_delta_env_window)   # rolling variance, W=8

tau_dynamic = tau_min + (tau_max - tau_min) / (1 + tau_var_scale * sigma2_delta)
# Low  sigma2 (stable env)  → tau_dynamic → tau_max (8.0) → deep fixation allowed
# High sigma2 (volatile env) → tau_dynamic → tau_min (2.0) → switching encouraged
```

**DwellTimeRegularizer sign logic (critical):**
```python
if dwell_count <= tau_capacity:
    return -penalty * entropy      # subtract negative → loss increases → fixation
else:
    excess = dwell_count - tau_capacity
    return +bonus * entropy        # subtract positive → loss decreases → switching
```

The sign is subtracted from the total loss in the training loop:  
`L_total = L_task + ... − L_dwell`  
This is correct and intentional: fixation pressure increases loss, switching pressure decreases it.

---

### 4. PolicyNet — Explicit Transition Control

**Theory:** converts implicit gate optimization into policy-driven decision-making.

**Architecture:**
```
Input: [x̄(d), Δx_hybrid(1), Δx_err(1), Φ(1), σ²_scaled(1), τ_scaled(1)]
       = d + 5 dimensions

Shared backbone: Linear(d+5 → 64) → ReLU → Linear(64 → 64) → ReLU

Three heads:
  stay_switch_head: Linear(64 → 2)  → softmax   (stay=0, switch=1)
  target_head:      Linear(64 → K)  → softmax   (preferred expert)
  mode_head:        Linear(64 → 2)  → softmax   (soft=0, hard=1)
```

Input normalization:
```python
sigma2_scaled = tanh(sigma2_delta * 10.0)       # amplify small values
tau_scaled    = tanh((dynamic_tau - 5.0) / 5.0) # center around midpoint
```

**Teacher signal (heuristic):**
- Switch if Φ > threshold OR σ²_Δ > threshold
- Hard routing if stable (Φ low AND τ_dynamic high)
- Target = argmin batch MSE per expert

**STE for discrete gradients:**  
When mode_head predicts hard(1), top-1 expert is selected via one-hot.  
The Straight-Through Estimator passes gradients through the argmax,  
allowing backpropagation through discrete routing decisions.

---

## Observability: Homeomorphic Identity Proxies

Homeomorphic Identity (the system maintains a consistent transformation law across transitions)  
is not directly measurable. The following observables serve as proxies:

| Observable | Preserved (healthy) | Broken |
|------------|--------------------|----|
| Stable Entropy | Converges toward 0 (near-hard fixation) | Remains high → no fixation |
| Transition Entropy | Remains high (≥ 0.85) | Drops → no exploration |
| ΔH = H_trans − H_stable | Large positive (≥ 0.7) | Near zero → flat behavior |
| Expert specialization | Regime-consistent Top-1 per regime | Hub dominance or uniform |
| Switch Latency | Moderate, stable across epochs | Collapses to 0 (chaotic) or diverges |

**Empirical result (Nomadic Full, 3-seed avg):**  
Stable H = 0.108, Trans H = 0.896, ΔH = 0.788 → proxies indicate preserved identity.

**Documented breakdown case:**  
Switch Latency collapse observed at Stage 1 (early training, seed 42).  
Interpreted as: gate lost consistent response to Δx → Homeomorphic Identity breakdown.

---

## One-Line Summary

> Nomadic Intelligence transforms routing from a **selection problem**  
> into a **transition dynamics control problem** —  
> not asking *which expert*, but *how, when, and for how long* to transition.
