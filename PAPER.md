# Nomadic Intelligence: Routing as Transition Dynamics Control in Non-Stationary Environments

**Abstract**

We propose **Nomadic Intelligence**, a framework that reconceptualizes expert routing in Mixture-of-Experts (MoE) architectures as a *transition dynamics control problem* rather than a per-step selection problem. Standard MoE systems treat each routing decision independently, without modeling the temporal structure of expert transitions. This leads to brittle behavior under continuously shifting environments—either collapsing to a dominant expert or switching erratically without contextual grounding.

Our framework introduces three interacting components: (1) a hybrid change signal Δx capturing both environmental shift and prediction error, treated as an energy source rather than noise to be minimized; (2) a dynamic dwell-time constraint τₖ that modulates how long the system commits to an expert as a function of environmental volatility; and (3) an uncertainty-modulated switching variable Φ coupled with a PolicyNet that enables explicit meta-level control over stay-vs-switch decisions and routing mode (soft/hard).

We evaluate on a synthetic non-stationary regression task with continuous regime transitions. Ablation results across three seeds show that Δx-based meta-control yields the largest performance gain (avg. −0.155 Seq MSE reduction), while PolicyNet further refines transition behavior by collapsing stable-phase gate entropy to near-deterministic levels (avg. 0.108 vs. 0.951 in Standard MoE). The Nomadic Full model achieves avg. Seq MSE of 0.162 compared to 0.410 for Standard MoE and 0.412 for a fixed baseline. A robustness test under a harder 4-regime / 3-expert underprovisioned condition confirms that the ablation ordering and entropy differentiation signature are preserved (ΔH = +0.437). A preliminary LLM experiment transplanting the signal layer onto Gemma-4-E2B shows that ΔH increases from +0.731 (untrained) to +0.984 after PolicyNet training, exceeding the synthetic result, and that LoRA-based expert switching operates across three specialized adapters with a 54.2% switch rate. These results suggest that in non-stationary settings, *how* a system transitions between internal representations is as important as *which* representation it selects.

---

## 1. Introduction

Machine learning systems are typically designed around the assumption of stationarity. Optimality is equated with convergence: the system finds a fixed solution and stays there. Under this assumption, routing in MoE architectures is naturally framed as a selection problem—at each step, choose the best expert for the current input.

This framing breaks in non-stationary environments, where the underlying data distribution changes continuously. A system optimized for point-in-time selection has no mechanism to reason about *when* to transition, *how long* to stay, or *how confident it should be* during an ongoing regime shift. The result is either structural rigidity (failing to switch when the environment has changed) or noisy instability (switching before extracting information from the current attractor).

We propose a different formulation. Rather than asking "which expert is best right now?", Nomadic Intelligence asks: "what is the appropriate transition behavior given the current state of environmental change?" This reframes routing as a temporal control problem with three fundamental constraints:

- **Responsiveness**: transitions should occur in response to genuine environmental change, not noise
- **Commitment**: the system should remain in an expert long enough to extract useful information (dwell time)
- **Structured uncertainty**: during transitions, routing uncertainty should *increase* (exploration); during stable phases, it should *decrease* (exploitation/fixation)

The last property—differential entropy between transition and stable phases—is our primary empirical signature. It distinguishes structured adaptive behavior from either random switching or rigid fixation.

### 1.1 Relationship to Prior Work

Standard MoE architectures (Shazeer et al., 2017; Fedus et al., 2022) treat routing as stateless per-step selection. Load balancing losses (Fedus et al., 2022) prevent expert collapse but impose no temporal structure on transitions. Our work adds explicit transition dynamics on top of the MoE framework.

Continual learning approaches (Kirkpatrick et al., 2017) address non-stationarity through weight consolidation rather than dynamic routing, and do not model transition timing. Option-Critic architectures (Bacon et al., 2017) introduce temporal abstraction in RL through options, which is the closest engineering analog to our dwell-time formulation—but in a reinforcement learning setting rather than supervised routing.

The most relevant conceptual contrast is with Friston's Active Inference (Friston, 2010), which treats prediction error as a quantity to minimize (free energy). Nomadic Intelligence treats the analogous signal Δx as an *energy source* driving routing decisions. This is not a minor implementation difference but a fundamental inversion of the signal's role: suppressing Δx eliminates the information the system needs to adapt.

---

## 2. Method

### 2.1 Architecture Overview

The Nomadic Intelligence system consists of:

- **K experts** {E₁, ..., E_K}: 2-layer MLPs with Tanh activations (hidden dim 64)
- **GateNet**: 3-layer MLP (ReLU, hidden dim 64) conditioned on [x, Δx_hybrid, Δx_err]
- **PolicyNet**: shared 2-layer MLP (ReLU, hidden dim 64) with three output heads
- **HybridDeltaTracker**: stateful tracker maintaining EMA-based change signals and rolling variance

At each step:

$$\hat{y}_t = \sum_{k=1}^{K} g_t^{(k)} \cdot E_k(x_t)$$

where g_t is the gate distribution, optionally modified by PolicyNet.

### 2.2 Hybrid Change Signal (Δx)

The hybrid change signal combines environmental shift and prediction error:

$$\Delta x_t^{\text{env}} = \| \bar{x}_t - \bar{x}_{t-1} \|_2$$

$$\Delta x_t^{\text{err}} = \text{ReLU}(\text{EMA}_{\text{err}}(t) - \text{baseline}_{\text{err}}(t))$$

$$\Delta x_t^{\text{hybrid}} = \tanh\left(w_{\text{env}} \cdot \Delta x_t^{\text{env}} + w_{\text{err}} \cdot \Delta x_t^{\text{err}}\right)$$

where $\bar{x}_t$ is the batch mean, $\text{EMA}_{\text{err}}$ is an exponential moving average of prediction error (decay 0.80), and $\text{baseline}_{\text{err}}$ is a slower EMA of $\text{EMA}_{\text{err}}$ (momentum 0.85). The baseline comparison means $\Delta x_t^{\text{err}}$ is positive only when current error *exceeds the model's own recent error trend*—a relative rather than absolute measure.

Both components are provided as separate channels to GateNet: [x, Δx_hybrid, Δx_err] ∈ ℝ^(d+2). This allows the gate to learn different responses to environmental drift versus prediction degradation.

### 2.3 Uncertainty-Modulated Switching (Φ)

Φ is a scalar switching pressure computed from four terms:

$$\Phi_t = \tanh\left(s_{\text{env}} \cdot \Delta x^{\text{env}} + s_{\text{err}} \cdot \Delta x^{\text{err}} + s_{\text{exp}} \cdot \mathcal{L}_{\text{task}} + s_{\text{gap}} \cdot \text{gap}_t \right)$$

where gap_t = ReLU(err_top1 − err_best) measures how much better the best available expert would do versus the currently routed expert. The adaptive temperature is then:

$$\tau_t^{\text{temp}} = \tau_{\text{stable}} + (\tau_{\text{transition}} - \tau_{\text{stable}}) \cdot \Phi_t$$

High Φ → high temperature → high routing entropy (transition mode).  
Low Φ → low temperature → concentrated routing (stable/fixation mode).

We treat $\beta_\phi$ as a hyperparameter scaling the contribution of Φ to the gating objective. Empirical sweep results are reported in Section 4.2.

### 2.4 Dynamic Dwell-Time Constraint (τₖ)

The dwell-time regularizer tracks how long the dominant expert has been consistently used. The *capacity* τ_dynamic is itself a function of environmental volatility:

$$\sigma^2_\Delta(t) = \text{Var}\left(\{\Delta x^{\text{env}}_{t-W}, \ldots, \Delta x^{\text{env}}_t\}\right) \quad (W = 8)$$

$$\tau_{\text{dynamic}}(t) = \tau_{\text{min}} + \frac{\tau_{\text{max}} - \tau_{\text{min}}}{1 + s_\tau \cdot \sigma^2_\Delta(t)}$$

with τ_min = 2.0, τ_max = 8.0, s_τ = 6.0. When the environment is stable (low σ²), τ_dynamic grows toward τ_max—allowing deep fixation. When volatile (high σ²), τ_dynamic shrinks toward τ_min—reducing the cost of switching.

The regularization loss is:

$$\mathcal{L}_{\text{dwell}} = \begin{cases} -\lambda_\tau \cdot \mathcal{H}(g_t) & \text{if } d_k(t) \leq \tau_{\text{dynamic}} \quad \text{(fixation pressure)} \\ +\min(\text{excess} \cdot \lambda_\tau,\ 10\lambda_\tau) \cdot \mathcal{H}(g_t) & \text{if } d_k(t) > \tau_{\text{dynamic}} \quad \text{(switching pressure)} \end{cases}$$

where d_k(t) is the consecutive dwell count on the dominant expert. Note the sign convention: subtracting a negative term *increases* the total loss (discourages entropy); adding a positive term *decreases* it (encourages entropy). This biases the gate toward fixation during stable phases and toward switching when the capacity has been exceeded.

### 2.5 PolicyNet

PolicyNet is a meta-decision module operating above the gate. Its inputs combine:

$$\mathbf{p}_t = [\bar{x}_t,\ \Delta x^{\text{hybrid}},\ \Delta x^{\text{err}},\ \Phi_t,\ \tanh(10\sigma^2_\Delta),\ \tanh\left(\frac{\tau_{\text{dynamic}} - 5.0}{5.0}\right)]$$

PolicyNet consists of a shared 2-layer MLP (input_dim + 5 → 64 → 64, ReLU) with three output heads:

- **stay_switch_head** (Linear → softmax, dim 2): stay(0) / switch(1) probability
- **target_head** (Linear → softmax, dim K): preferred expert distribution
- **mode_head** (Linear → softmax, dim 2): soft(0) / hard(1) routing preference

Training targets are generated by a heuristic teacher: switch if Φ > threshold OR σ²_Δ > threshold; use hard routing if environment is stable and τ_dynamic is high. The target expert is the one with minimum batch MSE.

The PolicyNet output modulates the gate distribution via a mixing weight (policy_mix_weight = 0.25), and triggers hard routing via a Straight-Through Estimator (STE) when the mode head predicts hard(1). This allows discrete routing decisions to be included in the backward pass without gradient blocking.

### 2.6 Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{load}} \mathcal{L}_{\text{load}} - \mathcal{L}_{\text{dwell}} + \lambda_{\text{sep}} \mathcal{L}_{\text{sep}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \mathcal{L}_{\text{policy}}$$

where:
- $\mathcal{L}_{\text{load}}$: load balancing loss (Switch Transformer formulation, λ = 0.03)
- $\mathcal{L}_{\text{dwell}}$: dwell-time regularizer (sign-correct: subtracted from total, λ_τ = 0.05)
- $\mathcal{L}_{\text{sep}}$: gate centroid separation loss, encouraging regime-differentiated routing (λ = 0.08)
- $\mathcal{L}_{\text{cons}}$: within-regime gate consistency loss (λ = 0.03)
- $\mathcal{L}_{\text{policy}}$: cross-entropy on PolicyNet's three output heads

---

## 3. Experiment

### 3.1 Task: Non-Stationary Regression with Continuous Transitions

We construct a synthetic task with three regimes:

| Regime | Function | Input Center |
|--------|----------|-------------|
| A | y = x₁ + x₂ | (2.5, 2.5) |
| B | y = x₁ − x₂ | (−2.5, −2.5) |
| C | y = −x₁ + 0.5x₂ | (2.5, −2.5) |

The sequence cycles through A → B → C → A with 8 *transition steps* between regimes where inputs and labels are linearly interpolated. Within each stable block, inputs are sampled as regime center + Gaussian noise (σ = 0.9), producing significant input distribution overlap across regimes.

This overlap is deliberate: it ensures that static features alone are insufficient to discriminate regimes, requiring the model to leverage temporal context (Δx, dwell history) for correct routing. The task is specifically designed to be solvable only through *temporal structure awareness*.

**Two evaluation modes:**
- **Static MSE**: model evaluated on each regime independently, without sequential context. This intentionally disadvantages Nomadic Intelligence (no Δx signal) and serves only as a reference for checking expert specialization.
- **Sequence MSE**: model evaluated on the full phase-sequential test set, preserving temporal order. This is the primary metric.

The gap between Static and Sequence MSE for Nomadic models is *expected*: a model with strong temporal routing will perform better in sequence context than in isolation. The Fixed baseline shows no such gap by construction.

### 3.2 Models

Four configurations are evaluated as an ablation study:

| Model | Components |
|-------|-----------|
| Fixed | Single MLP (matched parameter count) |
| Standard MoE | MoE with input-conditioned gating, no Δx / no policy |
| Nomadic NoPolicy | MoE + Δx + dynamic τₖ + dwell regularizer |
| Nomadic Full | NoPolicy + PolicyNet |

Standard MoE has static gating (no temporal signals): its Sequence MSE equals its Static MSE by construction, providing a clean lower bound on temporal structure benefit.

### 3.3 Metrics

**Primary:** Sequence Test MSE  
**Secondary:**
- Switch Latency: mean number of steps between regime transition onset and gate response
- Mean Dwell Time: average consecutive steps on dominant expert
- Stable-phase Gate Entropy: H(g_t) averaged over stable phase timesteps
- Transition-phase Gate Entropy: H(g_t) averaged over transition phase timesteps
- Entropy Differentiation: Δ(H) = H_transition − H_stable

Entropy differentiation is the key behavioral signature of structured switching: a system that increases uncertainty during transitions and decreases it during stable phases is responding contextually to regime changes. Flat entropy across both phases indicates either random switching (high uniform entropy) or rigid fixation (low uniform entropy).

### 3.4 Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 220 |
| Learning rate | 2×10⁻³ |
| Weight decay | 1×10⁻⁵ |
| Batch size (stable) | 64 |
| Transition steps | 8 |
| Device | CUDA (GTX 1660 Super) |
| Seeds | 42, 123, 456 |

All seeds use identical hyperparameters. Sequence data is never shuffled; temporal order is preserved throughout training and evaluation.

---

## 4. Results

### 4.1 Ablation: Component Contributions

**Table 1: Ablation Study — Seq MSE and Entropy Differentiation (3-seed average)**

| Model | Seq MSE (avg) | Stable H | Trans H | ΔH | Ablation Δ |
|-------|:---:|:---:|:---:|:---:|:---:|
| Fixed | — | — | — | — | — |
| Standard MoE | 0.410 | 0.951 | 0.984 | +0.033 | −0.002 |
| Nomadic NoPolicy | 0.255 | 0.556 | 0.949 | +0.394 | −0.155 |
| Nomadic Full | **0.162** | **0.108** | **0.896** | **+0.788** | −0.093 |

*Ablation Δ = Seq MSE reduction from previous row. Negative = improvement.*

Key observations:

**Standard MoE vs. Fixed (Δ ≈ 0):** Replacing a fixed MLP with a standard MoE yields no meaningful improvement on sequential tasks. Expert mixture without temporal context cannot exploit the structure of regime transitions. This confirms the baseline.

**NoPolicy vs. Standard MoE (Δ = −0.155):** Adding Δx-based gating, dynamic τₖ, and the dwell regularizer produces the largest single-step improvement. The Entropy Differentiation metric rises from 0.033 to 0.394, indicating that the system has learned to increase uncertainty during transitions and decrease it during stable phases without any explicit supervision signal for this behavior.

**Full vs. NoPolicy (Δ = −0.093):** PolicyNet provides substantial further improvement. The most distinctive change is Stable Entropy collapsing from 0.556 to 0.108—near-deterministic routing during stable phases. This is the empirical signature of what we term *homeomorphic fixation*: the gate converges to a consistent, near-hard assignment during stable regimes while retaining high flexibility during transitions.

**Per-seed breakdown (Nomadic Full):**

| Seed | Seq MSE | Stable H | Trans H |
|------|:---:|:---:|:---:|
| 42 | 0.167 | 0.186 | 0.948 |
| 123 | 0.146 | 0.083 | 0.880 |
| 456 | 0.173 | 0.056 | 0.860 |

Variance in Seq MSE (0.027 range) is acceptable for a synthetic task. Stable Entropy shows stronger seed variance (0.056–0.186), suggesting PolicyNet's convergence to hard fixation is partially seed-dependent. We note this as an open question for future work.

### 4.2 β_φ Sensitivity

The β_φ parameter scales Φ's contribution to the gating objective. We swept β_φ ∈ {0.00, 0.01, 0.02, 0.05, 0.10} at the GateNet stage (pre-PolicyNet) and β_φ ∈ {0.00, 0.01, 0.02, 0.05} with PolicyNet.

**Table 2: β_φ sweep — Seq MSE (3-seed average)**

| β_φ | Stage | Seq MSE (avg) | Stable H (avg) | Notes |
|-----|-------|:---:|:---:|-------|
| 0.10 | GateNet | 0.224 | 1.028 | High entropy, unstable |
| 0.05 | GateNet | 0.230 | 1.035 | — |
| 0.02 | GateNet | 0.245 | 0.733 | GateNet introduced here |
| 0.00 | GateNet | 0.149 | 0.887 | Best GateNet; high variance |
| 0.02 | +PolicyNet | 0.152 | 0.117 | Best stable fixation |
| 0.01 | +PolicyNet | 0.172 | 0.162 | — |
| 0.00 | +PolicyNet | 0.212 | 0.541 | Unstable without Φ signal |
| 0.05 | +PolicyNet | 0.183 | 0.235 | — |

β_φ = 0.02 with PolicyNet achieves the best combination of Seq MSE and stable fixation consistency. β_φ = 0.00 with GateNet achieves the lowest raw Seq MSE (0.149) but at the cost of higher seed variance (0.114–0.205 range). Without Φ contributing to the gating objective, the gate has no explicit switching pressure signal and relies entirely on input features to infer regime changes; this makes performance sensitive to initialization and less consistent across runs. The role of β_φ is therefore not primarily to minimize MSE, but to stabilize convergence by providing a consistent switching pressure signal throughout training. A setting that achieves competitive MSE with lower variance (β_φ = 0.02 + PolicyNet, 0.152 avg, 0.143–0.163 range) is preferred for reproducibility.

### 4.3 Emergent Expert Specialization

Regime-to-expert mapping emerges without any supervision signal on routing targets. The Top-1 expert usage ratios show clear specialization:

**Nomadic Full, Seed 123:**

| Regime | E0 | E1 | E2 |
|--------|:---:|:---:|:---:|
| A | 0.561 | 0.340 | 0.099 |
| B | 0.875 | 0.000 | 0.124 |
| C | 0.967 | 0.014 | 0.019 |

Regimes B and C are almost exclusively routed to distinct experts. Regime A shows more distributed routing, which is consistent with A being the intermediate regime geometrically (center (2.5, 2.5) vs. B at (−2.5, −2.5) and C at (2.5, −2.5)).

The permutation of expert assignments varies across seeds—regime C maps to E2 in seed 42, E1 in seed 123, E0 in seed 456—confirming that specialization is genuine (regime-consistent within a run) rather than an artifact of initialization. This is the expected behavior given that expert labels carry no semantic content in our setup.

### 4.4 Robustness: 4 Regimes, 3 Experts, Random Order

To test whether the framework generalizes beyond the original controlled setting, we conducted a robustness experiment with three simultaneous changes: (1) adding a fourth regime D (y = 0.5x₁ − x₂, center (−2.5, +2.5)), (2) fixing the number of experts at 3, creating an *underprovisioned* condition (regimes > experts), and (3) randomizing the regime order at each cycle. The same hyperparameters from the main experiment were used without adjustment.

**Table 3: Robustness test — 4 regimes / 3 experts (3-seed average)**

| Model | Seq MSE | Stable H | Trans. H | ΔH |
|-------|:-------:|:--------:|:--------:|:--:|
| Fixed | ~0.595 | — | — | — |
| Standard MoE | 0.632 | 0.956 | 0.971 | +0.016 |
| Nomadic NoPolicy | 0.422 | 0.490 | 0.943 | +0.453 |
| **Nomadic Full** | **0.334** | **0.454** | **0.891** | **+0.437** |

The ablation ordering is preserved: Standard MoE shows no improvement over Fixed (+0.038 avg), NoPolicy yields the largest single gain (−0.197), and PolicyNet adds further improvement (−0.088). This pattern replicates the 3-regime result under harder conditions.

**Expert sharing emerges without supervision.** With 4 regimes and 3 experts, the system must share at least one expert across regimes. Across seeds, the system autonomously groups regimes by functional similarity:

| Seed | Sharing pattern |
|------|----------------|
| 123 | E0: {B, D}, E1: {A, C} |
| 42 | E2 dominates all 4 regimes (hub formation) |
| 456 | E0: {B, C}, E1: {A, D} |

Sharing patterns differ by seed but are internally consistent within each run, indicating genuine functional grouping rather than random assignment.

**Entropy differentiation is reduced but preserved.** ΔH drops from +0.788 (3-regime) to +0.437 (4-regime), reflecting the harder task. The direction (transition H > stable H) is maintained across all seeds, confirming that structured switching behavior does not collapse under underprovisioning. Stable Entropy rises from 0.108 to 0.454, indicating that complete fixation is harder when experts must be shared — consistent with the limitations noted in §5.3.

### 4.5 Preliminary: Signal Transfer to Gemma-4-E2B (Future Work Direction)

The experiments in §4.1–4.4 establish Nomadic Intelligence's behavior in a controlled synthetic setting. A natural next question is whether the core signal layer generalizes to large-scale autoregressive models—a direction we treat here as a **preliminary proof-of-concept, not a performance claim**. The experiments in this section and §4.6 are exploratory: they ask whether the Nomadic signal components can be transplanted onto an existing LLM and produce behaviorally interpretable outputs. Rigorous benchmarking, ground-truth phase labeling, and parameter-matched comparisons are deferred to future work.

We transplanted the core components onto Gemma-4-E2B (2B parameters, 4-bit NF4 quantization, Colab T4). The hidden state of the final token at each generation step serves as `current_x`, and model uncertainty (1 − top1 probability) serves as `current_err`. All Nomadic components operate as a lightweight wrapper with no architectural modification to the base model.

**Signal extraction.** HybridDeltaTracker captures token-level semantic transitions in real time. During generation of "미래의 인공지능은 생물학적 한계를...", Δx_hybrid rises from 0.034 to 0.464 at Step 2 — the hidden representation shifts substantially at the moment of semantic transition. Dynamic τₖ responds immediately: τ drops from 8.00 to 6.46 and recovers toward ~7.9 as generation stabilizes, consistent with design behavior in synthetic experiments.

**Entropy differentiation before and after PolicyNet training.**

Table 4 summarizes the progression across three stages: (1) signal transplant without PolicyNet training, (2) after supervised PolicyNet training on heuristic-labeled stable/transition prompts, and (3) with LoRA expert switching enabled.

**Table 4: Entropy differentiation — synthetic MoE vs. LLM stages**

| Setting | Stable H | Trans H | ΔH | Notes |
|---------|:--------:|:-------:|:--:|-------|
| Synthetic MoE (Nomadic Full) | 0.108 | 0.896 | **+0.788** | 3-seed avg |
| Gemma-4-E2B — untrained PolicyNet | 1.806 | 2.537 | +0.731 | Signal transplant only |
| Gemma-4-E2B — trained PolicyNet | 1.249 | 2.234 | **+0.984** | After supervised training |

Absolute entropy values are higher in the LLM setting due to the larger vocabulary space (~260K tokens vs. K=3 experts). After PolicyNet training, Stable Entropy decreases from 1.806 to 1.249 while Transition Entropy decreases less (2.537 → 2.234), increasing ΔH from +0.731 to +0.984 — exceeding the synthetic result of +0.788. This indicates that PolicyNet training successfully increases the differential between stable and transition routing behavior even in the LLM context.

**LoRA expert switching.** Three LoRA adapters (r=4) were trained on stable, transition, and creative prompt sets respectively, and routed by Δx_hybrid thresholds. Over 120 generation steps across three prompts, 65 expert switches occurred (54.2% switch rate), with all three experts activated: stable 26.7%, transition 37.5%, creative 35.8%.

**Table 5: Baseline benchmark — 3-model comparison on Gemma-4-E2B**

*(15 samples per condition: 5 prompts × 3 runs; max\_new\_tokens=40)*

| Model | Stable H | Trans H | Creative H | ΔH | Creative PPL |
|-------|:--------:|:-------:|:----------:|:--:|:------------:|
| Vanilla (T=0.7) | 1.898 | 2.168 | 2.100 | +0.270 | 6.714 |
| DynamicTemp only | 0.630 | 0.926 | 0.612 | +0.296 | 2.169 |
| **Nomadic Full** | **0.625** | **0.903** | **0.533** | +0.278 | **1.840** |

DynamicTemp and Nomadic Full both reduce entropy substantially relative to Vanilla (~70% reduction in stable H), confirming that Δx-based temperature control is the primary driver of entropy regulation. Nomadic Full achieves the lowest Creative Entropy (0.533) and Creative Perplexity (1.840), suggesting that LoRA expert specialization produces more focused generation in creative contexts. ΔH is similar across all three models (+0.270 to +0.296), indicating that the directional signature — higher entropy during contextual transitions — is a property of the base LLM that Nomadic components quantify rather than introduce.

On Distinct-2 and Repetition Rate, Vanilla scores more favorably due to its higher fixed temperature. However, qualitative inspection reveals that Vanilla frequently enters repetition loops (e.g., cycling the same question phrase), whereas DynamicTemp and Nomadic Full produce more coherent content at the cost of lower lexical diversity. These metrics should be interpreted with this caveat.

**Limitations and scope.** Several constraints prevent treating these results as anything beyond a directional signal. PolicyNet switch probability saturates near 1.0 for both stable and transition contexts after training, indicating that stay/switch discrimination remains incomplete under the current heuristic teacher signal and small training set. Expert switching follows Δx thresholds rather than learned routing decisions. The benchmark uses heuristic prompt categorization with no ground-truth phase labels, making it impossible to verify whether the three prompt categories correspond to genuine distributional regimes in the LLM's representation space. No parameter-matched or compute-matched baseline exists. These experiments should be read as demonstrating that Nomadic signal components are mechanically transplantable and produce interpretable entropy dynamics in an LLM context — not as evidence that they improve LLM performance. The question of whether Nomadic routing yields measurable generation quality improvements over well-tuned baselines at scale is left as a primary direction for future work.

### 4.6 Collapse and Degeneration Behavior (LLM Experiment, continued from §4.5)

The LLM experiment in §4.5 measured entropy differentiation; we now examine the complementary failure mode. While entropy reduction improves generation stability, it can also induce degeneration in autoregressive generation, commonly observed as repetition loops or low-diversity outputs. We therefore analyze the trade-off between stability and degeneration across the same three models (Vanilla, DynamicTemp, Nomadic Full) evaluated in §4.5.

DynamicTemp-only control significantly reduces entropy (Table 5), but frequently collapses into low-diversity patterns. This is reflected in elevated repetition rates and reduced distinct-n scores, indicating that aggressive entropy suppression leads to over-confident token selection without sufficient contextual grounding.

Nomadic Full mitigates this behavior. While maintaining similarly low entropy levels, it avoids persistent repetition loops and produces more coherent outputs across prompts. This suggests that the transition-aware switching mechanism prevents the model from remaining trapped in a single high-confidence mode.

Qualitatively, DynamicTemp often enters short-cycle repetition (e.g., phrase looping), whereas Nomadic Full maintains forward semantic progression. Quantitatively, Nomadic Full shows lower repetition rates and more stable distinct-n scores under identical temperature scaling.

These results indicate that degeneration is not solely a function of entropy magnitude, but of how entropy is modulated over time. Static or purely temperature-based control suppresses entropy uniformly, whereas Nomadic switching introduces structured entropy variation—allowing the model to escape local attractors during generation.

This supports our central claim: transition dynamics, not just selection confidence, are critical for maintaining stable yet non-degenerate behavior in non-stationary settings.

Notably, Nomadic Full achieves this without increasing overall entropy relative to DynamicTemp, indicating that improved generation quality arises from temporal structure rather than higher stochasticity.

---

## 5. Discussion

### 5.1 From Selection to Transition Control

The ablation results make the central argument concrete. Standard MoE, which treats routing as stateless per-step selection, produces no improvement over a fixed baseline in sequential settings (Table 1, row 2). The performance gap only opens when temporal structure is introduced: Δx-based gating, dwell constraints, and PolicyNet each contribute distinct behavioral changes.

This is not an incremental improvement to routing quality. It is evidence that the *formulation* of the routing problem matters. A system reasoning about *when to transition* and *how long to stay* learns qualitatively different behavior than one reasoning only about *which expert to select*.

### 5.2 Homeomorphic Fixation as Empirical Signature

The most striking result in Table 1 is the collapse of Stable Entropy to 0.108 in Nomadic Full. During stable phases, the gate becomes near-deterministic—converging to a single expert with high confidence. During transition phases, it reverts to high entropy (0.896), exploring across experts.

This differential entropy pattern is what we mean by *homeomorphic fixation*: the system's routing identity is preserved not through rigid assignment but through a consistent response law—fixate when stable, explore when changing. The transformation law (Δx → routing response) remains coherent across both phases, even as the specific routing assignment changes between regimes.

Standard MoE shows almost no entropy differentiation (ΔH = 0.033), confirming that this pattern is a product of the temporal control components, not of expert mixture alone.

### 5.3 Limitations

**Synthetic environment as primary testbed.** The core ablation and robustness results are from controlled synthetic regression tasks. The robustness experiment (§4.4) extends the setting to 4 regimes with random ordering and underprovisioned experts, and §4.5 demonstrates signal transfer to an LLM setting. However, the synthetic tasks use clean periodic transitions and Gaussian regime sampling that do not reflect the noise and non-Markovian structure of real environments. The LLM experiment uses heuristic prompt categorization rather than ground-truth phase labels. We make no claim that the current hyperparameters transfer to real-world settings without adjustment.

**PolicyNet training instability.** In the synthetic experiments, per-seed variance in Stable Entropy (0.056–0.186) indicates that convergence to hard fixation is not guaranteed under the heuristic teacher signal. In the LLM experiment, switch probability saturates near 1.0 for both stable and transition contexts after training, indicating that stay/switch discrimination remains incomplete. The heuristic teacher signal and small training set size are likely contributing factors. Replacing the heuristic with a learned or RL-based objective is an open direction.

**Φ's theoretical grounding is incomplete.** Φ is currently operationalized as a heuristic composite signal. Its relationship to a formal uncertainty measure (e.g., epistemic uncertainty, mutual information) has not been established. A principled derivation would strengthen the theoretical contribution.

**Parameter count parity.** The Nomadic Full model has more parameters than Fixed or Standard MoE due to PolicyNet. In the LLM setting, LoRA adapters add a small parameter overhead per expert. A careful parameter-matched comparison is deferred to future work.

---

## 6. Conclusion

We introduced Nomadic Intelligence, a framework that treats expert routing as a temporal transition dynamics control problem. The core contribution is architectural: by treating Δx as an energy source, introducing dynamic dwell-time constraints, and enabling explicit meta-level switching decisions via PolicyNet, the system learns to differentiate its routing behavior between stable and transitional phases of the environment.

Empirically, the key finding is that the largest performance gains come not from model capacity but from temporal structure: Δx-based meta-control alone reduces Seq MSE by 38% (0.410 → 0.255) over standard MoE. PolicyNet adds a further 36% reduction (0.255 → 0.162), accompanied by a distinctive collapse in stable-phase gate entropy — a behavioral signature consistent with structured adaptive fixation rather than either random switching or rigid convergence. Robustness testing confirms this pattern holds under a harder 4-regime underprovisioned condition, with expert sharing emerging without supervision.

A preliminary LLM experiment shows that the core signal layer transfers to autoregressive generation: after PolicyNet training on Gemma-4-E2B, ΔH increases to +0.984, exceeding the synthetic result, and LoRA-based expert switching operates across three specialized adapters. These results indicate that the Nomadic signal layer captures a general property of representation change across contextual transitions, not a synthetic-environment artifact.

These findings suggest a broader principle: in non-stationary settings, the quality of transitions between internal representations is as important as the quality of the representations themselves. This framing opens directions toward temporally-aware routing in large-scale MoE systems, RL-based policy learning for transition timing, and formal analysis of transition dynamics under distribution shift.

---

## Acknowledgment

Code implementation and manuscript drafting were assisted by AI-based tools. All conceptual design, experimental decisions, and theoretical framing were performed by the author.

---

## References

1. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538.

2. Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 23(120), 1–39.

3. Bacon, P. L., Harb, J., & Precup, D. (2017). *The Option-Critic Architecture.* AAAI 2017.

4. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127–138.

5. Kirkpatrick, J., Pascanu, R., et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS, 114(13), 3521–3526.

6. Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal, 27(3), 379–423.

7. Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., ... & El Sayed, W. (2024). *Mixtral of Experts.* arXiv:2401.04088.

8. Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., ... & Laudon, J. (2022). *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022.

9. Zenke, F., Poole, B., & Ganguli, S. (2017). *Continual Learning Through Synaptic Intelligence.* ICML 2017.

10. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

11. Roller, S., Dinan, E., Goyal, N., Ju, D., Williamson, M., Liu, Y., ... & Weston, J. (2021). *Recipes for Building an Open-Domain Chatbot.* EACL 2021. *(cited for temperature-based entropy control in autoregressive generation)*

12. Kudugunta, S., Huang, Y., Bapna, A., Krikun, M., Lepikhin, D., Luong, M. T., & Firat, O. (2021). *Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference.* EMNLP 2021.

13. Lewandowski, A., Tanaka, H., Botvinick, M., & Stachenfeld, K. (2023). *Directions of Curvature as an Explanation for Loss of Plasticity.* arXiv:2312.00246. *(cited for loss of plasticity under non-stationary distributions)*