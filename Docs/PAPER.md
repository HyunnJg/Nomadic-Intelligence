#Nomadic Routing: A Control-Theoretic Framework for Transition-Aware Expert Routing under Non-Stationary Dynamics

**Abstract**

The goal of this work is not to outperform recurrent architectures in predictive accuracy, but to introduce explicit control over transition dynamics as a separate objective.

We reinterpret expert routing in mixture-of-experts (MoE) models as a control problem over transition dynamics. While existing approaches treat routing as a per-step selection mechanism, this formulation becomes limiting under non-stationary conditions where the timing and structure of transitions carry important information beyond instantaneous input features.

In this work, we introduce Nomadic Routing, a control-theoretic framework built on a structural separation between two distinct signals. The hybrid change signal (Δx) captures *whether* the environment has shifted — it is a measure of distributional uncertainty about the world. The uncertainty-modulated switching signal (Φ) addresses a different question: *given that change is occurring, should the current routing policy be revised?* We formalize Φ as approximating the free energy differential between maintaining the current expert assignment and transitioning to an alternative, operationalized as a tractable weighted composite of environmental, predictive, and counterfactual evidence terms. A dynamic dwell-time constraint and a policy network complete the framework, regulating commitment to experts and meta-level routing decisions over time.

Through synthetic and real-world experiments — including time-series forecasting and language modeling — we show that the effectiveness of the framework depends on an intermediate regime of change pressure. On a synthetic non-stationary regression task, Δx-based meta-control yields the largest single gain (−38% Seq MSE over Standard MoE). PolicyNet further reduces error and collapses stable-phase gate entropy to near-deterministic levels (0.108 vs. 0.951 in Standard MoE), a pattern we term *homeomorphic fixation*: the system fixates during stable phases and switches adaptively during transitions, preserving routing identity as a consistent response law rather than a fixed assignment. A parameter-matched baseline confirms that gains do not arise from increased capacity. When dynamics are either too smooth or too volatile, these effects diminish.

Our results suggest that routing performance is not solely determined by representational capacity, but by how transition dynamics are structured and utilized. These improvements should be interpreted as evidence of transition-aware behavior rather than as claims of superior predictive performance.

---

## 1 Introduction

Mixture-of-experts (MoE) architectures are typically formulated as a routing problem, where a gating function selects experts at each step. This formulation implicitly assumes that routing is a stateless selection process, driven solely by instantaneous input features.

However, in non-stationary environments, the structure of transitions — when to switch experts, when to remain, and how long to commit — carries information that cannot be captured by per-step decisions alone. Stateless routing mechanisms ignore this temporal structure, limiting their ability to respond coherently to regime changes.

In this work, we reinterpret expert routing as a control problem over transition dynamics. Rather than selecting experts independently at each step, we model routing as a process that regulates transitions over time, balancing stability and adaptation under changing conditions.

We introduce Nomadic Routing, a control-theoretic framework for transition-aware expert routing. The framework is built on a structural separation between two signals with distinct roles: a hybrid change signal (Δx) that measures *whether* the environment has shifted, and an uncertainty-modulated switching signal (Φ) that determines *whether a routing revision is warranted* given that shift. These are complemented by a dynamic dwell-time constraint that regulates expert commitment and a policy network that operates at the meta-level above the gate.

Across synthetic and real-world experiments, we show that the effectiveness of routing depends on an intermediate regime of change pressure. Under these conditions, the model exhibits structured switching behavior, characterized by stable fixation during consistent phases and adaptive exploration during transitions — a pattern we term *homeomorphic fixation*.

Importantly, our objective differs from recurrent approaches that aim to improve predictive accuracy through implicit temporal encoding. Instead, we focus on explicitly shaping the dynamics of transitions, treating routing as a control problem rather than a prediction problem.

### 1.1 Relationship to Prior Work

Standard MoE architectures [16] treat routing as stateless per-step selection. Load balancing losses [13] prevent expert collapse but impose no temporal structure on transitions. While large-scale sparse models like Switch Transformers [13] and Mixtral [7] have demonstrated the efficiency of expert specialization, their routing remains primarily a stateless selection problem. Our framework departs from these and alternative mechanisms like Expert Choice Routing [8] or Hash Layers [11] by explicitly modeling the temporal sequence of expert transitions.

The hybrid signal Δx draws inspiration from Bayesian changepoint detection [1], identifying regime shifts through distributional drift rather than per-step noise. Furthermore, our formulation of strategic dwell time τₖ instantiates the Marginal Value Theorem [3] within the context of neural routing, balancing the benefit of current expert commitment against the pressure to transition in volatile environments. This connection supports interpreting routing as a form of temporal control rather than static selection, and is related to the option-critic architecture [2], which introduces temporal abstraction in RL through options — our formulation, however, operates in supervised routing rather than reinforcement learning, and treats dwell time as an environmentally-adaptive constraint rather than a learned sub-policy.

The most relevant conceptual connection is with Friston's Free Energy Principle (FEP) [4, 15]. FEP posits that adaptive systems minimize surprise — or equivalently, the upper bound on sensory entropy — by updating their generative models to better predict incoming signals. This is a powerful and broadly applicable framework: the reason biological agents can minimize surprise at all is that the world itself exhibits structured, interpretable regularities. A universe of pure noise would make FEP-based adaptation impossible.

Nomadic Routing generalizes this insight rather than contradicting it. Where FEP treats prediction error as a quantity to suppress unconditionally, we observe that unconditional suppression leads to a degenerate stable state: a system that fixates permanently on the current expert representation, regardless of whether the environment has changed. The cost of pure surprise-minimization, absent a mechanism for strategic transition, is pathological rigidity.

Our framework resolves this by separating the roles of Δx and Φ. Δx quantifies environmental uncertainty — the raw signal that the world's structure has shifted. Φ is not a second uncertainty measure, but a *decision variable*: it approximates the free energy differential between maintaining the current routing policy and transitioning to a new one. This separation is architectural, not merely terminological:

| | FEP / Standard MoE | Nomadic Routing |
|---|---|---|
| Role of Δx | Prediction error to suppress | Environmental uncertainty signal to integrate |
| Role of Φ | — | Free energy differential: cost of staying vs. gain of switching |
| Objective | Minimize surprise toward zero | Navigate the boundary between fixation and transition |
| Stable state | Convergence (entropy suppression as terminal goal) | Homeomorphic fixation with finite dwell time |
| Identity | Fixed generative model or static gate | Consistent transformation law under change |

A system that treats Δx solely as noise to be eliminated will converge — but convergence is not always correct. What Nomadic Routing adds is a second layer of judgment: given that uncertainty exists (Δx > 0), *is it informative enough to warrant a transition?* This is precisely the question that Φ answers. Rather than routing away from uncertainty, Φ uses it as navigational evidence, enabling the system to maintain structured fixation when uncertainty is below the transition threshold, and to switch adaptively when it is not. This perspective aligns with broader views of adaptive systems where identity emerges from consistent response laws under varying conditions [20], and leverages information-theoretic measures [6, 27] to characterize routing transitions [21].

Continual learning approaches [5, 9] address non-stationarity through weight consolidation rather than dynamic routing, and do not model transition timing. The loss of plasticity in non-stationary distributions [12, 14, 30] poses a fundamental challenge to long-term adaptation. While traditional continual learning methods [5, 9, 22] focus on weight preservation — whether through elastic weight consolidation, synaptic intelligence, or episodic memory — our approach targets the routing level. Prior work on sparse model adaptation [23] and knowledge transfer across domains [25, 26] demonstrates that architectural routing structure itself can serve as a mechanism for preserving specialization; our framework operationalizes this insight through explicit transition dynamics control rather than post-hoc weight regularization.

Taken together, these perspectives position Nomadic Routing not as a modification of existing MoE methods, but as a reframing of routing as a control problem over transition dynamics under non-stationary conditions.

### 1.2 Contributions

We reformulate MoE routing as a control problem over transition dynamics and show that its effectiveness depends on an intermediate regime of change pressure. Our contributions are as follows:

- We propose **Nomadic Routing**, a framework that models expert routing as a temporally structured control process rather than a stateless selection mechanism.
- We introduce a structural separation between the hybrid change signal (Δx), which measures environmental uncertainty, and the switching signal (Φ), which approximates the free energy differential between maintaining and revising the current routing policy — a distinction that enables adaptive switching and stable fixation under changing conditions.
- We empirically demonstrate that the effectiveness of routing depends on an intermediate regime of change pressure, identifying boundary conditions where the approach succeeds or fails.
- We validate the framework across synthetic tasks, real-world time series, and language modeling, showing consistent behavioral patterns across domains.
- We demonstrate that predictive accuracy and transition structure are separable objectives, requiring different inductive biases (see §4.10).

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

Δx serves as an evidence signal contributing to the posterior comparison implemented by Φ.

The hybrid change signal combines environmental shift and prediction error:

$$\Delta x_t^{\text{env}} = \| \bar{x}_t - \bar{x}_{t-1} \|_2$$

$$\Delta x_t^{\text{err}} = \text{ReLU}(\text{EMA}_{\text{err}}(t) - \text{baseline}_{\text{err}}(t))$$

$$\Delta x_t^{\text{hybrid}} = \tanh\left(w_{\text{env}} \cdot \Delta x_t^{\text{env}} + w_{\text{err}} \cdot \Delta x_t^{\text{err}}\right)$$

where $\bar{x}_t$ is the batch mean, $\text{EMA}_{\text{err}}$ is an exponential moving average of prediction error (decay 0.80), and $\text{baseline}_{\text{err}}$ is a slower EMA of $\text{EMA}_{\text{err}}$ (momentum 0.85). The baseline comparison means $\Delta x_t^{\text{err}}$ is positive only when current error *exceeds the model's own recent error trend*—a relative rather than absolute measure.

Both components are provided as separate channels to GateNet: [x, Δx_hybrid, Δx_err] ∈ ℝ^(d+2). This allows the gate to learn different responses to environmental drift versus prediction degradation.

### 2.3 Uncertainty-Modulated Switching (Φ)

#### Conceptual Motivation

The separation between Δx and Φ is structural rather than cosmetic. Δx measures *whether* the environment has changed — it is a signal about the world. Φ addresses a different question: *given that change is occurring, should the current routing policy be revised?*

This corresponds to a decision-theoretic comparison between maintaining the current expert assignment and transitioning to an alternative. We interpret Φ as a control variable that governs this trade-off.

Formally, we define Φ as approximating the **difference in expected free energy** between two actions:

$$\Phi_t \approx \mathcal{F}_t(\text{stay}) - \mathcal{F}_t(\text{switch})$$

When $\Phi_t > 0$, transitioning reduces expected free energy; when $\Phi_t \approx 0$, the current assignment remains near-optimal, leading to stable fixation.

#### Theoretical Mapping

We further interpret Φ as a **functional approximation to a log-posterior ratio**:

$$\Phi_t \approx \log \frac{P(\text{switch} \mid \mathcal{D}_t)}{P(\text{stay} \mid \mathcal{D}_t)}$$

where $\mathcal{D}_t$ denotes the current observation context.

Under mild assumptions (e.g., independent evidence sources and locally linear aggregation), this log-ratio can be approximated as a weighted sum of evidence terms:

$$\Phi_t \propto \sum_i s_i \cdot \phi_i(\cdot)$$

where $s_i$ controls the **precision (confidence)** of each information source.

#### Operationalization

In practice, Φ is computed as:

$$\Phi_t = \tanh\left(s_{\text{env}} \cdot \Delta x^{\text{env}} + s_{\text{err}} \cdot \Delta x^{\text{err}} + s_{\text{exp}} \cdot \mathcal{L}_{\text{task}} + s_{\text{gap}} \cdot \text{gap}_t \right)$$

where:

- $\Delta x^{\text{env}}$ — *Environmental prior*: captures distributional shift, reducing the prior probability that the current expert remains valid.
- $\Delta x^{\text{err}}$ — *Likelihood update*: reflects evidence that the current expert's predictive likelihood has degraded relative to its recent baseline.
- $\mathcal{L}_{\text{task}}$ — *Absolute error signal*: indicates global mismatch between model predictions and observations.
- $\text{gap}_t$ — *Lower bound of expected information gain*: measures the performance deficit between the current expert and the best available alternative:

$$\text{gap}_t = \text{ReLU}(\epsilon_{\text{top1}} - \epsilon_{\text{best}})$$

This term directly approximates the utility of switching.

#### Interpretation

Each component of Φ corresponds to a distinct aspect of Bayesian inference:

| Component | Role |
|---|---|
| $\Delta x^{\text{env}}$ | Prior shift (epistemic uncertainty) |
| $\Delta x^{\text{err}}$ | Likelihood evidence |
| $\mathcal{L}_{\text{task}}$ | Absolute model mismatch |
| $\text{gap}_t$ | Expected information gain (decision utility) |

Together, these terms approximate the question: *Is the expected cost of remaining with the current expert greater than the expected cost of switching?*

Importantly, Φ does **not** compute an exact posterior, as doing so would require maintaining full distributions over all experts. Instead, it preserves the **directional structure of Bayesian inference** in a computationally tractable form.

#### Control Effect on Routing

Φ modulates routing entropy through temperature control:

$$\tau_t^{\text{temp}} = \tau_{\text{stable}} + (\tau_{\text{transition}} - \tau_{\text{stable}}) \cdot \Phi_t$$

- High Φ → high temperature → high entropy → **transition mode**
- Low Φ → low temperature → low entropy → **Homeomorphic Fixation mode**

This establishes Φ as a **continuous transition gate** rather than a binary switching trigger.

#### Why Not Information-Geometric Φ?

Divergence-based alternatives (e.g., KL or JSD between gate distributions) fail under stable fixation:

$$\bar{g}_t \approx \bar{g}_{t-1} \;\Rightarrow\; D(\bar{g}_t \,\|\, \bar{g}_{t-1}) \approx 0$$

This causes switching pressure to collapse precisely when the dwell-time constraint requires it. The inclusion of $\text{gap}_t$ resolves this issue by providing a **task-level signal** that remains informative even when routing distributions are stationary. Empirical comparison against JSD, KL, and heterogeneity-based variants is reported in §4.7.

#### Limitations

The relationship between Φ and a formally derived Bayesian posterior or mutual information objective is not exact and remains an open question. We therefore interpret Φ as a **structured approximation** to the decision variable governing routing transitions, rather than a fully derived probabilistic quantity (see §5.3). We treat $\beta_\phi$ as a hyperparameter scaling Φ's contribution to the gating objective; empirical sweep results are reported in §4.2.

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

The switching signal Φ enters the objective indirectly: it modulates the routing temperature (§2.3), shaping the gate distribution that all loss terms act upon, and contributes to the gating objective scaled by $\beta_\phi$ (§4.2).

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

Four configurations are evaluated as an ablation ladder:

| Model | Components |
|-------|-----------|
| Fixed | Single MLP, hidden_dim 64 (4,417 params) |
| Standard MoE | MoE with input-conditioned gating, no Δx / no policy (17,798 params) |
| Nomadic NoPolicy | MoE + Δx + dynamic τₖ + dwell regularizer (22,926 params) |
| Nomadic Full | NoPolicy + PolicyNet (23,053 params) |

The ladder has a theoretical motivation: Fixed MLP corresponds to the limiting case of Nomadic Routing where dwell time is unconstrained ($\tau_k \to \infty$), collapsing to permanent fixation in a single attractor. Standard MoE introduces multiple attractors but no temporal structure over transitions. The NoPolicy and Full configurations progressively add the temporal control components that distinguish the framework. This ordering reflects increasing degrees of transition dynamics awareness rather than arbitrary model selection.

Parameter-matched variants (Fixed hidden_dim 150 / Standard MoE hidden_dim 74, both ≈ 23,300 params) are evaluated separately in §4.9 to isolate the contribution of temporal structure from model capacity. A GRU-gated MoE baseline (26,502 params) and regime-specialist Oracle are evaluated in §4.10 to compare implicit vs. explicit temporal learning.

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

All seeds use identical hyperparameters. Sequence data is never shuffled; temporal order is preserved throughout training and evaluation. All experiments were run using `config_extended.yaml` as the hyperparameter source; the YAML file takes precedence over code-level defaults where values differ (notably β_φ = 0.02 as reported in §4.2).

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

β_φ = 0.02 with PolicyNet achieves the best combination of Seq MSE and stable fixation consistency. β_φ = 0.00 with GateNet achieves the lowest raw Seq MSE (0.149) but at the cost of higher seed variance (0.114–0.205 range). Without Φ contributing to the gating objective, the gate has no explicit switching pressure signal and relies entirely on input features to infer regime changes; this makes performance sensitive to initialization and less consistent across runs. The role of β_φ is therefore not primarily to minimize MSE, but to stabilize convergence by providing a consistent switching pressure signal throughout training. A setting that achieves competitive MSE with lower variance (β_φ = 0.02 + PolicyNet, 0.152 avg, 0.121–0.182 range) is preferred for reproducibility.

We additionally confirmed that the β_φ ≥ 0.02 stability floor holds after Dynamic τₖ is introduced: at β_φ = 0.00 and β_φ = 0.01, Static MSE collapses to 4.72 and 4.42 respectively in at least one seed, replicating the pre-PolicyNet instability pattern. Dynamic τₖ alone is insufficient to stabilize Static MSE without the Φ signal contribution.

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

To test whether the framework generalizes beyond the original controlled setting, we conducted a robustness experiment with three simultaneous changes: (1) adding a fourth regime D (y = −x₁ − x₂, center (−2.5, +2.5)), (2) fixing the number of experts at 3, creating an *underprovisioned* condition (regimes > experts), and (3) randomizing the regime order at each cycle. The same hyperparameters from the main experiment were used without adjustment.

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

### 4.5 Preliminary: Signal Transfer to LLMs (Gemma-4-E2B and Gemma-4-26B)

These results should not be interpreted as improvements in language modeling performance, but as evidence that the proposed signal produces consistent behavioral patterns across scales.

The experiments in §4.1–4.4 establish Nomadic Intelligence's behavior in a controlled synthetic setting. A natural next question is whether the core signal layer generalizes to large-scale autoregressive models — a direction we treat here as a **preliminary proof-of-concept, not a performance claim**. The experiments in this section and §4.6 are exploratory: they ask whether the Nomadic signal components can be transplanted onto existing LLMs and produce behaviorally interpretable outputs across model scales. Rigorous benchmarking, ground-truth phase labeling, and parameter-matched LLM comparisons are deferred to future work.

We conducted two experiments: (1) signal transplant onto Gemma-4-E2B (2B parameters, 4-bit NF4, Colab T4) with heuristic stable/transition/creative prompt categories; (2) signal transplant onto Gemma-4-26B (A4B, 4-bit NF4, Colab A100) with task-domain LoRA experts (math / code / language). In both cases, the hidden state of the final token at each generation step serves as `current_x`, and model uncertainty (1 − top1 probability) serves as `current_err`. All Nomadic components operate as a lightweight wrapper with no architectural modification to the base model.

**Signal extraction (E2B).** HybridDeltaTracker captures token-level semantic transitions in real time. During generation of "미래의 인공지능은 생물학적 한계를...", Δx_hybrid rises from 0.034 to 0.464 at Step 2 — the hidden representation shifts substantially at the moment of semantic transition. Dynamic τₖ responds immediately: τ drops from 8.00 to 6.46 and recovers toward ~7.9 as generation stabilizes, consistent with design behavior in synthetic experiments.

**Entropy differentiation — E2B (heuristic prompt categories).**

Table 4 summarizes the E2B progression across three stages: (1) signal transplant without PolicyNet training, (2) after supervised PolicyNet training on heuristic-labeled stable/transition prompts, and (3) with LoRA expert switching enabled.

**Table 4: Entropy differentiation — synthetic MoE vs. LLM experiments**

| Setting | Stable H | Trans H | ΔH | Notes |
|---------|:--------:|:-------:|:--:|-------|
| Synthetic MoE (Nomadic Full) | 0.108 | 0.896 | **+0.788** | 3-seed avg |
| Gemma-4-E2B — untrained PolicyNet | 1.806 | 2.537 | +0.731 | Signal transplant only |
| Gemma-4-E2B — trained PolicyNet | 1.249 | 2.234 | **+0.984** | After supervised training |
| Gemma-4-26B — untrained PolicyNet | 0.340 | 0.890 | +0.550 | Task-domain LoRA |
| Gemma-4-26B — trained PolicyNet | **0.177** | 0.675 | +0.498 | After supervised training |

Absolute entropy values differ across model scales: E2B produces higher baseline entropy (~1.8–2.5) than 26B (~0.2–1.5) due to the larger model's higher generation confidence. This scale difference is consistent across all three conditions (Vanilla, DynamicTemp, Nomadic Full) in the 26B benchmark, confirming it reflects model-level baseline confidence rather than a reduction in entropy differentiation capability. Within each model, the directional pattern holds: PolicyNet training increases ΔH in both cases.

**LoRA expert switching (E2B).** Three LoRA adapters (r=4) trained on stable, transition, and creative prompt sets, routed by Δx_hybrid thresholds. Over 120 generation steps across three prompts, 65 expert switches occurred (54.2% switch rate), with all three experts activated: stable 26.7%, transition 37.5%, creative 35.8%.

**Gemma-4-26B experiment — task-domain LoRA experts.**

To address the ground-truth labeling limitation of the E2B experiment, the 26B experiment uses task-domain LoRA experts with clearer distributional boundaries: Expert A (math) trained on mathematical reasoning prompts, Expert B (code) trained on code generation prompts, Expert C (language) trained on free-text generation prompts. Math and code constitute the stable domain (high model confidence, low prediction uncertainty); language constitutes the transition domain (exploratory, higher uncertainty). This provides a more principled phase label than the E2B heuristic categories.

**Table 5: Baseline benchmark — 3-model comparison on Gemma-4-26B**

*(45 samples: 15 prompts × 3 runs; max\_new\_tokens=40; task-domain phase labeling)*

| Model | Stable H | Trans H | ΔH | Lang PPL | Switch Rate |
|-------|:--------:|:-------:|:--:|:--------:|:-----------:|
| Vanilla (T=0.7) | 0.473 | 1.463 | +0.991 | 3.045 | 0.000 |
| DynamicTemp only | 0.187 | 0.728 | +0.541 | 2.399 | 0.000 |
| **Nomadic Full** | **0.177** | **0.675** | +0.498 | **1.948** | **0.046** |

Nomadic Full achieves the lowest Stable Entropy (0.177) among all three conditions, consistent with stable-phase fixation. DynamicTemp reduces entropy substantially but produces higher language-domain perplexity (2.399) than Nomadic Full (1.948) — a 19.4% gap — suggesting that LoRA expert specialization contributes meaningfully to generation quality beyond what temperature modulation alone provides. ΔH is lower than the E2B result (+0.498 vs. +0.984), consistent with the model-scale entropy compression observed across all conditions. Expert switching occurs at a 4.6% rate overall, with code domain (7.7%) seeing more switches than math (5.7%) or language (0.3%) — the near-zero language switch rate indicates that PolicyNet routes language prompts to the stable expert during generation, a limitation discussed below.

**Table 6: E2B baseline benchmark (heuristic categories, for reference)**

*(15 samples per condition: 5 prompts × 3 runs; max\_new\_tokens=40)*

| Model | Stable H | Trans H | Creative H | ΔH | Creative PPL |
|-------|:--------:|:-------:|:----------:|:--:|:------------:|
| Vanilla (T=0.7) | 1.898 | 2.168 | 2.100 | +0.270 | 6.714 |
| DynamicTemp only | 0.630 | 0.926 | 0.612 | +0.296 | 2.169 |
| **Nomadic Full** | **0.625** | **0.903** | **0.533** | +0.278 | **1.840** |

DynamicTemp and Nomadic Full both reduce entropy substantially relative to Vanilla (~70% reduction in stable H), confirming that Δx-based temperature control is the primary driver of entropy regulation. Nomadic Full achieves the lowest Creative Perplexity (1.840), consistent with the 26B finding that LoRA specialization benefits generation quality in exploratory contexts.

**Limitations and scope.** Several constraints apply to both experiments. In the E2B experiment, PolicyNet switch probability saturates near 1.0 for both stable and transition contexts after training, indicating that stay/switch discrimination remains incomplete. In the 26B experiment, language-domain switch rate is near-zero (0.3%), indicating that during generation the PolicyNet routes language prompts to the stable expert — the domain label provides a principled training signal, but the learned policy does not reliably execute language-domain switching at inference time. In both experiments, expert switching follows a combination of Δx thresholds and PolicyNet output rather than purely learned routing decisions. No parameter-matched LLM baseline exists. These experiments demonstrate that Nomadic signal components are mechanically transplantable and produce interpretable entropy dynamics across two model scales — they do not constitute evidence of general LLM performance improvement.

### 4.6 Collapse and Degeneration Behavior (LLM Experiment, continued from §4.5)

The LLM experiments in §4.5 measured entropy differentiation; we now examine the complementary failure mode. While entropy reduction improves generation stability, it can also induce degeneration in autoregressive generation, commonly observed as repetition loops or low-diversity outputs. We analyze the trade-off between stability and degeneration in both the E2B and 26B experiments.

In both settings, DynamicTemp-only control significantly reduces entropy but frequently collapses into low-diversity patterns — elevated repetition rates indicate that aggressive entropy suppression leads to over-confident token selection without sufficient contextual grounding. On 26B, DynamicTemp achieves Stable H of 0.187 and repetition rate of 0.166 (stable domain); Nomadic Full matches the entropy level (0.177) but reduces repetition rate to 0.128, a 22.8% reduction.

Nomadic Full mitigates degeneration in both experiments. While maintaining entropy levels comparable to DynamicTemp, it avoids persistent repetition loops and produces more coherent outputs. The transition-aware switching mechanism prevents the model from remaining trapped in a single high-confidence mode — on 26B, language-domain perplexity drops from 2.399 (DynamicTemp) to 1.948 (Nomadic Full), confirming that generation quality improvement comes from temporal routing structure rather than from entropy magnitude alone.

These results support the central claim: transition dynamics, not just selection confidence, are critical for maintaining stable yet non-degenerate behavior. The finding holds across both model scales despite the absolute entropy difference between E2B and 26B.

### 4.7 Φ Variant Comparison

To empirically evaluate alternative formulations of Φ and address the partial theoretical grounding noted in §5.3, we compared four variants against the EMA composite baseline across three seeds on the same synthetic regression task used in §4.1–4.3.

**Variants.**

- **Phi_EMA** (baseline): $\Phi = \tanh(s_\text{env} \cdot \Delta x^\text{env} + s_\text{err} \cdot \Delta x^\text{err} + s_\text{exp} \cdot \mathcal{L}_\text{task} + s_\text{gap} \cdot \text{gap}_t)$
- **Phi_JSD**: $\Phi = \tanh(\alpha \cdot \text{JSD}(\bar{g}_t \| \bar{g}_{t-1}))$, Jensen-Shannon divergence between consecutive batch-mean gate distributions
- **Phi_KL**: $\Phi = \tanh(\alpha \cdot \text{KL}(\bar{g}_t \| \bar{g}_{t-1}))$, asymmetric forward KL divergence
- **Phi_Switch**: $\Phi = \text{stay\_switch\_probs}[:,1]$, PolicyNet switch head output used directly as Φ
- **Phi_JSD_v2**: $\Phi = \tanh(s_\text{div} \cdot \text{std}_i[\text{JSD}(g_i \| \bar{g}_t)] + s_\text{ema} \cdot \text{EMA}(\text{mean}_i[\text{JSD}(g_i \| \bar{g}_t)]))$, intra-batch routing heterogeneity

**Results.**

| Φ Variant | Seq MSE | ΔH | Stable Ent | Switch Lat |
|---|---|---|---|---|
| Phi_EMA | 0.285 ± 0.018 | **0.544 ± 0.001** | **0.324** | 0.454 |
| Phi_JSD_v2 | 0.276 ± 0.030 | 0.444 ± 0.048 | 0.514 | 1.176 |
| Phi_JSD | 0.264 ± 0.043 | 0.289 ± 0.025 | 0.644 | 1.222 |
| Phi_KL | **0.249 ± 0.019** | 0.338 ± 0.024 | 0.594 | 0.407 |
| Phi_Switch | 0.441 ± 0.181 | 0.141 ± 0.048 | 0.813 | 1.204 |

Phi_EMA achieves the highest ΔH (0.544) with near-zero variance across seeds and is the only variant to produce sharp stable-phase fixation (Stable Entropy 0.324). Information-geometric variants (Phi_JSD, Phi_KL) achieve competitive Seq MSE but fail to reproduce the entropy differentiation signature: their Φ values collapse near zero during stable phases because a fixated gate satisfies $\bar{g}_t \approx \bar{g}_{t-1}$, causing JSD/KL divergence to vanish precisely when the system needs sustained switching pressure from the DwellTimeRegularizer.

Phi_JSD_v2 partially addresses this by computing per-sample routing heterogeneity $\text{std}_i[\text{JSD}(g_i \| \bar{g}_t)]$ rather than batch-mean divergence, maintaining a nonzero Φ signal even during fixation. ΔH improves to 0.444, but falls below Phi_EMA (0.544) because intra-batch routing noise is a less informative signal than the task-aware explanation deficit $\text{gap}_t = \text{ReLU}(\epsilon_{\text{top1}} - \epsilon_{\text{best}})$.

Phi_Switch (end-to-end learned Φ) collapses across seeds (Seq MSE 0.441 ± 0.181) with the PolicyNet switch head saturating at switch probability 1.0 within the first 25 epochs — confirming the training instability noted in §5.3 and motivating RL-based policy learning as future work.

The experiment provides post-hoc empirical justification for the EMA composite design: the combination of environment change detection ($\Delta x^\text{env}$) and task-level explanation deficit ($\text{gap}_t + \mathcal{L}_\text{task}$) is necessary for sustained switching pressure during stable phases. Neither component alone suffices. The remaining theoretical gap — establishing the precise conditions under which the four-term composite constitutes a valid lower bound on expected free energy reduction from a routing transition — is deferred to future work (see §5.3).

### 4.8 Task Generalization

The experiments in §4.1–4.7 use a controlled synthetic task with linear regime functions, gradual transitions (steps=8), and Gaussian noise. To assess whether the reported gains depend on these specific design choices, we evaluated five task variants while holding all other hyperparameters fixed.

**Variants.** (1) *Nonlinear*: regime functions replaced with sin(x₁)·x₂, x₁²−x₂², and tanh(x₁+x₂)·|x₂|; input centers unchanged. (2) *Abrupt*: transition_steps reduced from 8 to 2. (3) *Gradual*: transition_steps increased from 8 to 24. (4) *Heavy-tail*: input noise sampled from Student-T (df=2) instead of Gaussian. (5) *Combined*: nonlinear functions + abrupt transitions + heavy-tail noise simultaneously.

**Models evaluated**: Fixed MLP, Standard MoE, Nomadic Full (3 seeds each). Nomadic NoPolicy is omitted as its ablation contribution was established in §4.1.

**Table 7: Task Generalization Results (3-seed mean)**

| Variant | Model | Seq MSE | ΔH | Stable Ent |
|---------|-------|:-------:|:--:|:----------:|
| §4.1 (reference) | Fixed | 0.409 | — | — |
| §4.1 (reference) | Standard MoE | 0.410 | 0.017 | 0.952 |
| §4.1 (reference) | **Nomadic Full** | **0.165** | **0.781** | **0.091** |
| Nonlinear | Fixed | 0.221 | — | — |
| Nonlinear | Standard MoE | 0.280 | 0.071 | 0.977 |
| Nonlinear | **Nomadic Full** | **0.174** | **0.341** | 0.634 |
| Abrupt | Fixed | 0.307 | — | — |
| Abrupt | Standard MoE | 0.317 | 0.012 | 1.061 |
| Abrupt | **Nomadic Full** | **0.160** | −0.009 | 0.985 |
| Gradual | Fixed | 0.436 | — | — |
| Gradual | Standard MoE | 0.416 | 0.074 | 0.984 |
| Gradual | Nomadic Full | 0.424 | 0.111 | 0.775 |
| Heavy-tail | Fixed | 5.657 | — | — |
| Heavy-tail | Standard MoE | 6.160 | 0.068 | 0.443 |
| Heavy-tail | **Nomadic Full** | **4.415** | 0.006 | 0.896 |
| Combined | Fixed | 7,771 | — | — |
| Combined | Standard MoE | 16,846 | 0.006 | 0.519 |
| Combined | Nomadic Full | 17,769 | 0.025 | 0.401 |

**Nonlinear.** Nomadic Full achieves Seq MSE 0.174 vs. Standard MoE 0.280 (−38%), and ΔH 0.341 vs. 0.071 (4.8×). ΔH is reduced from the §4.1 level (0.781) but remains substantially positive, and Stable Entropy (0.634) is clearly lower than Standard MoE (0.977). The mechanism operates under nonlinear functions; the reduction in ΔH reflects the increased representational demand on the Expert networks rather than a failure of the temporal control components.

**Abrupt (steps=2).** Nomadic Full achieves Seq MSE 0.160 vs. Standard MoE 0.317 (−50%), preserving the MSE advantage at the same magnitude as §4.1. However, ΔH collapses to −0.009. This is a structural measurement artifact rather than a behavioral failure: with transition_steps=2, the transition window spans only 2 batches compared to 40 stable-phase batches per cycle, making stable/transition entropy differentiation statistically unreliable. Seeds 123 and 456 achieve Seq MSE of 0.123 and 0.087 respectively — below the §4.1 result — suggesting that abrupt transitions favor rapid switching strategies that do not require stable-phase fixation.

**Gradual (steps=24).** All three models show increased MSE relative to §4.1. Nomadic Full (0.424) performs comparably to Standard MoE (0.416), with high seed variance (std=0.152). With steps=24, the transition proportion of the full sequence rises from ~23% (steps=8) to ~50%, causing the Δx_env signal to remain in a perpetual intermediate state and preventing the DwellTimeRegularizer from establishing stable-phase fixation. This is a genuine limitation of the current framework.

**Heavy-tail.** Nomadic Full achieves the lowest absolute MSE (4.415) among all models, outperforming both Standard MoE (6.160) and Fixed (5.657). ΔH, however, collapses to 0.006. Student-T noise introduces sporadic extreme values that cause episodic spikes in Δx_err, which the error-EMA baseline cannot distinguish from genuine regime transitions. This corrupts the Φ signal, preventing the stable/transition entropy differentiation while leaving the overall MSE ranking intact.

**Combined.** All models fail under simultaneous nonlinear + abrupt + heavy-tail conditions, with MSE in the thousands-to-tens-of-thousands range and extreme seed variance. Nomadic Full performs comparably to Standard MoE (both ~17,000 average), with no meaningful ΔH. The interaction of nonlinear output scale, extreme noise values, and the abrupt transition window destabilizes PolicyNet training. This represents the current architecture's stability boundary under compound stress.

**Summary.** Table 8 characterizes each variant:

**Table 8: Task Generalization Summary**

| Variant | MSE advantage | ΔH preserved | Primary finding |
|---------|:---:|:---:|---|
| Nonlinear | ✓ +0.105 | △ 0.341 | Mechanism transfers to nonlinear functions |
| Abrupt | ✓ +0.157 | ✗ (structural) | MSE gains robust; ΔH measurement breaks down |
| Gradual | ✗ ≈0 | △ 0.111 | Long transitions prevent stable-phase fixation |
| Heavy-tail | ✓ +1.745 | ✗ (noise) | MSE robust; Δx_err sensitive to extreme noise |
| Combined | ✗ −923 | ✗ | Compound stress exceeds stability boundary |

### 4.9 Parameter-Matched Baseline

A potential confound in the ablation results (§4.1) is model capacity: Nomadic Full contains PolicyNet, adding approximately 5,255 parameters over Standard MoE (23,053 vs. 17,798 total). To rule out the possibility that performance gains arise from this additional capacity rather than from temporal control mechanisms, we evaluated parameter-matched baselines at Nomadic Full's parameter count.

**Setup.** Fixed MLP hidden_dim was increased from 64 to 150 (23,251 parameters) and Standard MoE hidden_dim from 64 to 74 (23,538 parameters), both matching Nomadic Full. The same three seeds (42, 123, 456) and identical training procedure were used. Nomadic NoPolicy and Nomadic Full were run concurrently as reference baselines.

**Table 9: Parameter-Matched Baseline Results (3-seed mean)**

| Model | Params | Seq MSE | ΔH | Stable Ent | Trans Ent |
|-------|-------:|:-------:|:--:|:----------:|:---------:|
| Fixed (h=64) | 4,417 | 0.409 | — | — | — |
| Fixed (h=150, matched) | 23,251 | 0.409 | — | — | — |
| Standard MoE (h=64) | 17,798 | 0.409 | 0.017 | 0.952 | 0.969 |
| Standard MoE (h=74, matched) | 23,538 | 0.410 | 0.027 | 0.961 | 0.988 |
| Nomadic NoPolicy | 22,926 | **0.214** | 0.375 | 0.548 | 0.923 |
| **Nomadic Full** | **23,053** | **0.165** | **0.781** | **0.091** | **0.872** |

**Key findings.** Increasing Fixed MLP capacity from 4,417 to 23,251 parameters yields no measurable improvement (Seq MSE 0.409 → 0.409). Increasing Standard MoE capacity from 17,798 to 23,538 parameters similarly produces no improvement (0.409 → 0.410) and raises ΔH only marginally from 0.017 to 0.027 — a value that remains 29× below Nomadic Full (0.781). Stable Entropy stays near 0.961 in both Standard MoE variants, compared to 0.091 in Nomadic Full.

The parameter-matched Standard MoE achieves 28× less entropy differentiation than Nomadic Full despite having slightly more parameters (23,538 vs. 23,053). This result rules out model capacity as a confound: the performance gap between Standard MoE and Nomadic Full is attributable to the temporal control mechanisms (Δx-based gating, dynamic dwell constraints, PolicyNet), not to the additional parameters they introduce. The transition from Standard MoE (matched) to Nomadic NoPolicy — both at comparable parameter budgets — yields a Seq MSE reduction of −0.196, driven entirely by the introduction of temporal structure. Capacity scaling produces no equivalent gain, confirming that *how* the model processes time matters more than how large it is.

### 4.10 Recurrent Gate + Oracle Baseline

The experiments in §4.1–4.9 establish that temporal structure improves routing performance, but leave open the question of whether *implicit* temporal learning (e.g., a recurrent gate without explicit Δx) achieves the same behavioral outcome as Nomadic's explicit transition dynamics control. We address this with a GRU-gated MoE baseline and a regime-specialist Oracle upper bound.

**Setup.** The GRU MoE replaces GateNet entirely with a GRUCell conditioned on batch-mean input x (no Δx). Hidden state h carries temporal context implicitly across sequential minibatches, reset at each epoch. To prevent sequence memorization, training sequences use shuffled regime ordering per cycle; Dropout(p=0.15) and 5× weight decay regularize the hidden state. GRU MoE has 26,502 parameters — more than Nomadic Full (23,053). The Oracle trains three regime-specialist MLPs independently (Expert_A on regime A data only, Expert_B on B, Expert_C on C), then routes each test batch to the expert corresponding to its dominant regime label. This represents the achievable upper bound given the expert capacity and label access.

**Table 10: Recurrent Gate + Oracle Results (3-seed mean)**

| Model | Params | Seq MSE | ΔH | Stable H | Trans H | Switch Lat |
|-------|-------:|:-------:|:--:|:--------:|:-------:|:----------:|
| Standard MoE | 17,798 | 0.415 | 0.033 | 0.979 | 1.012 | — |
| GRU MoE | 26,502 | **0.105** | 0.354 | 0.365 | 0.719 | 2.287 |
| **Nomadic Full** | **23,053** | 0.215 | **0.755** | **0.176** | **0.931** | **1.232** |
| Oracle | 3×4,417 | 0.330 | — | — | — | — |

**GRU MoE achieves lower Seq MSE than Nomadic Full (0.105 vs. 0.215).** This result has two interpretations. First, GRU genuinely learns temporal context: ΔH rises from Standard MoE's 0.033 to 0.354, confirming that recurrent hidden state provides useful temporal information. Second, GRU benefits structurally from seeing all batch types during training, including transition batches with linearly-interpolated labels, while the Oracle's regime-specialist experts are trained only on stable-phase data and are structurally disadvantaged on transition batches — explaining why Oracle MSE (0.330) exceeds Nomadic's (0.215) despite having label access. The GRU/Oracle MSE ratio (0.319) falling below 1.0 reflects this asymmetry rather than a violation of the upper bound.

**Nomadic Full achieves 2.14× higher ΔH than GRU (0.755 vs. 0.354) with 52% lower Stable Entropy (0.176 vs. 0.365).** GRU reduces stable-phase entropy from Standard MoE's 0.979 to 0.365 — a meaningful improvement — but does not reach the near-deterministic fixation of Nomadic Full (0.176). The transition-phase entropy tells a parallel story: GRU reaches 0.719 while Nomadic reaches 0.931, indicating that Nomadic's gate remains more exploratory during transitions.

**Switch latency favors Nomadic (1.232 vs. 2.287).** Despite GRU's implicit temporal integration, Nomadic's explicit Δx-based switching pressure responds to regime transitions 46% faster than GRU's hidden state-driven routing.

**Interpretation.** The GRU baseline establishes that implicit temporal learning is *sufficient for Seq MSE improvement* but *insufficient for homeomorphic fixation*. The distinction is not merely quantitative: GRU's partial entropy reduction (ΔH 0.354, Stable H 0.365) represents a model that has learned to hedge across experts during transitions but cannot commit to a single expert during stable phases. Nomadic's explicit dwell-time regularizer and PolicyNet create the pressure for stable-phase fixation that GRU's gradient-only learning does not achieve. This confirms the central design claim: the behavioral signature of structured adaptive fixation requires explicit transition dynamics control, not just temporal information.

Although GRU achieves lower Seq MSE, this result highlights a key distinction:implicit temporal encoding is sufficient for prediction, but insufficient for producing structured transition behavior.

GRU learns to interpolate across regimes, whereas Nomadic Routing enforces explicit transition dynamics, leading to faster switching, stronger fixation, and higher entropy differentiation. This suggests that predictive accuracy and transition structure are separable objectives, requiring different inductive biases.

This result highlights a fundamental trade-off: models optimized for prediction accuracy may implicitly smooth over transitions, whereas explicit transition control preserves structural interpretability of regime changes.

---

### 4.11 Real Non-Stationary Time Series

The synthetic experiments establish Nomadic Full's behavioral signature under controlled conditions. This section tests whether that signature transfers to real-world time series, and identifies the environmental conditions under which the Φ signal activates.

**Datasets.** We evaluate on two datasets using identical model architecture and hyperparameters as the synthetic experiments (input normalization adjusted per dataset):

- **ETTh1** (Electricity Transformer Temperature, hourly, 2016–2018): 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT). Regime labels derived from 7-day rolling std of OT, partitioned at the 33rd and 67th percentiles into Low / Mid / High volatility regimes. Average stable run length: ~85 steps.
- **Bitcoin** (BTC/USD daily close, 2013–2021): 4 features (log_return, vol5, vol30, momentum10). Regime labels derived from 30-day rolling volatility, same 3-partition scheme. Average stable run length: ~1 step.

**Prediction targets.** ETTh1 is evaluated under two horizons: 1-step-ahead (OT$_{t+1}$) and 24-step-ahead (OT$_{t+24}$). Bitcoin predicts log_return$_{t+1}$. All models use train-split normalization statistics; time ordering is preserved (no shuffle).

**Table 10: Real Time Series Results (3-seed mean ± std)**

| Dataset / Horizon | Model | Seq MSE | MSE std | ΔH | Stable H | Trans H | Switch Lat |
|---|---|---|---|---|---|---|---|
| ETTh1 1-step | Standard MoE | 0.008 | 0.001 | −0.004 | 1.090 | 1.086 | 2.397 |
| ETTh1 1-step | **Nomadic Full** | 0.008 | 0.001 | **+0.028** | **1.039** | 1.067 | 1.648 |
| ETTh1 24-step | Standard MoE | 0.147 | 0.016 | +0.001 | 0.943 | 0.943 | 2.385 |
| ETTh1 24-step | **Nomadic Full** | **0.126** | 0.011 | −0.009 | 1.066 | 1.057 | **1.543** |
| Bitcoin | Standard MoE | 1.019 | 0.028 | +0.045 | 0.888 | 0.932 | 1.000 |
| Bitcoin | **Nomadic Full** | 1.049 | 0.031 | +0.018 | 1.055 | 1.073 | 1.911 |

**Prediction pressure as activation condition.** The three conditions reveal a consistent pattern:

On **ETTh1 24-step**, prediction difficulty is sufficient to sustain nonzero Δx_err (MSE ≈ 0.13–0.17 throughout training). Nomadic Full achieves 13.9% lower Seq MSE than Standard MoE. ΔH is near-zero (−0.009) rather than the synthetic result (+0.755), indicating that homeomorphic fixation is not achieved — but the MSE gain is preserved, confirming that Φ-driven routing adjustment operates even without full stable-phase fixation.

On **ETTh1 1-step**, prediction error collapses to ≈0.007 by epoch 50 and remains near-zero throughout. Both Δx_env (continuous physical time series, small batch-to-batch shift) and Δx_err (near-zero) fail to generate routing pressure. Stable Entropy for both models saturates near log(3) ≈ 1.099 (near-uniform routing), and no MSE advantage is observed (Nomadic: 0.008 vs. Standard MoE: 0.008, −6.1% degradation within noise).

On **Bitcoin**, log-return prediction is near-random (MSE ≈ 1.0, consistent with market efficiency). Δx_err is persistently near its EMA baseline, and Δx_env is dominated by noise rather than structured distributional shift. The result mirrors ETTh1 1-step: Stable H remains high (1.055), no homeomorphic fixation, and no MSE advantage.

**Interpretation.** These results identify a prediction pressure boundary condition for Nomadic's Φ signal. When prediction error is either near-zero (easy task) or near-random (noise-dominated), neither Δx channel generates sufficient routing pressure to differentiate stable and transition phases. This is structurally identical to the gradual-transition failure mode in §4.8: when the signal cannot establish a stable-phase baseline, the system cannot detect departures from it.

The 24-step ETTh1 result occupies a middle regime: prediction error is large enough to activate Δx_err, and the physical system (transformer load) has sufficiently structured regime transitions (avg. stable run ~85 steps) to permit partial regime discrimination. The MSE gain under this condition suggests that the Φ mechanism contributes to routing quality even when full homeomorphic fixation is not achieved.

Across all three real-world conditions, no negative transfer is observed in the MSE-favorable cases, and the failure modes are consistent with those identified in synthetic task generalization (§4.8). These results identify the operational regime of the method rather than its limitations.

---

### 4.12 Component Ablation on ETTh1 (24-step)

To isolate which components drive the MSE improvement observed in §4.11, we perform a brick-removal ablation on ETTh1 24-step-ahead prediction: each component is removed independently from Nomadic Full, and three combinations are additionally tested. Experiments are repeated across three seeds and three ema_decay values (0.50, 0.80, 0.95) to assess sensitivity to the EMA time constant.

**Setup.** Ablation variants: (1) w/o PolicyNet — gate routing only, no meta-level stay/switch control; (2) w/o Δx_err — GateNet input reduced to [x, Δx_env], error channel removed; (3) w/o L_cons — consistency loss λ_cons set to zero; (4) w/o L_dwell — dwell-time regularizer disabled. Combination variants: w/o {PolicyNet, L_dwell}, w/o {Δx_err, L_cons}, w/o {PolicyNet, L_cons}.

**Table 11: ETTh1 Ablation Results (24-step, 3-seed mean ± std)**

| Variant | ema=0.50 MSE | ema=0.80 MSE | ema=0.95 MSE |
|---|---|---|---|
| Nomadic Full | 0.1081 ± 0.012 | 0.1166 ± 0.005 | 0.1123 ± 0.005 |
| w/o PolicyNet | 0.1061 ± 0.013 | 0.1271 ± 0.017 | 0.1179 ± 0.018 |
| w/o Δx_err | 0.1173 ± 0.008 | 0.1321 ± 0.015 | 0.1274 ± 0.006 |
| w/o L_cons | **0.1050 ± 0.012** | **0.1099 ± 0.005** | **0.1009 ± 0.007** |
| w/o L_dwell | 0.1081 ± 0.012 | 0.1166 ± 0.005 | 0.1123 ± 0.005 |
| w/o {PolicyNet, L_dwell} | — | 0.1271 ± 0.017 | — |
| w/o {Δx_err, L_cons} | — | 0.1476 ± 0.037 | — |
| w/o {PolicyNet, L_cons} | — | 0.1139 ± 0.004 | — |

**Finding 1: L_dwell is inactive on continuous time series.** Across all three ema_decay values, w/o L_dwell produces results identical to Nomadic Full (MSE and ΔH match to four decimal places). The DwellTimeRegularizer tracks expert switches at the batch level by comparing each batch's dominant expert to the previous batch's. In the synthetic setting, regime structure is imposed at the batch level, so dwell counts accumulate meaningfully. In ETTh1, the dominant expert fluctuates within regimes due to continuous variation in the input features, preventing the dwell counter from reaching τₖ. L_dwell is therefore a dead component in this setting — not harmful, but contributing nothing.

**Finding 2: Δx_err removal consistently degrades MSE.** Removing the error channel raises MSE across all three ema_decay values (+0.009 to +0.015), confirming that Δx_err carries routing-relevant signal even when full homeomorphic fixation is not achieved. The effect is largest at ema_decay=0.80, suggesting that intermediate EMA speeds best preserve the relative error signal that distinguishes genuine prediction difficulty from noise.

**Finding 3: L_cons is counterproductive in continuous time series.** Removing L_cons improves MSE in all three ema_decay conditions, with the largest gain at ema_decay=0.95 (0.1123 → 0.1009, −10.2%). In the synthetic setting, L_cons penalizes gate variance within a regime, which reinforces stable expert assignment when regimes have clear distributional boundaries. In ETTh1, however, intra-regime gate variation is structurally expected: even within a stable volatility regime, the physical system (transformer load) varies continuously, and natural gate movement is the appropriate response. L_cons suppresses this variation, over-regularizing the routing and degrading prediction quality.

**Finding 4: L_cons and Δx_err interact.** When both are removed simultaneously (w/o {Δx_err, L_cons}), MSE rises to 0.1476 — substantially worse than removing either alone (0.1321 and 0.1099 respectively). This superadditive degradation indicates that L_cons provides a partial stability function in the absence of the error channel: when Δx_err is unavailable, L_cons constrains erratic gate behavior that would otherwise go unpenalized. The two components are not independently substitutable.

**Finding 5: PolicyNet contribution is ema_decay-dependent.** At ema_decay=0.80, removing PolicyNet raises MSE by +0.0105. At ema_decay=0.50 (faster EMA, noisier signal), PolicyNet removal slightly lowers MSE (0.1081 → 0.1061), suggesting that under noisy Φ signals the policy introduces more variance than it removes. At ema_decay=0.95, PolicyNet removal raises MSE by +0.0056. The implication is that PolicyNet's contribution in real time series settings depends on signal quality — unlike the synthetic setting where its role in homeomorphic fixation is unconditional.

**Summary.** The ETTh1 ablation confirms that Δx_err is the most robust contributor across all conditions. PolicyNet and L_cons contribute conditionally, with their effectiveness depending on signal quality and intra-regime variation structure. L_dwell requires redesign for continuous time series settings. The hyperparameter values validated on synthetic data (λ_cons=0.03, τₖ_min=3) are not directly transferable to real environments — finding optimal values for each deployment context is deferred to future work.

The inactivity of L_dwell in continuous environments highlights that dwell-time constraints are most relevant in discretely structured regimes, and suggests that alternative formulations are needed for continuous domains.

---

## 5 Discussion

### 5.1 When does Nomadic Routing help?

Our experiments indicate that the effectiveness of Nomadic Routing is contingent on the structure of environmental dynamics. The framework is most beneficial when the input distribution exhibits intermediate levels of change—sufficient to signal meaningful transitions, yet stable enough to support consistent expert specialization. In contrast, environments characterized by continuous smooth drift or high-frequency volatility do not provide conditions under which stable fixation can emerge.

This observation suggests that the framework operates within a specific regime of change pressure, rather than providing uniform improvements across all settings. Understanding and characterizing this regime is therefore central to applying the method effectively.

---

### 5.2 Routing as a control problem

The results support a reinterpretation of expert routing as a control problem over transitions. Instead of focusing solely on selecting the best expert at each step, the framework emphasizes managing the timing and structure of transitions between experts.

Central to this reinterpretation is the structural separation between Δx and Φ. Δx functions as an environmental sensor — it detects that the world has changed. Φ functions as a decision variable — it determines whether that change is informative enough to warrant revising the current routing policy, approximating the free energy differential between staying and switching. This separation allows the system to distinguish between noise (high Δx, low Φ) and genuine regime transitions (high Δx, high Φ), a distinction that stateless gating cannot make. Dwell-time constraints then regulate commitment, enabling a balance between stability and adaptability.

This perspective differs from approaches that rely primarily on static gating or weight-based adaptation, and suggests that temporal structure — specifically, the explicit modeling of transition timing as a separate control objective — plays a meaningful role in routing behavior under non-stationary conditions.

---

### 5.3 Homeomorphic Fixation as Empirical Signature

The most striking result in Table 1 is the collapse of Stable Entropy to 0.108 in Nomadic Full. During stable phases, the gate becomes near-deterministic — converging to a single expert with high confidence. During transition phases, it reverts to high entropy (0.896), exploring across experts.

This differential entropy pattern is what we term *homeomorphic fixation*: the system's routing identity is preserved not through rigid assignment but through a consistent response law — fixate when stable, explore when changing. The transformation law (Δx → routing response) remains coherent across both phases, even as the specific routing assignment changes between regimes.

Standard MoE shows almost no entropy differentiation (ΔH = 0.033), confirming that this pattern is a product of the temporal control components, not of expert mixture alone. It also distinguishes Nomadic Routing from the GRU-gated baseline (§4.10): implicit recurrent learning achieves lower Seq MSE but produces ΔH of only 0.354, establishing that prediction improvement and homeomorphic fixation are separable objectives requiring different inductive biases.

**Synthetic environment as primary testbed.** The core ablation and robustness results are from controlled synthetic regression tasks with clean periodic transitions and Gaussian regime sampling. These do not reflect the noise and non-Markovian structure of real environments; no claim is made that current hyperparameters transfer to real-world settings without adjustment. The robustness experiment (§4.4) extends the setting to 4 regimes with random ordering and underprovisioned experts. §4.8 tests five task variants and identifies three structural failure modes: (1) gradual transitions (steps=24) raise the transition proportion of the sequence to ~50%, preventing Δx_env from establishing stable-phase fixation; (2) heavy-tail noise (Student-T, df=2) introduces episodic spikes in Δx_err that the error-EMA baseline cannot distinguish from genuine regime transitions, corrupting the Φ signal; (3) the combined condition (nonlinear + abrupt + heavy-tail) destabilizes PolicyNet training through extreme output scales and noise. Concrete improvement directions include a transition-ratio-aware Φ scaling for gradual settings, a robust estimator (e.g., Huber-loss-based EMA) for Δx_err under heavy-tail noise, and output normalization with gradient clipping for compound stress conditions.

**Prediction pressure boundary in real time series.** Real-world experiments (§4.11) reveal that the Φ signal requires sufficient prediction pressure to activate. On ETTh1 1-step-ahead prediction, MSE collapses to ≈0.007 within 50 epochs and both Δx channels remain near-zero thereafter, producing uniform routing and no MSE advantage. On Bitcoin, near-random prediction error (MSE ≈ 1.0) provides no stable EMA baseline for Δx_err to detect departures from. Tasks where the model converges too quickly or where error is structurally irreducible will not benefit from the Φ mechanism in its current form.

**PolicyNet training instability.** Per-seed variance in Stable Entropy (0.056–0.186) in the synthetic experiments indicates that convergence to hard fixation is not guaranteed under the heuristic teacher signal. In the LLM experiment, switch probability saturates near 1.0 for both stable and transition contexts after training, indicating that stay/switch discrimination remains incomplete. Replacing the heuristic teacher with a learned or RL-based objective is an open direction.

**Component transferability gap.** The ETTh1 ablation (§4.12) reveals that component contributions observed in the synthetic setting do not transfer uniformly to real environments. L_dwell is completely inactive on continuous time series because its batch-level expert-switch counter cannot accumulate to τₖ when dominant experts fluctuate within regimes. L_cons, beneficial in synthetic settings with clear inter-regime boundaries, becomes counterproductive on ETTh1 where intra-regime gate variation is structurally appropriate. These are not implementation failures — the components function as designed — but the design assumptions (batch-level regime structure, high intra-regime gate stability) do not hold in continuous physical time series. Adapting L_dwell to operate on a sliding window of individual time steps rather than batch-level switches, and making λ_cons adaptive to measured intra-regime variance, are concrete redesign directions deferred to future work.

**Hyperparameter non-transferability.** The loss weights (λ_cons=0.03, τₖ_min=3, ema_decay=0.80) validated on synthetic data are not directly transferable to real environments. The ETTh1 ablation shows that ema_decay meaningfully affects both MSE and component interactions: at ema_decay=0.50, PolicyNet removal slightly improves MSE, while at ema_decay=0.95 it degrades performance — the opposite pattern. The current framework provides a conceptual design with validated synthetic performance; finding optimal hyperparameter configurations for each deployment environment requires environment-specific tuning that this work does not address.

**Oracle upper bound is not tight.** The regime-specialist Oracle (§4.10) trains each expert on stable-phase data only and routes via dominant regime labels. Because transition batches carry linearly-interpolated labels that no single regime expert can optimally approximate, Oracle MSE (0.330) exceeds Nomadic Full's (0.215). A stronger Oracle using α-weighted expert mixtures for transition batches would provide a tighter upper bound and is deferred to future work.

**Φ design: theoretical gap.** Φ is operationalized as a composite approximation of the free energy differential between staying with the current routing policy and transitioning to a new one (§2.3). The comparative experiment (§4.7) provides post-hoc empirical justification: pure information-geometric variants (JSD, KL) fail to maintain switching pressure during stable phases because gate divergence vanishes when the system fixates, while the EMA composite sustains nonzero Φ through the task-aware $\text{gap}_t$ term. The conceptual motivation — treating Φ as a Bayesian cost-benefit comparison rather than a raw uncertainty measure — is laid out in §2.3; however, the formal derivation of this correspondence under a specified generative model has not been established. In particular, the precise conditions under which the four-term composite constitutes a valid lower bound on expected free energy reduction from a routing transition remain an open question. Formalizing this connection is deferred to future work.

---

### 5.4 Future Directions

Future work may explore extensions of the framework in several directions.

**Principled transition signal formulation.** The current Φ composite is empirically motivated (§4.7) and conceptually framed as a free energy differential (§2.3); formalizing this correspondence under a specified generative model remains an open theoretical problem. One concrete avenue is deriving Φ from a per-sample routing heterogeneity measure, connecting $\text{std}_i[\text{JSD}(g_i \| \bar{g}_t)]$ to expected free energy reduction from a policy revision, and establishing whether the four-term composite constitutes a valid lower bound on this quantity.

**RL-based policy learning.** The current PolicyNet is trained via a heuristic teacher signal. A natural extension is to replace this with a reinforcement learning objective that directly optimizes transition timing. One formulation treats routing as a three-term reward problem:

$$R_{\text{total}}(t) = \alpha \cdot R_{\text{sync}}(t) - \beta \cdot P_{\text{dogma}}(t) + \gamma \cdot R_{\text{nomad}}(t)$$

where $R_{\text{sync}}$ rewards low-latency response to Δx, $P_{\text{dogma}}$ penalizes prolonged structural rigidity (gate entropy near zero for excess dwell steps), and $R_{\text{nomad}}$ rewards high-entropy trajectories that successfully complete regime transitions. This formulation makes the current heuristic teacher targets derivable rather than hand-specified, and connects dwell-time control to temporal difference learning.

**Sequence-aware architectures.** Integrating Nomadic Routing with architectures that maintain explicit sequence state — such as state-space models or recurrent transformers — could enable richer temporal modeling than the current EMA-based tracker.

**Large-scale evaluation.** The LLM experiments in §4.5 are preliminary. Parameter-matched comparisons, ground-truth phase labeling, and evaluations across more model families and task benchmarks are needed to assess the scalability and generality of the signal transfer approach.

---

## 6 Conclusion

In this work, we reformulate expert routing in mixture-of-experts models as a control problem over transition dynamics, rather than a stateless selection mechanism. This shift in perspective allows routing to be understood not only in terms of which expert to select, but in terms of when to remain, when to switch, and how to structure transitions under non-stationary conditions.

Central to this reformulation is a structural separation between two signals. The hybrid change signal Δx measures whether the environment has shifted — it is an environmental sensor. The switching signal Φ determines whether that shift is informative enough to warrant a routing revision, approximating the free energy differential between maintaining and revising the current policy. This separation allows the system to distinguish noise from genuine regime transitions, enabling adaptive switching when warranted and stable fixation otherwise. A dynamic dwell-time constraint and PolicyNet complete the framework, regulating commitment and meta-level routing decisions over time.

Across synthetic and real-world experiments, we show that the effectiveness of this formulation depends on an intermediate regime of change pressure. Under these conditions, the model exhibits structured transition behavior characterized by near-deterministic fixation during stable phases and adaptive switching during transitions — a pattern we term *homeomorphic fixation*, in which routing identity is preserved as a consistent response law rather than a fixed assignment.

Our results further demonstrate that predictive accuracy and transition structure are separable objectives. While recurrent approaches such as GRU-based routing achieve lower prediction error through implicit temporal encoding, they do not produce the same degree of structured transition behavior. Explicit control over transition dynamics — through the Δx/Φ separation, dwell constraints, and policy-level routing — introduces a distinct inductive bias that implicit temporal learning does not replicate.

The empirical findings identify the operational regime of the framework. In environments with insufficient or excessive change — continuous drift, high-frequency volatility, or noise-dominated signals — the free energy differential collapses, and the benefits of explicit control diminish. These results delineate the conditions under which transition-aware routing is meaningful, rather than representing failures of the framework.

Finally, this work opens several directions for future research. Formalizing the relationship between the Φ composite and a proper Bayesian posterior update, extending dwell-time constraints to continuous domains, and integrating transition-aware routing into large-scale architectures remain open problems. More broadly, viewing routing as a control problem over free energy differentials suggests a shift in how adaptive systems are designed: from optimizing static selection rules toward regulating the dynamics of change itself.

---

## Acknowledgment

Code implementation and manuscript drafting were assisted by AI-based tools. All conceptual design, experimental decisions, and theoretical framing were performed by the author.

---

## References

1. Adams, R. P., & MacKay, D. J. (2007). *Bayesian online changepoint detection.* arXiv:0710.3742.

2. Bacon, P. L., Harb, J., & Precup, D. (2017). *The option-critic architecture.* AAAI.

3. Charnov, E. L. (1976). *Optimal foraging, the marginal value theorem.* Theoretical Population Biology, 9(2), 129–136.

4. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127–138.

5. Kirkpatrick, J., Pascanu, R., et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS, 114(13), 3521–3526.

6. Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal, 27(3), 379–423.

7. Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., ... & El Sayed, W. (2024). *Mixtral of Experts.* arXiv:2401.04088.

8. Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., ... & Laudon, J. (2022). *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022.

9. Zenke, F., Poole, B., & Ganguli, S. (2017). *Continual Learning Through Synaptic Intelligence.* ICML 2017.

10. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

11. Roller, S., Dinan, E., Goyal, N., Ju, D., Williamson, M., Liu, Y., ... & Weston, J. (2021). *Hash Layers for Large Sparse Models.* arXiv:2106.04426.

12. Dohare, S., Hernandez-Garcia, A., Lan, Q., Parashar, U., Mahmood, A. R., & Sutton, R. S. (2024). *Maintaining plasticity in deep continual learning.* Nature, 632, 314–320.

13. Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* Journal of Machine Learning Research, 23(120), 1–39.

14. Lewandowski, A., Tanaka, H., Botvinick, M., & Stachenfeld, K. (2023). *Directions of curvature as an explanation for loss of plasticity.* arXiv:2312.00246.

15. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.

16. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.* arXiv:1701.06538.

17. Bengio, Y., Courville, A., & Vincent, P. (2013). *Representation Learning: A Review and New Perspectives.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1798–1828.

18. Du, N., Huang, Y., Dai, A. M., Simon, S., Lepikhin, D., ... & Zhou, Y. (2022). *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.* ICML 2022.

19. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Kirkpatrick, J., Hazen, A., ... & Hadsell, R. (2016). *Progressive Neural Networks.* arXiv:1606.04671.

20. Deleuze, G., & Guattari, F. (1987). *A Thousand Plateaus: Capitalism and Schizophrenia.* University of Minnesota Press.

21. Tishby, N., Pereira, F. C., & Bialek, W. (2000). *The information bottleneck method.* arXiv:physics/0004057.

22. Lopez-Paz, D., & Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning.* NeurIPS 2017.

23. Komatsuzaki, A., Joan, P., Dai, A. M., ... & Gontijo-Lopes, R. (2022). *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR 2023.

24. Zuo, S., Liu, X., Jiao, J., ... & Gao, J. (2021). *Taming Sparsely Activated Transformer with Stochastic Expert Selection.* ICLR 2022.

25. Isele, D., & Cosgun, A. (2018). *Selective Experience Replay for Lifelong Learning.* AAAI 2018.

26. Bengio, Y., Deleu, T., Rahaman, N., Ke, R., Lachapelle, S., ... & Golemo, F. (2019). *A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms.* ICLR 2020.

27. Kullback, S., & Leibler, R. A. (1951). *On Information and Sufficiency.* Annals of Mathematical Statistics, 22(1), 79–86.

28. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention is All You Need.* NeurIPS 2017.

29. Schmidhuber, J. (1991). *A Possibility for Implementing Curiosity and Objective Self-Criticism in a Model-Based Robot.* Simulation of Adaptive Behavior.

30. Sutton, R. S., Koop, A., & Silver, D. (2007). *On the Role of Tracking in Stationary and Non-stationary Environments.* ICML 2007 Workshop.
