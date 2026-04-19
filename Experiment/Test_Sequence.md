# Test Sequence: Experimental Evolution of Nomadic Intelligence

This document outlines the chronological sequence of experiments conducted to validate the Nomadic Intelligence framework. The testing strategy follows a progression from controlled synthetic environments (to isolate behavioral signatures) to extreme stress tests, and finally to real-world physical and autoregressive language modeling tasks.

---

## Phase 1: Proof of Concept (Synthetic Baseline)
*The objective of this phase was to establish whether transition dynamics control could outperform stateless selection in a controlled non-stationary environment.*

* **Stage 1.1: Core Component Ablation (§4.1)**
  * **Environment:** 3-Regime Continuous Regression (Gaussian noise, linear transitions).
  * **Test:** Incremental addition of $\Delta x$ gating, dynamic $\tau_k$, and PolicyNet against a Standard MoE baseline.
  * **Key Finding:** Nomadic Full achieved a 60% Seq MSE reduction (0.410 → 0.162) and established the "Homeomorphic Fixation" signature ($\Delta H = +0.788$).
* **Stage 1.2: $\beta_\phi$ Parameter Sensitivity (§4.2)**
  * **Test:** Sweeping the switching pressure weight ($\beta_\phi$) from 0.00 to 0.10.
  * **Key Finding:** $\beta_\phi = 0.02$ with PolicyNet provides the optimal balance of Seq MSE (0.152) and cross-seed stability, confirming $\Phi$ acts as a stabilizing force.

## Phase 2: Stress Testing & Architectural Baselines
*The objective was to break the model and rule out confounding variables such as model capacity or implicit recurrent learning.*

* **Stage 2.1: Robustness / Underprovisioning (§4.4)**
  * **Test:** 4 Regimes, 3 Experts, Random regime ordering.
  * **Key Finding:** The system successfully demonstrated emergent "Expert Sharing," maintaining the ablation ordering (Nomadic Full Seq MSE 0.334) and preserving $\Delta H$ (+0.437).
* **Stage 2.2: Task Generalization Boundaries (§4.9)**
  * **Test:** Nonlinear functions, Abrupt (steps=2) vs. Gradual (steps=24) transitions, Heavy-tail noise (Student-T df=2), and Combined stress.
  * **Key Finding:** The mechanism is robust to nonlinear and abrupt conditions. Identified critical failure modes under gradual transitions (where the $\Delta x$ baseline fails to fixate) and heavy-tail noise (where noise mimics transitions).
* **Stage 2.3: Parameter-Matched & Oracle Baselines (§4.8, §4.10)**
  * **Test:** Scaling Standard MoE to match Nomadic Full's parameters (~23.5K), and testing against a GRU-gated MoE.
  * **Key Finding:** Capacity increase yielded no Seq MSE improvement (0.410 → 0.410). GRU MoE achieved lower Seq MSE but failed to achieve true Homeomorphic Fixation (Stable H = 0.365 vs Nomadic's 0.176), proving *explicit* transition control is required for structured behavioral identity.

## Phase 3: Real-World Time Series (Physical vs. Financial)
*The objective was to test the axiomatic assumptions against the noise and inertia of real-world datasets.*

* **Stage 3.1: The Noise Limit (Bitcoin)**
  * **Test:** BTC/USD daily close return prediction.
  * **Key Finding:** Prediction error remains near-random. The $\Phi$ signal fails to activate (Stable H = 1.055), confirming that extreme noise without structural regime shifts breaks the transition tracking mechanism.
* **Stage 3.2: The Inertia Limit (ETTh1 1-step ahead)**
  * **Test:** Electricity Transformer Temperature (OT) $t+1$ prediction.
  * **Key Finding:** Prediction error collapses to ≈0.007 due to physical inertia. The lack of "prediction pressure" prevents the $\Phi$ signal from differentiating stable/transition phases.
* **Stage 3.3: The Real-World Victory (ETTh1 24-step ahead) (§4.11)**
  * **Test:** ETTh1 $OT_{t+24}$ prediction.
  * **Key Finding:** With adequate prediction pressure and structured physical regimes, Nomadic Full achieved a **13.9% lower Seq MSE (0.126 vs 0.147)** than Standard MoE, proving the asymmetric advantage of temporal control in real environments.

## Phase 4: Autoregressive LLM Transfer
*The objective was to verify if the temporal signal layers could scale to large language models without architectural modification.*

* **Stage 4.1: Gemma-4-E2B Signal Transplant (§4.5)**
  * **Test:** Extracting $\Delta x$ from hidden states/uncertainty and routing 3 specialized LoRA adapters.
  * **Key Finding:** After PolicyNet training, the LLM exhibited a $\Delta H$ signature of **+0.984** (exceeding synthetic results), maintaining structured generation and a 54.2% switch rate across creative and stable prompts.

## Phase 5: [Next/Current] Continual Learning & Knowledge Preservation
*The objective is to prove that Nomadic Intelligence not only adapts to new terrains but preserves knowledge of past terrains (defense against catastrophic forgetting).*

* **Stage 5.1: Backward Transfer Validation (Planned)**
  * **Test:** Sequential training on isolated regimes (Regime A → B → C) followed by retroactive performance evaluation on prior regimes.
  * **Hypothesis:** Homeomorphic Fixation naturally isolates regime-specific gradients, granting Nomadic Intelligence inherent resilience against catastrophic forgetting compared to static MoE or dense networks.