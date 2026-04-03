# Experiment Setup

## 1. Environment

We evaluate Nomadic Intelligence in a synthetic non-stationary regression task with three regimes:

* **Regime A**: y = x₁ + x₂
* **Regime B**: y = x₁ - x₂
* **Regime C**: y = -x₁ + 0.5x₂

The environment transitions continuously between regimes, creating a phase-shift setting where static models are insufficient.

---

## 2. Model Variants

We evaluate the following configurations:

* **Fixed Baseline**

  * Single MLP with equivalent parameter count

* **Nomadic (Base)**

  * MoE with Δx-based gating

* **+ Load Balancing**

  * Prevent expert collapse

* **+ τₖ Lower Bound**

  * Strategic dwell time constraint

* **+ φ (Uncertainty Signal)**

  * Transition-aware switching

* **+ PolicyNet**

  * Explicit switching control (soft/hard decisions)

---

## 3. Metrics

We use the following evaluation metrics:

### Primary Metric

* **Sequence Test MSE**

  * Measures performance in non-stationary sequence

### Secondary Metrics

* Static Test MSE (reference only)
* Switch Latency
* Mean Dwell Time
* Gate Entropy:

  * Stable phase entropy
  * Transition phase entropy

---

## 4. Training Details

* Epochs: 220
* Device: CUDA (GTX 1660 Super)
* Seeds: 42, 123, 456

---

## 5. How to Run

```bash
python run_structured.py --config config.yaml --seed 42 --save_dir outputs_seed42
```

---

## 6. Key Observation

* Nomadic model consistently outperforms fixed baseline in sequence MSE
* Transition entropy > stable entropy across all runs
* Expert specialization emerges without supervision
