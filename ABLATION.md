# Ablation Study

This section isolates the contribution of each component in the Nomadic Intelligence architecture.

---

## 1. Base Model

* Δx-based gating only
* Result:

  * Sequence performance improves
  * Expert collapse observed

👉 Interpretation:
Switching exists, but not structured.

---

## 2. + Load Balancing

* Prevents expert dominance

Result:

* Expert usage becomes distributed
* Regime separation improves

👉 Interpretation:
Fixes spatial collapse, but not temporal behavior.

---

## 3. + τₖ Lower Bound (Dwell Control)

* Encourages minimum residence time per expert

Result:

* Switch latency stabilizes
* Collapse across seeds resolved

👉 Interpretation:
Temporal dynamics become controllable.

---

## 4. + φ (Uncertainty-driven Switching)

* Switching driven by Δx + error dynamics

Result:

* Transition entropy increases
* Sequence MSE improves significantly

👉 Interpretation:
Switching becomes context-aware rather than random.

---

## 5. + PolicyNet

* Explicit decision:

  * stay vs switch
  * soft vs hard routing

Result:

* Switching behavior becomes interpretable
* Control over transition dynamics increases

👉 Interpretation:
Routing → policy-driven transition

---

## 6. Summary

| Component      | Role                        |
| -------------- | --------------------------- |
| Load Balancing | Prevent expert collapse     |
| τₖ Lower Bound | Stabilize dwell time        |
| φ              | Enable meaningful switching |
| PolicyNet      | Control switching behavior  |

---

## Key Conclusion

> Performance gain is not from a single component,
> but from the interaction between uncertainty (φ), dwell control (τₖ), and structured routing.
