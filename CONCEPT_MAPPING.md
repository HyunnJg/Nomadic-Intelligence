# Concept to Implementation Mapping

This document connects theoretical concepts to actual code components.

---

## Core Mapping

| Concept              | Implementation                 |
| -------------------- | ------------------------------ |
| Δx (change signal)   | input shift + prediction error |
| φ (uncertainty)      | beta_phi scaling               |
| Dwell time (τₖ)      | tau_k_min, tau_k_penalty       |
| Anti-dogmatism       | entropy regularization         |
| Expert diversity     | load balancing loss            |
| Switching policy     | PolicyNet                      |
| Hard/soft transition | STE + temperature              |

---

## Interpretation

### 1. Δx

* Captures environmental and internal mismatch
* Drives switching trigger

---

### 2. φ

* Controls switching intensity
* High φ → exploration
* Low φ → stability

---

### 3. τₖ (Dwell Time)

* Prevents:

  * rapid noise switching
  * rigid fixation

---

### 4. PolicyNet

* Explicit decision layer
* Converts implicit gating → policy control

---

## Key Insight

> The system is not selecting experts.
> It is controlling *how transitions happen*.

---

## One-line Summary

Nomadic Intelligence transforms:

* from **selection problem**
* to **transition dynamics control problem**
