# Visualization Guide

This section describes recommended plots for analyzing Nomadic Intelligence behavior.

---

## 1. Expert Switching Timeline

**Plot:**

* X-axis: time step
* Y-axis: selected expert (Top-1)

**Purpose:**

* Visualize switching pattern
* Detect:

  * collapse (single expert)
  * chaotic switching
  * structured transitions

---

## 2. Entropy Over Time

**Plot:**

* X-axis: time
* Y-axis: gate entropy

**Compare:**

* Stable phase vs transition phase

**Expected:**

* Transition entropy > stable entropy

---

## 3. Dwell Time Distribution

**Plot:**

* Histogram of dwell durations

**Purpose:**

* Verify τₖ behavior:

  * too short → noise
  * too long → fixation

---

## 4. Optional: φ vs Switching

**Plot:**

* X-axis: φ
* Y-axis: switching probability

**Purpose:**

* Validate φ as control signal

---

## Key Insight

Visualization is critical to demonstrate:

> Nomadic Intelligence is not random switching,
> but structured, context-aware transition.
