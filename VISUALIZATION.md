# Visualization Guide

This section describes recommended plots for analyzing Nomadic Intelligence behavior.

---

## 1. Expert Switching Timeline

**Plot:**

import matplotlib.pyplot as plt

def plot_expert_timeline(expert_indices):
    plt.figure()

    plt.plot(expert_indices)
    plt.xlabel("Time step")
    plt.ylabel("Expert Index")
    plt.title("Expert Switching Timeline")

    plt.tight_layout()
    plt.savefig("figure_expert_timeline.png")
    plt.close()

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

def plot_entropy(entropy_values):
    plt.figure()

    plt.plot(entropy_values)
    plt.xlabel("Time step")
    plt.ylabel("Entropy")
    plt.title("Gate Entropy Over Time")

    plt.tight_layout()
    plt.savefig("figure_entropy.png")
    plt.close()

* X-axis: time
* Y-axis: gate entropy

**Compare:**

* Stable phase vs transition phase

**Expected:**

* Transition entropy > stable entropy

---

## 3. Dwell Time Distribution

**Plot:**

def plot_dwell_histogram(dwell_times):
    plt.figure()

    plt.hist(dwell_times, bins=30)
    plt.xlabel("Dwell Time")
    plt.ylabel("Frequency")
    plt.title("Dwell Time Distribution")

    plt.tight_layout()
    plt.savefig("figure_dwell.png")
    plt.close()

* Histogram of dwell durations

**Purpose:**

* Verify τₖ behavior:

  * too short → noise
  * too long → fixation

---

## 4. Optional: φ vs Switching

**Plot:**

def plot_phi_vs_switch(phi_values, switch_flags):
    plt.figure()

    plt.scatter(phi_values, switch_flags)
    plt.xlabel("Phi")
    plt.ylabel("Switch (0/1)")
    plt.title("Phi vs Switching Behavior")

    plt.tight_layout()
    plt.savefig("figure_phi_switch.png")
    plt.close()

* X-axis: φ
* Y-axis: switching probability

**Purpose:**

* Validate φ as control signal

---

## Key Insight

Visualization is critical to demonstrate:

> Nomadic Intelligence is not random switching,
> but structured, context-aware transition.
