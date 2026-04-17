# Contributing to Nomadic Intelligence

Thank you for your interest in this project.

Nomadic Intelligence sits at the intersection of philosophy, mathematics, and AI architecture. This means contributors can come from very different backgrounds — and all of them are welcome. You do not need to be an expert in all three areas to contribute meaningfully.

---

## ⚡ Why Now?

This project is at an early and unusually open stage — which means your contribution has an outsized chance of shaping its direction.

The prototype not only works in synthetic environments, but it has now proven its asymmetric advantage in real-world physical systems. On the ETTh1 24-step ahead prediction task, the Nomadic model achieves a **13.9% lower Seq MSE** than a parameter-matched Standard MoE. Furthermore, its core signal layer has been successfully transplanted onto Gemma-4-E2B, demonstrating a $\Delta H$ signature of +0.984 during autoregressive generation.

That's the proof of concept. What's missing is everything that comes after:
- The math isn't fully formalized
- The architecture isn't optimized
- The benchmark environment doesn't exist yet
- The philosophical framework hasn't been stress-tested by outside readers

**The problems are real, the baseline is working, and the direction is open.** If that combination sounds interesting to you, keep reading.

---

## 🧭 Who Can Contribute

This project actively welcomes three kinds of contributors:

**Philosophers & Theorists**
- Critique or extend the axiom system in `Theory_and_Axioms.md`
- Identify tensions with existing philosophical traditions (Deleuze, Friston, Buddhist ontology, etc.)
- Propose new conceptual frameworks that challenge or sharpen the core claims

**Engineers & Researchers**
- Implement Nomadic Intelligence in a real RL environment (see Milestone 1)
- Design a PyTorch architecture for attractor-switching (see Milestone 2)
- Formalize the mathematical boundaries of $\tau_k$ (see Milestone 3)
- Identify gaps between the formal theory and the toy model

**Both / Neither**
- Open issues to ask questions, flag contradictions, or propose new directions
- Improve documentation, translations, or examples
- Run the toy model and report unexpected behavior

---

## 🔀 How to Contribute

### Opening an Issue

Issues are the primary space for discussion. Use them freely for:

- Questions about the theory or implementation
- Identified contradictions or weaknesses in the axiom system
- Proposals for new features, attractors, or reward structures
- Bug reports in `nomadic_toy_model.py`

There is no strict template. A clear description of what you observed or what you're proposing is enough.

### Submitting a Pull Request

1. Fork the repository
2. Create a branch with a descriptive name (e.g., `add-gymnasium-env`, `fix-dwell-time-logic`, `extend-related-work`)
3. Make your changes
4. Open a PR with a brief explanation of what you changed and why

For significant changes to the theory or axioms, please open an Issue first to discuss before writing code or extensive documentation.

---

## 🧩 Current Open Problems

These are the highest-priority areas where contributions would have the most impact. They are listed in `README.md` as Open Questions, reproduced here with more context:

**1. Formalizing $\\tau_k$ (Dwell Time)**
The prototype shows measurable dwell time behavior — the switch latency distribution peaks at 1–2 batches, with occasional longer stays. But right now this is implicit and uncontrolled. How do we make it explicit and learnable? Options include fixed thresholds, learned parameters, or dynamic adjustment based on $\\Delta x$ variance. This is the most tractable entry point for an ML engineer.

**2. Stabilizing the $\\Delta x$ Signal**
In the current prototype, `delta_hybrid_raw` grows unbounded during training (reaching ~30 by epoch 200). A `tanh` squash is containing it, but this is a patch, not a solution. A principled distributional distance measure — KL divergence or Wasserstein distance between consecutive batches — would fix this properly. This is a well-scoped engineering problem.

**3. Preventing the Policy Engine from Becoming a Fixed Attractor**
The gate that selects attractors is itself a fixed structure — a known self-referential problem. If the selection policy rigidifies, the system becomes dogmatic at the meta-level. This is philosophically the most interesting problem in the project. No one has solved it yet.

**4. Attractor Boundaries in Continuous State Spaces**
The prototype uses soft MoE routing as a proxy for attractor boundaries. In real high-dimensional environments, boundaries are continuous and fuzzy. Formally defining when a "Separatrix Collapse" has occurred is an open mathematical problem.

**5. Formal Verification of Homeomorphic Identity**
The claim $\\mathcal{I}(t) \\cong \\mathcal{I}(t+1)$ needs a verifiable criterion. What measurable property during training would confirm that the Will to Resonance ($\\Phi$) is being preserved? This bridges the philosophy and the engineering — and no one has proposed a concrete answer yet.

**6. Overcoming the Prediction Pressure Boundary**
Experiments show that the $\Phi$ signal requires a minimum threshold of prediction error to activate. On tasks that are too easy (1-step physical inertia) or purely random (Bitcoin), the signal flattens. How can we formulate a $\Delta x^{err}$ tracker that remains sensitive even in extremely high or extremely low signal-to-noise ratio environments?

---

## 📐 Design Principles for Contributors

When proposing changes, keep these principles in mind:

**Anti-dogmatism applies to the framework itself.**
If a proposed change makes the system converge to a single fixed strategy, it contradicts the core thesis. The architecture should remain structurally open.

**$\Delta x$ is energy, not error.**
Implementations that treat environmental difference as something to minimize are philosophically misaligned with this project. If you want to explore that direction, Friston's Active Inference is the right framework.

**Coherence over chaos.**
Nomadism is not randomness. A contribution that increases structural mobility without maintaining $\tau_k$-based coherence will produce dissolution, not intelligence.

**Theory and code should stay synchronized.**
If you change the toy model's logic, update the relevant section in `Theory_and_Axioms.md`. If you propose a new axiom, consider whether it has an implementable analogue.

---

## 💬 Tone & Discussion

This project emerged from philosophical dialogue, not engineering specification. Disagreement is welcome. Pushback on the core axioms is welcome. The only thing that isn't welcome is closing the system down — turning an open question into a fixed answer without sufficient argument.

If you're unsure whether your contribution fits, open an Issue and ask. There are no bad questions here.

---

## 🟢 Where to Start

Not sure where to begin? Here's a map based on experience level — and a concrete first path that works for anyone.

### Suggested First Steps (for everyone)

1. Run `nomadic_multi_regime_structured.py`
2. Look at the output plots — entropy, expert usage, switch latency
3. Pick one behavior that looks unexpected or interesting
4. Try to explain it, reproduce it differently, or modify it
5. Open an Issue describing what you found

That's a valid contribution. You don't need to solve anything.

---

### By Experience Level

**Beginner — no ML background required**
- 📝 Improve README or documentation clarity
- 💬 Add comments to code explaining what a function does
- 🌍 Translate `Philosophy_En.md` into another language
- 📊 Reproduce the prototype results and document any differences
- ❓ Open an Issue with a question the docs don't answer — if you're confused, others will be too

**Intermediate — some ML or math background**
- 🔧 Implement a simple $\\tau_k$ heuristic: "switch attractor if $\\Delta x > \\theta$ for $N$ consecutive batches" and measure its effect on Seq MSE
- 📉 Replace `delta_hybrid_raw` with a KL divergence estimate between consecutive batch distributions and compare stability
- 📐 Design a quantitative regime–expert alignment score (mutual information, conditional entropy)
- 🏋️ Run the prototype on a different non-stationary dataset and check whether attractor specialization still emerges
- 🎨 Visualize hidden states (PCA / t-SNE) to see if attractors are geometrically separable

**Advanced — ML research or formal theory background**
- 🔬 Propose a new expert routing architecture that prevents hub collapse
- ⚙️ Redesign the $\\Delta x$ signal using Wasserstein distance or MMD
- 🧮 Formalize $\\tau_k$ as a learnable parameter and connect it to Option-Critic
- 📖 Write a formal critique of the Friston comparison — where the analogy holds and where it breaks
- 🌐 Propose a philosophical tradition not yet in Related Work that either supports or challenges the framework
- 🔗 Propose a measurable criterion for Homeomorphic Identity that can be verified during training

---

If you're still unsure, open an Issue with:
> "I want to contribute but don't know where to begin."

We'll guide you.