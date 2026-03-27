# Contributing to Nomadic Intelligence

Thank you for your interest in this project.

Nomadic Intelligence sits at the intersection of philosophy, mathematics, and AI architecture. This means contributors can come from very different backgrounds — and all of them are welcome. You do not need to be an expert in all three areas to contribute meaningfully.

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

**1. Determining $\tau_k$ (Dwell Time)**
How should the system decide how long to stay in an attractor? Options include fixed thresholds, learned parameters, or dynamic adjustment based on $\Delta x$ variance. No solution exists yet.

**2. Preventing the Policy Engine from Becoming a Fixed Attractor**
The rule that selects attractors (the `_select_attractor` function) is itself a fixed structure. This is a known self-referential problem. If the selection policy rigidifies, the system becomes dogmatic at the meta-level.

**3. Attractor Boundaries in Continuous State Spaces**
The toy model uses discrete attractors with hard boundaries. In real high-dimensional environments, attractor boundaries are continuous and fuzzy. How do we define and detect them?

**4. Formal Verification of Homeomorphic Identity**
The claim $\mathcal{I}(t) \cong \mathcal{I}(t+1)$ needs a verifiable criterion. What measurable property during training would confirm that the Will to Resonance ($\Phi$) is being preserved?

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
