# GRL Documentation

**Generalized Reinforcement Learning (GRL)** is a framework that treats actions as parametric operators on state space, enabling continuous action generation, smooth generalization, and physically interpretable policies.

This directory contains comprehensive documentation including tutorial papers, theoretical foundations, quantum-inspired extensions, and implementation guides.

## Original Publication

**Chiu, P.-H., & Huber, M. (2022).** *Generalized Reinforcement Learning: Experience Particles, Action Operator, Reinforcement Field, Memory Association, and Decision Concepts.* arXiv:2208.04822.

**[Read on arXiv →](https://arxiv.org/abs/2208.04822)** (37 pages, 15 figures)

---

## Contents

### Tutorial Papers: Reinforcement Fields

**[GRL0/](GRL0/)** — Two-part tutorial series based on the original paper

**Part I: Particle-Based Learning** (8/10 chapters complete)
- Augmented state-action space
- Particle memory as belief state
- MemoryUpdate algorithm
- RKHS foundations and Riesz representer
- RF-SARSA (next)

**Part II: Emergent Structure & Spectral Abstraction** (Planned)
- Functional clustering in RKHS
- Spectral concept discovery
- Hierarchical policy organization

**Quantum-Inspired Extensions** (9 chapters complete)
- RKHS-quantum mechanics connections
- Complex-valued RKHS and amplitude-based learning
- Concept subspaces and projections
- Learning mechanisms beyond GP
- Principled memory dynamics (formation, consolidation, retrieval)

**[Start Learning →](GRL0/tutorials/00-overview.md)** | **[Research Roadmap →](ROADMAP.md)**

---

### Extensions: Actions as Operators (In Development)

Future documentation for operator-based extensions (Papers A, B, C)

**Current Status:**

- **Paper A** (Theory): ~70% complete — draft finished, figures in progress
- **Paper B** (Algorithms): Planned for Q2 2026
- **Paper C** (Applications): Planned for Q3 2026

**See:** `dev/papers/` for current drafts (not yet public)

**Future Public Documentation:**

#### Theory (from Paper A)
- Operator manifolds and parametric actions
- Operator algebra (composition, Lie groups)
- Generalized Bellman equation
- Energy regularization (least-action principle)

#### Algorithms (from Paper B)
- Operator-Actor-Critic (OAC)
- Neural operator architectures
- Training stability techniques
- Benchmark results

#### Applications (from Paper C)
- Physics-based control
- Robotic manipulation
- Compositional behaviors
- Transfer learning

---

## Documentation Organization

**Public documentation** (this directory):

- Tutorial papers, guides, API references
- Quantum-inspired extensions
- Intended for sharing with the research community

**Private development notes** (`dev/`):

- Research drafts, brainstorming, work-in-progress
- Paper drafts (A, B, C)
- Not intended for public sharing

**Research planning**:

- **[ROADMAP.md](ROADMAP.md)** — Comprehensive research plan and timeline
- Tracks GRL v0, quantum-inspired extensions, Papers A/B/C, and future directions

---

**Last Updated**: January 14, 2026
