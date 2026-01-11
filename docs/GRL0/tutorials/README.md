# GRL Tutorial Chapters

**Format**: Tutorial paper chapters  
**Audience**: ML practitioners with basic RL knowledge  
**Style**: Narrative, educational, self-contained

---

## Overview

These chapters form the core of the GRL tutorial paper. Each chapter builds on previous ones while remaining accessible for selective reading.

---

## Chapter Index

### Part I: Foundations

| # | Title | Key Concepts | Status |
|---|-------|--------------|--------|
| 00 | [Overview](00-overview.md) | What is GRL, motivation | ✅ Complete |
| 01 | [Core Concepts](01-core-concepts.md) | Augmented space, parametric actions | ✅ Complete |
| 02 | RKHS Foundations | Kernels, functional spaces | ⏳ Planned |
| 03 | Energy Landscapes | Fitness vs energy, sign conventions | ⏳ Planned |

### Part II: Reinforcement Field

| # | Title | Key Concepts | Status |
|---|-------|--------------|--------|
| 04 | Reinforcement Field | Functional field, gradient | ⏳ Planned |
| 05 | Particle Memory | Experience representation | ⏳ Planned |

### Part III: Algorithms

| # | Title | Key Concepts | Status |
|---|-------|--------------|--------|
| 06 | MemoryUpdate | Belief transition, Algorithm 1 | ⏳ Planned |
| 07 | RF-SARSA | Two-layer TD, Algorithm 2 | ⏳ Planned |

### Part IV: Interpretation

| # | Title | Key Concepts | Status |
|---|-------|--------------|--------|
| 08 | Soft State Transitions | Emergent uncertainty | ⏳ Planned |
| 09 | POMDP Interpretation | Belief-based view | ⏳ Planned |
| 10 | Complete System | Putting it together | ⏳ Planned |

---

## Reading Recommendations

### New to GRL

Start with Chapter 00 (Overview), then proceed sequentially.

### Familiar with RL Theory

Skim Chapter 00, focus on Chapters 02-04 for mathematical foundations.

### Want to Implement

Read Chapter 00, skim 01-05, focus on 06-07, then see `../implementation/`.

---

## Chapter Template

Each chapter follows this structure:

1. **Introduction**: Why this topic matters
2. **Main Content**: Narrative explanation with examples
3. **Key Takeaways**: Summary points
4. **Next Steps**: Connection to following chapters

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $s$ | Environment state |
| $\theta$ | Action parameters |
| $z = (s, \theta)$ | Augmented state-action point |
| $k(\cdot, \cdot)$ | Kernel function |
| $\mathcal{H}_k$ | RKHS induced by kernel $k$ |
| $Q^+(z)$ | Field value at $z$ |
| $E(z)$ | Energy at $z$, equals $-Q^+(z)$ |
| $\Omega$ | Particle memory |

---

**Last Updated**: January 11, 2026
