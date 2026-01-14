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
| 02 | [RKHS Foundations](02-rkhs-foundations.md) | Kernels, inner products, function spaces | ✅ Complete |
| 03 | [Energy and Fitness](03-energy-and-fitness.md) | Sign conventions, EBM connection | ✅ Complete |

### Part II: Reinforcement Field

| # | Title | Key Concepts | Status |
|---|-------|--------------|--------|
| 04 | [Reinforcement Field](04-reinforcement-field.md) | Functional field, RKHS gradient | ✅ Complete |
| 04a | [Riesz Representer](04a-riesz-representer.md) (supplement) | Gradients in function space, examples | ✅ Complete |
| 05 | [Particle Memory](05-particle-memory.md) | Particles as basis, memory as belief | ✅ Complete |

### Part III: Algorithms

| # | Title | Key Concepts | Status |
|---|-------|--------------|--------|
| 06 | [MemoryUpdate](06-memory-update.md) | Belief transition, Algorithm 1, particle evolution | ✅ Complete |
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

Start with Chapter 00 (Overview), then proceed sequentially through the completed chapters.

### Familiar with RL Theory

Skim Chapter 00, focus on Chapters 02-04 for mathematical foundations.

### Want to Implement

Read Chapter 00, skim 01-04, then wait for Chapters 05-07 on algorithms.

### Quick Understanding

Read Chapters 00, 01, and 04 for the essential concepts.

---

## Chapter Progression

```
00-Overview
    ↓
01-Core Concepts
    ↓
02-RKHS Foundations
    ↓
03-Energy and Fitness
    ↓
04-Reinforcement Field
    ↓
04a-Riesz Representer (supplement)
    ↓
05-Particle Memory
    ↓
06-MemoryUpdate  ← We are here
    ↓
07-RF-SARSA (planned)
    ↓
...
```

---

## Chapter Template

Each chapter follows this structure:

1. **Header**: Purpose, prerequisites, key concepts
2. **Introduction**: Why this topic matters
3. **Main Content**: Narrative explanation with examples
4. **Key Takeaways**: Summary points
5. **Next Steps**: Connection to following chapters

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $s$ | Environment state |
| $\theta$ | Action parameters |
| $z = (s, \theta)$ | Augmented state-action point |
| $k(\cdot, \cdot)$ | Kernel function |
| $\mathcal{H}_k$ | RKHS induced by kernel $k$ |
| $Q^+(z)$ | Field value (fitness) at $z$ |
| $E(z)$ | Energy at $z$, equals $-Q^+(z)$ |
| $\Omega$ | Particle memory |
| $w_i$ | Weight of particle $i$ |

---

## Key Equations

### Reinforcement Field
$$Q^+(z) = \sum_{i=1}^N w_i \, k(z, z_i)$$

### RKHS Inner Product
$$\langle k(x_1, \cdot), k(x_2, \cdot) \rangle_{\mathcal{H}_k} = k(x_1, x_2)$$

### Energy-Fitness Relationship
$$E(z) = -Q^+(z)$$

### Functional Gradient
$$\nabla_z Q^+(z) = \sum_{i=1}^N w_i \, \nabla_z k(z, z_i)$$

### Boltzmann Policy
$$\pi(\theta | s) \propto \exp(\beta \, Q^+(s, \theta))$$

---

**Last Updated**: January 11, 2026
