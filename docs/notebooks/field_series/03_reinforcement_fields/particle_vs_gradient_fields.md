# Particle-Based Q⁺ Fields vs. True Gradient Fields

**A detailed comparison for understanding GRL policy behavior**

---

## Overview

When visualizing GRL's inferred policy as a vector field, you may notice that arrows appear mostly **parallel** rather than **converging** on the goal. This document explains why this happens and what it reveals about the nature of particle-based reinforcement fields.

---

## Two Types of Fields

### 1. True Gradient Field (Goal-Directed Potential)

A classical approach to navigation uses a **potential function** that directly encodes distance to the goal:

$$\phi(x, y) = -\|(x, y) - (x_{goal}, y_{goal})\|^2$$

The gradient of this potential:

$$\nabla \phi = -2 \begin{pmatrix} x - x_{goal} \\ y - y_{goal} \end{pmatrix}$$

This gradient field has a key property: **arrows at every point converge toward the goal**.

```
        ↘   ↓   ↙
         ↘  ↓  ↙
    →  →  [GOAL]  ←  ←
         ↗  ↑  ↖
        ↗   ↑   ↖
```

**Pros:**

- Globally optimal direction at every point
- Arrows naturally "rotate" to point at goal

**Cons:**

- Requires explicit knowledge of goal location
- Cannot encode constraints (obstacles) naturally
- No learning — purely geometric

---

### 2. Particle-Based Q⁺ Field (GRL)

GRL builds the reinforcement field from **experience particles**:

$$Q^+(z) = \sum_{i=1}^{N} w_i \, k(z, z_i)$$

where $z = (x, y, v_x, v_y)$ is the augmented state-action and $k$ is the RBF kernel.

The policy is inferred by finding the best action at each state:

$$\pi(s) = \arg\max_\theta Q^+(s, \theta)$$

**Key difference**: The field gradient depends on **where particles are located**, not on the goal position directly.

---

## Why Arrows Appear Parallel

### The Sparse Particle Problem

Consider synthetic particles placed along a diagonal path to the goal:

```
Particles:  (0.5, 0.5) → (1, 1) → (1.5, 1.5) → ... → (4, 4) [GOAL]
            All with action direction ≈ (0.7, 0.7)
```

At any query point $(x, y)$, the $Q^+$ gradient is dominated by the **nearest particles**. Since all particles encode similar action directions (toward upper-right), the inferred policy is approximately constant across the domain.

### Mathematical Explanation

The gradient of $Q^+$ with respect to action direction:

$$\nabla_\theta Q^+(s, \theta) = \sum_{i=1}^{N} w_i \nabla_\theta k((s, \theta), z_i)$$

If particles are clustered with similar action components, this gradient points in roughly the same direction everywhere — hence parallel arrows.

### Discrete Action Search Limitation

In the notebook, we search over 16 discrete angles:

```python
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
```

When the $Q^+$ landscape is relatively flat (due to sparse particles), many states select the same "best" angle, reinforcing the parallel appearance.

---

## What Would Make Arrows Converge?

### 1. Richer Particle Coverage

If particles recorded trajectories from **many starting points** approaching the goal from different angles:

```
From (0, 4):  particles with action ≈ (0.7, -0.7)  [down-right]
From (4, 0):  particles with action ≈ (-0.7, 0.7) [up-left... wait, away from goal!]
From (0, 0):  particles with action ≈ (0.7, 0.7)  [up-right]
```

With diverse coverage, the field would encode **local directional information** that varies across the domain.

### 2. Goal-Conditioned Particles

Particles could encode "turn toward goal" rather than absolute directions:

$$\theta_i = \text{angle toward goal from } s_i$$

This would make the particle-based field behave more like a gradient field.

### 3. Hybrid Approach

Combine particle-based learning with a goal-directed potential:

$$Q^+_{hybrid}(z) = Q^+_{particles}(z) + \lambda \cdot \phi_{goal}(s)$$

---

## The GRL Perspective: Feature, Not Bug

The parallel-arrow behavior reveals a fundamental property of GRL:

> **Policy quality depends on the richness of the particle memory.**

This is actually desirable because:

1. **No goal knowledge required**: The agent learns from experience, not from knowing where the goal is.

2. **Constraint encoding**: Negative particles naturally encode obstacles — something a pure gradient field cannot do.

3. **Generalization**: With enough diverse experience, the field generalizes to unseen states.

4. **Adaptability**: If the goal moves, the agent can learn new particles without redesigning the potential function.

---

## Visualization Comparison

### Gradient Field (Ideal)
```
    ↘  ↓  ↙
     ↘ ↓ ↙
  →  → ★ ←  ←    ★ = Goal
     ↗ ↑ ↖
    ↗  ↑  ↖
```

### Particle-Based Field (Sparse Diagonal Particles)
```
    ↗  ↗  ↗
    ↗  ↗  ↗
    ↗  ↗  ★       ★ = Goal
    ↗  ↗  ↗
    ↗  ↗  ↗
```

### Particle-Based Field (Rich Coverage)
```
    ↗  ↗  →
    ↗  ↗  ↗
    ↗  ↗  ★       ★ = Goal
    ↑  ↗  ↗
    ↑  ↑  ↗
```

---

## Implications for GRL Implementation

1. **Exploration matters**: Diverse trajectories lead to richer particle coverage.

2. **Particle placement**: Strategic particle placement (or learning algorithms like RF-SARSA) is crucial.

3. **Lengthscale tuning**: The kernel lengthscale $\ell$ controls how far each particle's influence spreads.

4. **Action discretization**: Finer action discretization can reveal more nuanced policy structure.

---

## Summary

| Aspect | Gradient Field | Particle-Based Q⁺ |
|--------|----------------|-------------------|
| **Source** | Goal position | Experience particles |
| **Arrows** | Converge on goal | Follow particle gradient |
| **Obstacles** | Cannot encode | Natural via negative particles |
| **Learning** | None (geometric) | From experience |
| **Coverage** | Global | Depends on particle density |

**Bottom line**: The parallel arrows in sparse particle scenarios are expected behavior. With richer experience, the particle-based field approaches the behavior of a goal-directed gradient field while retaining the ability to encode constraints and learn from experience.

---

## References

- GRL Paper: Figure 4 (2D Navigation Domain)
- Notebook: `notebooks/field_series/03_reinforcement_fields.ipynb`
- Related: `notebooks/field_series/01_classical_vector_fields.ipynb` (gradient fields)
