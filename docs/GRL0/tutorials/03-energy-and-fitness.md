# Chapter 3: Energy and Fitness

**Purpose**: Understand the value landscape and its two interpretations  
**Prerequisites**: Chapter 2 (RKHS Foundations)  
**Key Concepts**: Fitness function, energy function, sign conventions, energy-based models

---

## Introduction

In the previous chapters, we've described the reinforcement field as a function $Q^+(z)$ over augmented space. But what does this function represent? How should we interpret its values?

The original GRL paper called this a **fitness function** — higher values mean better configurations. Modern machine learning often uses **energy functions** — lower values mean better configurations.

These are two views of the same landscape, like looking at a mountain from above (where peaks are high) versus mapping depth (where valleys are low). Understanding this relationship is essential for:

- Connecting GRL to energy-based models
- Interpreting gradients correctly
- Extending to probabilistic and diffusion-based methods

---

## 1. Fitness Functions: The Original GRL View

### The Intuition

In evolutionary biology and optimization, **fitness** measures how "good" a configuration is. Higher fitness = better adapted = more likely to survive.

GRL uses the same intuition:

$$
F(s, \theta) \quad \text{high} \;\Rightarrow\; \text{good / compatible / desirable}
$$

The fitness function answers: *"How desirable is it to use action parameters $\theta$ in state $s$?"*

### Properties of the Fitness View

- **Improvement = ascent**: Getting better means going uphill
- **Gradients point toward improvement**: $\nabla F$ indicates the direction of increasing value
- **Policy = maximize fitness**: $\theta^* = \arg\max_\theta F(s, \theta)$

This is natural for RL where we maximize expected return.

---

## 2. Energy Functions: The Modern ML View

### The Intuition

In physics and modern ML, **energy** measures how much a system "resists" a configuration. Low energy = preferred = more likely.

$$

E(s, \theta) \quad \text{low} \;\Rightarrow\; \text{good / compatible / likely}
$$

The energy function answers: *"How much does the system resist this configuration?"*

### Properties of the Energy View

- **Improvement = descent**: Getting better means going downhill
- **Negative gradients point toward improvement**: $-\nabla E$ indicates improvement direction
- **Policy = minimize energy**: $\theta^* = \arg\min_\theta E(s, \theta)$

This is natural for physics-inspired models where systems seek minimum energy.

---

## 3. The Mathematical Relationship

### The Simple Connection

The relationship between fitness and energy is:

$$

E(s, \theta) = -F(s, \theta)
$$

That's it. Just a sign flip.

Optionally, with a temperature parameter:

$$

E(s, \theta) = -\frac{1}{\tau} F(s, \theta)
$$

### Same Landscape, Opposite Conventions

| Aspect | Fitness $F$ | Energy $E = -F$ |
|--------|------------|-----------------|
| Good configurations | High $F$ | Low $E$ |
| Improvement direction | $+\nabla F$ (ascent) | $-\nabla E$ (descent) |
| Optimal action | $\arg\max F$ | $\arg\min E$ |
| "Hills" | Desirable | Barriers |
| "Valleys" | Undesirable | Attractors |

**Critical point**: The extrema are in the same locations. A fitness maximum is an energy minimum:

$$

z^* = \arg\max_z F(z) = \arg\min_z E(z)
$$

The geometry is identical; only the labeling changes.

---

## 4. Why This Distinction Matters

### 4.1 Gradient Interpretation

Getting the sign right is essential for implementation:

**Fitness (GRL original)**:

- Reinforcement field gradient: $\nabla F(s, \theta)$ points toward improvement
- Policy improvement: Move in the direction of $\nabla F$

**Energy (modern)**:

- Force field: $-\nabla E(s, \theta)$ points toward improvement
- Policy improvement: Move against $\nabla E$

**Common bug**: Forgetting the sign when switching conventions.

### 4.2 Dynamics and Flows

For continuous-time extensions of GRL, the energy convention is standard.

**Gradient descent:**

$$

\frac{d\theta}{dt} = -\nabla E(s, \theta)
$$

**Langevin dynamics (with exploration):**

$$

d\theta_t = -\nabla E(s, \theta_t) \, dt + \sqrt{2\beta^{-1}} \, dW_t
$$

Writing these in fitness language works but feels unnatural to practitioners familiar with physics or diffusion models.

### 4.3 Probabilistic Interpretation

Fitness doesn't naturally define a probability distribution. Energy does:

$$

p(z) \propto \exp(-E(z)) = \exp(F(z))
$$

This **Boltzmann distribution** immediately enables:

- Probabilistic policies
- Sampling-based exploration
- Control-as-inference frameworks
- Connection to entropy-regularized RL

---

## 5. Energy-Based Models and GRL

### The Connection

GRL's reinforcement field is essentially an energy-based model (EBM) over augmented state-action space:

$$

E(z) = -Q^+(z) = -\sum_i w_i \, k(z, z_i)
$$

Each particle contributes to the energy landscape:

- **Positive weight** $w_i > 0$: Creates an energy well (attractor)
- **Negative weight** $w_i < 0$: Creates an energy barrier (repeller)
- **Kernel bandwidth**: Controls spatial extent of influence

### EBM Properties in GRL

| EBM Concept | GRL Realization |
|-------------|-----------------|
| Energy function | Negative reinforcement field $-Q^+$ |
| Low-energy regions | High-value state-action pairs |
| Energy minimization | Policy optimization |
| Boltzmann sampling | Softmax action selection |
| Energy gradient | Reinforcement field gradient |

### The Boltzmann Policy

GRL's softmax policy over action values is exactly Boltzmann sampling from the energy:

$$

\pi(\theta \mid s) \propto \exp\big(-\beta E(s, \theta)\big) = \exp\big(\beta Q^+(s, \theta)\big)
$$

where $\beta$ is inverse temperature:

- **$\beta \to 0$**: Uniform random actions (infinite temperature)
- **$\beta \to \infty$**: Greedy selection of energy minimum (zero temperature)
- **Finite $\beta$**: Stochastic exploration biased toward low energy

---

## 6. Visual Intuition

### The Landscape Metaphor

Imagine the augmented space as a terrain:

**Fitness view** (original GRL):

- Peaks = good actions (high fitness)
- Valleys = bad actions (low fitness)
- Policy = climb toward peaks
- Gradient = uphill direction

**Energy view** (modern):

- Valleys = good actions (low energy)
- Peaks = bad actions (high energy)
- Policy = descend toward valleys
- Gradient = steepest uphill (so move opposite)

### Particles Shape the Landscape

Each experience particle $(z_i, w_i)$ contributes a "bump" to the landscape:

- **Positive $w_i$ in fitness view**: A hill centered at $z_i$
- **Positive $w_i$ in energy view**: A valley (well) centered at $z_i$

The full landscape is the superposition of all particle contributions, smoothed by the kernel.

---

## 7. Practical Recommendations

### For GRL-v0 Reimplementation

1. **Preserve original conventions** when replicating the paper
2. **Match gradient signs** exactly to avoid bugs
3. **Test**: Verify that policy improvement increases $Q^+$

### For Modern Extensions

Switch to energy language when:

- Using diffusion-based methods
- Connecting to physics-based control
- Interfacing with EBM literature
- Implementing Langevin dynamics

### Convention Table

| Task | Use Fitness | Use Energy |
|------|------------|-----------|
| Reimplementing GRL-v0 | ✓ | |
| Reading original paper | ✓ | |
| Diffusion policy | | ✓ |
| Score matching | | ✓ |
| Control-as-inference | | ✓ |
| Connecting to EBMs | | ✓ |

---

## 8. The Energy Landscape in GRL

### Multi-Modal Landscapes

GRL's particle-based representation naturally handles multi-modal landscapes:

- Multiple particles can create multiple wells
- Each well corresponds to a viable policy mode
- No mode collapse (unlike some parametric methods)
- Exploration can discover multiple solutions

### Energy Wells and Basins

An **energy well** is a local minimum of $E(z)$ — a region where actions are good. The **basin of attraction** is the set of points that flow toward that well under gradient descent.

GRL's particle memory implicitly defines these basins through kernel overlap. Similar particles reinforce each other, creating deeper wells.

### No Explicit Parameterization

A key feature of GRL: the energy landscape is **not explicitly parameterized**. It emerges from:

- Particle positions $z_i$
- Particle weights $w_i$
- Kernel function $k$

This is fundamentally different from neural network-based EBMs where $E_\theta(z)$ is a parameterized function.

---

## 9. Score Functions and Gradients

### The Score Function

In modern generative models, the **score function** is the gradient of log-probability:

$$
\nabla_z \log p(z) = -\nabla_z E(z) \quad \text{(for Boltzmann } p \propto e^{-E}\text{)}
$$

For GRL's reinforcement field:

$$

\nabla_z \log \pi(z) \propto \nabla_z Q^+(z) = -\nabla_z E(z)
$$

### Connection to Diffusion Models

Diffusion models learn to reverse a noising process using the score function. GRL's reinforcement field gradient serves an analogous role: it indicates the direction toward high-value regions.

This connection opens paths to:

- Diffusion-based policy learning
- Score matching for value functions
- Denoising approaches to action selection

---

## 10. Summary

### The Core Relationship

$$

E(s, \theta) = -F(s, \theta)
$$

Fitness and energy are two views of the same landscape:

- **Fitness**: High = good (evolutionary/RL language)
- **Energy**: Low = good (physics/EBM language)

### Why It Matters

| Aspect | Impact |
|--------|--------|
| Gradients | Sign determines improvement direction |
| Dynamics | Energy form is standard for physics-based methods |
| Probability | $p \propto e^{-E}$ is immediate |
| Modern ML | EBM, diffusion, score matching use energy |

### The Core Innovation Remains

GRL's contribution was never about sign conventions. It was the idea that:

> **Reinforcement is a field over augmented state-action space, not a lookup table.**

Whether called fitness or energy, this field-based perspective is the foundation of GRL.

---

## Key Takeaways

1. **Fitness** (high = good) and **energy** (low = good) are equivalent via $E = -F$
2. **Same geometry**, opposite labeling conventions
3. **Gradients flip sign**: $\nabla F$ vs. $-\nabla E$ both point toward improvement
4. **Energy enables probability**: $p \propto \exp(-E)$ gives Boltzmann distribution
5. **GRL is an EBM** over augmented space with particle-based energy
6. **Boltzmann policy** = softmax over $Q^+$ = sampling from energy
7. **Choose convention** based on context: original GRL uses fitness, modern methods use energy

---

## Further Reading: Why Energy? The Physics Connection

**The energy formulation is not arbitrary!**

The energy function $E(z) = -Q^+(z)$ connects to one of the most fundamental principles in physics: the **principle of least action**. This principle states that systems evolve along trajectories that minimize an "action" functional, which naturally leads to energy-based formulations and Boltzmann distributions.

**📖 See Supplement: [Chapter 03a - The Principle of Least Action](03a-least-action-principle.md)**

This supplement explores:

- Classical mechanics and the action principle
- Path integral control theory
- Why GRL's Boltzmann policy emerges from action minimization
- How agents can **discover** smooth, optimal actions (not just select from fixed sets)
- Connection to neural network policy optimization

**For the quantum mechanical perspective:**
- **[Quantum-Inspired Chapter 09 - Path Integrals and Action Principles](../quantum_inspired/09-path-integrals-and-action-principles.md)**

---

## Next Steps

In **Chapter 4: The Reinforcement Field**, we'll explore:

- What exactly is a "functional field"?
- Why the reinforcement field is NOT a vector field
- How policy emerges from field geometry
- The continuous-time interpretation

---

**Related**: [Chapter 2: RKHS Foundations](02-rkhs-foundations.md), [Chapter 4: Reinforcement Field](04-reinforcement-field.md), [Supplement 03a: Least Action Principle](03a-least-action-principle.md)

---

**Last Updated**: January 14, 2026
