# Chapter 1: Core Concepts

**Purpose**: Introduce the fundamental building blocks of GRL  
**Prerequisites**: Chapter 0 (Overview)  
**Key Concepts**: Augmented state space, parametric actions, experience particles, kernel similarity

---

## Introduction

In Chapter 0, we introduced the central idea of GRL: treating actions as parametric operators rather than fixed symbols. Now we'll formalize the core building blocks that make this possible:

1. **Parametric Actions**: How actions are represented as parameter vectors
2. **Augmented State Space**: The joint space of states and action parameters  
3. **Experience Particles**: How we represent and store experience
4. **Kernel Similarity**: How we measure relationships between experiences

These concepts form the foundation on which the reinforcement field, algorithms, and policy inference are built.

---

## 1. Parametric Actions

### From Symbols to Parameters

In traditional RL, an action $a$ is either:

- A discrete symbol from a finite set: $a \in \{1, 2, ..., K\}$
- A continuous vector from a bounded region: $a \in \mathbb{R}^d$

In GRL, we take the continuous view further. An action is represented by a **parameter vector** $\theta \in \Theta$ that specifies an operator:

$$
\theta \to \hat{O}(\theta)
$$

The parameter space $\Theta$ is typically $\mathbb{R}^d$ for some dimension $d$, though it could be a more structured manifold.

### Examples

| Domain | Parameters $\theta$ | Operator $\hat{O}(\theta)$ |
|--------|---------------------|---------------------------|
| 2D Navigation | $(F_x, F_y)$ | Force vector applied to agent |
| Pendulum | $\tau$ | Torque applied to joint |
| Portfolio | $(w_1, ..., w_n)$ | Asset allocation weights |
| Image transformation | $(r, \theta, s)$ | Rotation, angle, scale |

### Why Parameters Matter

By treating actions as parameters, we gain:

**Continuity**: Nearby parameters $\theta$ and $\theta'$ produce similar effects. This enables smooth generalization.

**Differentiability**: We can compute gradients of outcomes with respect to $\theta$.

**Compositionality**: Parameters can be structured (e.g., hierarchical) to enable compositional actions.

**Interpretability**: Parameters often have physical meaning (force magnitude, angle, etc.).

---

## 2. Augmented State Space

### Combining States and Actions

The key insight of GRL is to reason about states and actions *together* as points in a unified space. We define the **augmented state-action point**:

$$
z = (s, \theta) \in \mathcal{Z} = \mathcal{S} \times \Theta
$$

where:

- $s \in \mathcal{S}$ is the environment state (possibly embedded/encoded)
- $\theta \in \Theta$ is the action parameter vector
- $\mathcal{Z}$ is the augmented space

### Why Augment?

In standard RL, we might learn $Q(s, a)$ — the value of taking action $a$ in state $s$. In GRL, we learn $Q^+(z) = Q^+(s, \theta)$ — a value function over the *entire* augmented space.

This has several advantages:

**Smooth Value Landscape**: The value function is smooth over the continuous augmented space, enabling generalization.

**Unified Representation**: State and action are treated symmetrically, enabling richer representations.

**Gradient Information**: We can compute $\nabla_\theta Q^+(s, \theta)$ — how value changes with action parameters.

### Embedding Functions

In practice, we often use embedding functions to transform raw states and actions into suitable representations:

$$

z = (x_s(s), x_a(\theta))
$$

where:

- $x_s: \mathcal{S} \to \mathbb{R}^{d_s}$ embeds states
- $x_a: \Theta \to \mathbb{R}^{d_a}$ embeds action parameters

These embeddings might be:

- Identity (raw features)
- Learned neural network encodings
- Hand-crafted features

The choice of embedding affects the geometry of the augmented space and thus how similarity and generalization work.

---

## 3. Experience Particles

### What is a Particle?

In GRL, experience is stored not as a replay buffer of transitions, but as a collection of **particles** in augmented space. Each particle is a tuple:

$$

\omega_i = (z_i, w_i) = ((s_i, \theta_i), w_i)
$$

where:

- $z_i = (s_i, \theta_i)$ is the **location** in augmented space
- $w_i \in \mathbb{R}$ is the **weight** (typically related to value)

### Particle Memory

The agent maintains a **particle memory**:

$$

\Omega = \{(z_1, w_1), (z_2, w_2), ..., (z_N, w_N)\}
$$

This collection of weighted particles represents the agent's accumulated experience. It's analogous to:

- A weighted sample approximation to a distribution
- A nonparametric function representation
- A memory of "what happened where and how good it was"

### Particles vs. Replay Buffer

| Replay Buffer | Particle Memory |
|---------------|-----------------|
| Stores transitions $(s, a, r, s')$ | Stores points $(z, w)$ in augmented space |
| Used for sampling and replaying | Used for function approximation and inference |
| Finite capacity, FIFO or priority | Dynamic, merging/pruning operations |
| Supports temporal learning | Supports spatial generalization |

### Particle Operations

The particle memory supports several operations:

**Add**: Insert a new particle $(z, w)$

**Query**: Evaluate the reinforcement field at a point $z$ using nearby particles

**Merge**: Combine similar particles to prevent unbounded growth

**Prune**: Remove low-influence particles

**Update**: Modify weights based on new reinforcement signals

These operations are formalized in Algorithm 1 (MemoryUpdate), covered in Chapter 6.

---

## 4. Kernel Similarity

### The Role of Kernels

How do we determine which particles are "nearby" or "similar"? GRL uses **kernel functions** to define similarity in augmented space.

A kernel $k: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ measures how similar two points are:

$$
k(z, z') = k((s, \theta), (s', \theta'))
$$

Higher values mean more similar.

### Common Kernels

**Radial Basis Function (RBF) / Gaussian**:

$$

k(z, z') = \exp\left(-\frac{\|z - z'\|^2}{2\ell^2}\right)
$$

where $\ell$ is the **lengthscale** controlling how quickly similarity decays with distance.

**Automatic Relevance Determination (ARD)**:

$$

k(z, z') = \exp\left(-\sum_{d=1}^{D} \frac{(z_d - z'_d)^2}{2\ell_d^2}\right)
$$

Each dimension has its own lengthscale $\ell_d$, allowing the kernel to learn which features matter most.

**Composite Kernels**:

For augmented space, we might use:

$$

k(z, z') = k_s(s, s') \cdot k_a(\theta, \theta')
$$

where $k_s$ and $k_a$ are separate kernels for state and action components.

### Why Kernels Matter

Kernels are central to GRL because they define:

**Generalization**: How experience at one point informs predictions at other points

**Smoothness**: How quickly the value function can change

**Feature Relevance**: Which dimensions of state/action matter (via ARD)

**Geometry**: The "shape" of the augmented space for learning

### Kernel-Induced Function Representation

Given particles $\Omega = \{(z_i, w_i)\}$ and kernel $k$, we can define a function over the entire augmented space:

$$

f(z) = \sum_{i=1}^{N} w_i \, k(z, z_i)
$$

This is the **reinforcement field** — a smooth function that assigns values to every point in augmented space based on the weighted contributions of all particles.

This representation:

- Is **nonparametric**: No fixed neural network architecture
- Is **smooth**: Inherits smoothness from the kernel
- **Generalizes**: Points far from any particle get low values
- Is **adaptive**: Adding particles reshapes the function

---

## 5. Putting It Together

Let's trace how these concepts connect:

### 1. Agent in State $s$

The agent observes state $s$ from the environment.

### 2. Consider Action Parameters

The agent considers action parameter $\theta$, forming augmented point $z = (s, \theta)$.

### 3. Query Particle Memory

Using the kernel, the agent computes the reinforcement field value:

$$

Q^+(z) = \sum_{i} w_i \, k(z, z_i)
$$

### 4. Select Action

The agent selects action parameters that maximize $Q^+$ (or samples according to a policy).

### 5. Execute and Observe

The action is executed, reward $r$ is received, next state $s'$ is observed.

### 6. Update Particles

A new particle is added or existing particles are updated based on the experience.

### 7. Repeat

The cycle continues, with the reinforcement field evolving as particles accumulate.

---

## Visual Intuition

Imagine a 2D augmented space where:

- The x-axis represents some aspect of state (e.g., position)
- The y-axis represents action parameter (e.g., force magnitude)

Each particle is a point in this 2D space with an associated weight (color/size indicating value).

The kernel defines how much each particle influences nearby points — like a Gaussian "bump" centered at each particle.

The reinforcement field is the sum of all these bumps — a smooth landscape over the entire space.

**High regions**: Good state-action combinations (high expected return)

**Low regions**: Poor combinations (low expected return)

**Sparse regions**: Uncertainty (few particles nearby)

Policy learning = navigating and reshaping this landscape.

---

## Key Takeaways

1. **Parametric actions** represent actions as parameter vectors $\theta$ that specify operators

2. **Augmented state space** $\mathcal{Z} = \mathcal{S} \times \Theta$ combines state and action parameters

3. **Experience particles** $(z_i, w_i)$ are weighted points in augmented space representing experience

4. **Kernel functions** $k(z, z')$ define similarity and enable smooth generalization

5. **The reinforcement field** $Q^+(z) = \sum_i w_i k(z, z_i)$ emerges from particles and kernel

6. Together, these enable **continuous action spaces**, **smooth generalization**, and **uncertainty quantification**

---

## Next Steps

In **Chapter 2: RKHS Foundations**, we'll explore:

- What is a Reproducing Kernel Hilbert Space (RKHS)?
- Why the reinforcement field lives in an RKHS
- The mathematical properties that make this useful
- Connection to Gaussian Processes

---

**Related**: [Chapter 0: Overview](00-overview.md), [Chapter 2: RKHS Foundations](02-rkhs-foundations.md)

---

**Last Updated**: January 11, 2026
