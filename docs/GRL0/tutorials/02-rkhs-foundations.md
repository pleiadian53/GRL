# Chapter 2: RKHS Foundations

**Purpose**: Understand the mathematical space where GRL lives  
**Prerequisites**: Chapter 1 (Core Concepts)  
**Key Concepts**: Reproducing Kernel Hilbert Space, inner products, function spaces, GP connection

---

## Introduction

In Chapter 1, we introduced kernel functions as a way to measure similarity between points in augmented space. We saw that the reinforcement field is a weighted sum of kernel evaluations:

$$
Q^+(z) = \sum_i w_i \, k(z, z_i)
$$

But why does this representation have such nice properties? Why does it generalize smoothly? Why can we take gradients?

The answer lies in a beautiful mathematical structure called a **Reproducing Kernel Hilbert Space (RKHS)**. Understanding RKHS is essential because it explains:

- Why GRL's value functions are well-behaved
- How generalization works mathematically
- What "functional gradient" really means
- Why GRL connects to Gaussian Processes

---

## 1. What is a Hilbert Space?

### Vectors Beyond Arrows

When you first learned about vectors, you probably thought of arrows in 2D or 3D space. But mathematically, a vector is anything that can be:

- **Added** to other vectors
- **Scaled** by numbers
- **Measured** for length and angle

Functions can be vectors too! Consider continuous functions on $[0, 1]$. We can:
- Add functions: $(f + g)(x) = f(x) + g(x)$
- Scale functions: $(cf)(x) = c \cdot f(x)$
- Measure: Using an inner product

### Inner Products

An inner product $\langle \cdot, \cdot \rangle$ generalizes the dot product. For vectors $u, v$:

$$
\langle u, v \rangle \to \text{a number measuring "alignment"}
$$

Properties:
- $\langle u, u \rangle \geq 0$ (non-negative)
- $\langle u, u \rangle = 0$ implies $u = 0$ (definite)
- $\langle u, v \rangle = \langle v, u \rangle$ (symmetric)
- Linear in each argument

### Definition of Hilbert Space

A **Hilbert space** is a vector space with an inner product that is **complete** (limits of sequences stay in the space).

**Examples**:
- $\mathbb{R}^n$ with the standard dot product
- The space of square-integrable functions $L^2$
- The space of functions induced by a kernel (RKHS)

---

## 2. Reproducing Kernel Hilbert Spaces

### The Special Property

An RKHS is a Hilbert space of functions with a remarkable property: **evaluation is continuous**.

What does this mean? In a general function space, knowing that two functions are "close" (small $\|f - g\|$) doesn't guarantee their values are close at any particular point. In an RKHS, it does.

### The Reproducing Property

An RKHS $\mathcal{H}_k$ has a kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ such that:

1. For each $x$, the function $k(x, \cdot)$ is in $\mathcal{H}_k$
2. For any $f \in \mathcal{H}_k$: $\langle f, k(x, \cdot) \rangle = f(x)$

The second property is called **reproducing**: inner product with $k(x, \cdot)$ "reproduces" the value at $x$.

### Kernel as Similarity

From the reproducing property:

$$
\langle k(x_1, \cdot), k(x_2, \cdot) \rangle_{\mathcal{H}_k} = k(x_1, x_2)
$$

**The kernel IS the inner product between feature representations.**

This is profound: the kernel function directly measures how similar two points are in the feature space induced by the RKHS.

---

## 3. Why RKHS Matters for GRL

### Functions as Vectors

In GRL, the value function $Q^+$ is not just "some function." It is a **vector in an RKHS**:

$$
Q^+(\cdot) = \sum_i w_i \, k(z_i, \cdot) \in \mathcal{H}_k
$$

This function is a linear combination of "basis functions" $k(z_i, \cdot)$, exactly like a finite-dimensional vector is a linear combination of basis vectors.

### Smoothness

RKHS functions inherit smoothness from the kernel. For example, with an RBF kernel:

$$
k(z, z') = \exp\left(-\frac{\|z - z'\|^2}{2\ell^2}\right)
$$

The induced functions are **infinitely differentiable**. Small changes in input produce small changes in output.

### Generalization

When we add a new particle $(z_N, w_N)$, the updated function:

$$
Q^+_{\text{new}}(z) = Q^+_{\text{old}}(z) + w_N k(z, z_N)
$$

The influence spreads smoothly according to the kernel. Points similar to $z_N$ (high $k(z, z_N)$) are affected more; distant points are affected less.

### Well-Defined Gradients

In an RKHS, we can differentiate the value function:

$$
\nabla_z Q^+(z) = \sum_i w_i \nabla_z k(z, z_i)
$$

This gradient exists and is smooth—essential for policy improvement.

---

## 4. Points Exist Only Through Functions

### A Philosophical Shift

Classical ML treats data points as primary objects and functions as derived. RKHS inverts this:

> **Functions are primary. Points exist only through how they act on functions.**

Each point $x$ is represented by the function $k(x, \cdot)$ — its "feature representation." Two points are compared via:

$$
k(x_1, x_2) = \langle k(x_1, \cdot), k(x_2, \cdot) \rangle
$$

Points don't have intrinsic coordinates; they have positions in function space.

### Epistemic Interpretation

In GRL, computing $k(z, z')$ doesn't just mean "these points are close." It means:

> **Evidence gathered at $z'$ is relevant for reasoning about $z$.**

This is **epistemic, not just geometric**. The kernel defines what counts as relevant experience.

---

## 5. Connection to Gaussian Processes

### GPs and RKHS Share Structure

Gaussian Processes are closely related to RKHS. For a GP with covariance function $k$:

| GP Object | RKHS Object |
|-----------|-------------|
| Covariance $k(x, x')$ | Inner product $\langle k(x, \cdot), k(x', \cdot) \rangle$ |
| Posterior mean | Vector in RKHS |
| Sample paths | (May or may not be in RKHS) |

### The Posterior Mean is Always in RKHS

Given data $\{(x_i, y_i)\}$, the GP posterior mean is:

$$
\mu(x) = \sum_{i=1}^n \alpha_i k(x, x_i)
$$

This is exactly a finite linear combination of kernel sections — by definition, an element of the RKHS.

**For GRL**: The value function $Q^+$ is constructed as a kernel superposition, which means it is guaranteed to be in the RKHS. All the nice mathematical properties apply.

### Sample Paths: A Subtlety

Individual random draws from a GP may or may not belong to the RKHS, depending on the kernel's smoothness:

| Kernel | Sample Paths in RKHS? |
|--------|----------------------|
| RBF (Gaussian) | Yes |
| Matérn ($\nu \geq 3/2$) | Yes |
| Brownian motion | No |

**For GRL**: We use posterior means, not random samples, so this subtlety doesn't affect us.

---

## 6. RKHS Inner Products and Probability

### Beyond Distance

In Euclidean space, similarity is measured by distance. In RKHS, similarity is measured by **inner products**.

The inner product $\langle f, g \rangle$ captures:
- How "aligned" two functions are
- The degree to which $f$ and $g$ "agree"
- A generalized notion of correlation

### Parallel to Quantum Mechanics

There's a deep structural parallel to quantum mechanics:

| Quantum Mechanics | GRL with RKHS |
|-------------------|---------------|
| State vector $\|\psi\rangle$ | Kernel feature $k(x, \cdot)$ |
| Inner product $\langle \phi \| \psi \rangle$ | Kernel evaluation $k(x, x')$ |
| Probability via $\|\langle \phi \| \psi \rangle\|^2$ | Compatibility via $k$ |
| Observables as operators | Value functionals |

In both frameworks:
- Inner products are fundamental
- Probability/compatibility emerges from overlap
- The "state" of the system is a vector in Hilbert space

### Probability as Derived, Not Primitive

Just as quantum mechanics derives probability from amplitudes:

$$
P(x) = |\langle \psi | x \rangle|^2
$$

GRL derives policy from field values:

$$
\pi(a|s) \propto \exp(\beta \, Q^+(s, a))
$$

In both cases, probability is a derived quantity, not a primitive input.

---

## 7. Practical Implications

### Kernel Choice Matters

The kernel defines:
- **Smoothness**: How quickly the value function can change
- **Lengthscale**: The "range of influence" of each particle
- **Feature relevance**: Which dimensions matter (via ARD kernels)

Common choices for GRL:

| Kernel | Properties | When to Use |
|--------|------------|-------------|
| RBF | Infinitely smooth, isotropic | Default choice, smooth domains |
| Matérn | Controllable smoothness | When less smoothness is appropriate |
| ARD-RBF | Learns feature relevance | High-dimensional with irrelevant features |
| Composite | Separate state/action kernels | When state and action have different scales |

### Computational Considerations

RKHS representations have complexity:
- **Memory**: $O(N)$ storage for $N$ particles
- **Query**: $O(N)$ kernel evaluations per point
- **Update**: $O(N)$ to add/modify particles

For large particle sets, approximations may be needed (inducing points, random features, neural approximations).

---

## 8. Summary: RKHS as the Foundation of GRL

### Key Concepts

| Concept | Meaning in GRL |
|---------|----------------|
| RKHS | The function space where $Q^+$ lives |
| Kernel | Defines similarity and smoothness |
| Inner product | Measures compatibility and overlap |
| Reproducing property | Evaluation is a linear functional |
| RKHS norm | Measures complexity/smoothness of functions |

### Why This Foundation Matters

1. **Mathematical Rigor**: All operations on $Q^+$ are well-defined
2. **Guaranteed Smoothness**: No pathological functions
3. **Principled Generalization**: Kernel determines how experience spreads
4. **Gradient Existence**: Policy improvement is well-posed
5. **Connection to GPs**: Uncertainty quantification is natural

### The Core Insight

> **GRL replaces pointwise reasoning with Hilbert-space reasoning.**

Similarity, value, policy, and learning are all geometric consequences of the RKHS inner product structure.

---

## Key Takeaways

1. **RKHS** is a Hilbert space of functions where evaluation is continuous
2. **The kernel** defines the inner product: $k(x, x') = \langle k(x, \cdot), k(x', \cdot) \rangle$
3. **Value functions** in GRL are vectors in RKHS: $Q^+ = \sum_i w_i k(z_i, \cdot)$
4. **Smoothness and generalization** are inherited from the kernel
5. **GP posterior means** are always in RKHS — validating GRL's mathematical foundation
6. **Inner products measure compatibility**, not just distance
7. **Points exist through functions**: $x$ is represented by $k(x, \cdot)$

---

## Next Steps

In **Chapter 3: Energy and Fitness**, we'll explore:
- The relationship between fitness and energy conventions
- How to interpret the value landscape
- Connection to energy-based models
- Why sign conventions matter

---

**Related**: [Chapter 1: Core Concepts](01-core-concepts.md), [Chapter 3: Energy and Fitness](03-energy-and-fitness.md)

---

**Last Updated**: January 11, 2026
