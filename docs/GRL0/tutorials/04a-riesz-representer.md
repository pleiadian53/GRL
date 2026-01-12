# Chapter 4a: The Riesz Representer — Gradients in Function Space

**Supplement to Chapter 4: Reinforcement Field**

## Why This Matters

In Chapter 4, we said:

> "The gradient $\nabla Q^+$ is the **Riesz representer** of a functional derivative."

But what does that mean? And why is it important for GRL?

This chapter unpacks the **Riesz representation theorem**—the mathematical machinery that lets us talk about "gradients" in infinite-dimensional function spaces.

**Prerequisites:** Chapter 2 (RKHS Foundations)

---

## The Problem: What Is a "Gradient" in Function Space?

### In Finite Dimensions (Familiar Territory)

In normal calculus, if you have a scalar function:

$$f(x) = x_1^2 + 2x_2$$

The gradient is a vector:

$$\nabla f = \begin{bmatrix} 2x_1 \\ 2 \end{bmatrix}$$

**Interpretation:** The gradient tells you the direction of steepest increase at each point.

**How we use it:** Inner product with a direction vector $v$:

$$\langle \nabla f, v \rangle = 2x_1 v_1 + 2 v_2$$

This gives the **directional derivative** of $f$ in direction $v$.

---

### In Infinite Dimensions (Function Space)

Now suppose you have a **functional**—a function that takes functions as input and returns a scalar:

$$L[f] = \int_0^1 f(x)^2 \, dx$$

**Example:** If $f(x) = x$, then:

$$L[f] = \int_0^1 x^2 \, dx = \frac{1}{3}$$

**Question:** What is the "gradient" of $L$ at $f$?

**Problem:** There's no finite-dimensional vector space here. Functions live in an infinite-dimensional space. What does $\nabla L$ even mean?

---

## The Idea: Represent the Derivative as a Function

The key insight from functional analysis:

> **Instead of computing a gradient vector, find the unique function $g$ that represents the derivative via inner product.**

Specifically, we want to find $g$ such that for any "direction" (test function) $h$:

$$\frac{d}{d\epsilon} L[f + \epsilon h] \bigg|_{\epsilon=0} = \langle g, h \rangle$$

This $g$ is called the **Riesz representer** of the derivative.

---

## The Riesz Representation Theorem

### Statement (Informal)

> **In a Hilbert space, every continuous linear functional can be represented as an inner product with a unique element of that space.**

### Statement (Formal)

Let $\mathcal{H}$ be a Hilbert space, and let $\phi: \mathcal{H} \to \mathbb{R}$ be a continuous linear functional. Then there exists a **unique** element $g \in \mathcal{H}$ such that:

$$\phi(f) = \langle g, f \rangle_{\mathcal{H}} \quad \text{for all } f \in \mathcal{H}$$

We call $g$ the **Riesz representer** of $\phi$.

### Why This Is Profound

This theorem says:

1. **Functionals are functions**: Any linear operation on functions can be represented by a specific function
2. **Derivatives live in the same space**: The derivative of a functional is itself an element of the Hilbert space
3. **Inner products encode everything**: All you need is the inner product structure

---

## Example 1: Point Evaluation Functional

Let's start simple.

### The Functional

In an RKHS $\mathcal{H}_k$ over domain $\mathcal{X}$, define:

$$\phi_x(f) = f(x)$$

This functional **evaluates** $f$ at point $x$.

### The Riesz Representer

By the Riesz representation theorem, there exists $g_x \in \mathcal{H}_k$ such that:

$$f(x) = \langle g_x, f \rangle_{\mathcal{H}_k}$$

**Question:** What is $g_x$?

**Answer:** It's the **kernel section** $k(x, \cdot)$!

By the reproducing property:

$$f(x) = \langle f, k(x, \cdot) \rangle_{\mathcal{H}_k}$$

**Interpretation:** The function $k(x, \cdot)$ **represents** the operation "evaluate at $x$."

---

## Example 2: Derivative of a Functional

Let's compute an actual derivative.

### The Functional

Consider:

$$L[f] = \int_0^1 f(x)^2 \, dx$$

We want the derivative at $f_0(x) = x$.

### Computing the Directional Derivative

For any test function $h$:

$$\frac{d}{d\epsilon} L[f_0 + \epsilon h] \bigg|_{\epsilon=0} = \frac{d}{d\epsilon} \int_0^1 (f_0 + \epsilon h)^2 \, dx \bigg|_{\epsilon=0}$$

$$= \int_0^1 2 f_0(x) h(x) \, dx = 2 \int_0^1 x \cdot h(x) \, dx$$

### The Riesz Representer

We need $g$ such that:

$$\int_0^1 x \cdot h(x) \, dx = \langle g, h \rangle_{L^2}$$

In $L^2[0,1]$ with the standard inner product:

$$\langle g, h \rangle = \int_0^1 g(x) h(x) \, dx$$

**Answer:** $g(x) = 2x$

**Interpretation:** The "gradient" of $L$ at $f_0$ is the function $g(x) = 2f_0(x) = 2x$.

---

## Example 3: Gradient of the GRL Value Functional

Now let's connect to GRL.

### The Setup

In GRL, the value function is:

$$Q^+(z) = \sum_i w_i k(z_i, z)$$

where $Q^+ \in \mathcal{H}_k$ (the RKHS induced by kernel $k$).

### The Functional

Consider the functional that evaluates $Q^+$ at a query point $z_0$:

$$\phi_{z_0}(Q^+) = Q^+(z_0)$$

### The Directional Derivative

For any "direction" $h \in \mathcal{H}_k$:

$$\frac{d}{d\epsilon} (Q^+ + \epsilon h)(z_0) \bigg|_{\epsilon=0} = h(z_0)$$

By the reproducing property:

$$h(z_0) = \langle h, k(z_0, \cdot) \rangle_{\mathcal{H}_k}$$

### The Riesz Representer

The gradient of the functional $\phi_{z_0}$ is:

$$\nabla \phi_{z_0} = k(z_0, \cdot)$$

**Interpretation:** The "direction" of steepest increase for the value function at $z_0$ is the kernel section $k(z_0, \cdot)$.

---

## Why This Matters for GRL

### 1. Gradients Are Functions, Not Vectors

In GRL, when we talk about the "gradient" $\nabla Q^+$, we mean:

> The unique function in $\mathcal{H}_k$ that represents how $Q^+$ changes in response to perturbations.

This gradient is **not** a finite-dimensional vector—it's an element of the infinite-dimensional RKHS.

### 2. The Reinforcement Field Is a Gradient Field

The reinforcement field is defined as:

$$\mathbf{F}(z) = \nabla_z Q^+(z)$$

Using the Riesz representation, this gradient is:

$$\nabla_z Q^+(z) = \sum_i w_i \nabla_z k(z_i, z)$$

Each $\nabla_z k(z_i, z)$ is itself a Riesz representer—the function that represents how $k(z_i, \cdot)$ changes at $z$.

### 3. Inner Products Compute Directional Derivatives

When we compute:

$$\langle \nabla Q^+, h \rangle$$

We're computing the **directional derivative** of $Q^+$ in direction $h$.

This is how the agent "probes" the value landscape to decide which direction (action) to take.

### 4. Policy Inference via Gradients

In GRL, policy inference involves finding the direction in augmented space that maximizes value:

$$\theta^* = \arg\max_\theta Q^+(s, \theta)$$

This is equivalent to following the gradient (Riesz representer) of $Q^+$ in the action parameter space.

---

## Notation Summary

Let's clarify all the notation we've used:

| Symbol | Meaning |
|--------|---------|
| $\mathcal{H}$ | A Hilbert space (e.g., RKHS) |
| $\phi: \mathcal{H} \to \mathbb{R}$ | A linear functional (maps functions to scalars) |
| $g \in \mathcal{H}$ | The Riesz representer of $\phi$ |
| $\langle g, f \rangle$ | Inner product in $\mathcal{H}$ |
| $L[f]$ | A functional that takes $f$ and returns a scalar |
| $\nabla L$ | The Riesz representer of the derivative of $L$ |
| $k(x, \cdot)$ | Kernel section (function of the second argument) |
| $Q^+ \in \mathcal{H}_k$ | The reinforcement field (value function in RKHS) |
| $\nabla Q^+$ | The Riesz representer of the value functional derivative |

---

## Visual Intuition

### Finite Dimensions

In $\mathbb{R}^2$:

```
Gradient:    ∇f = [2x₁, 2]  ← a vector

Direction:   v = [v₁, v₂]   ← another vector

Directional derivative: ⟨∇f, v⟩ = 2x₁v₁ + 2v₂
```

### Infinite Dimensions (RKHS)

In $\mathcal{H}_k$:

```
Gradient:    ∇Q⁺ = Σᵢ wᵢ ∇k(zᵢ, ·)  ← a function

Direction:   h ∈ ℋₖ                ← another function

Directional derivative: ⟨∇Q⁺, h⟩ = ∫ ∇Q⁺(z) h(z) dμ(z)
```

**Key insight:** Same structure, different space!

---

## Example 4: Computing a Concrete Gradient

Let's compute an explicit example with a Gaussian kernel.

### Setup

Gaussian RBF kernel:

$$k(z, z') = \exp\left(-\frac{\|z - z'\|^2}{2\sigma^2}\right)$$

Value function at a single particle:

$$Q^+(z) = w_1 k(z_1, z) = w_1 \exp\left(-\frac{\|z - z_1\|^2}{2\sigma^2}\right)$$

### Computing the Gradient

The gradient with respect to $z$ is:

$$\nabla_z Q^+(z) = w_1 \nabla_z \exp\left(-\frac{\|z - z_1\|^2}{2\sigma^2}\right)$$

Using the chain rule:

$$\nabla_z \exp\left(-\frac{\|z - z_1\|^2}{2\sigma^2}\right) = \exp\left(-\frac{\|z - z_1\|^2}{2\sigma^2}\right) \cdot \left(-\frac{z - z_1}{\sigma^2}\right)$$

So:

$$\nabla_z Q^+(z) = -\frac{w_1}{\sigma^2} (z - z_1) \exp\left(-\frac{\|z - z_1\|^2}{2\sigma^2}\right)$$

### Interpretation

- **Magnitude**: Largest near $z_1$, decays with distance
- **Direction**: Points from $z_1$ toward $z$ (if $w_1 > 0$, repulsive gradient)
- **Sign**: If $w_1 > 0$, gradient points away from particle; if $w_1 < 0$, toward particle

**For GRL policy inference:** The agent follows the gradient to move toward high-value regions (positive particles attract, negative particles repel).

---

## Practical Implications for GRL

### 1. Gradients Are Computable

Because RKHS gradients have Riesz representers, we can compute them explicitly:

$$\nabla Q^+(z) = \sum_i w_i \nabla_z k(z_i, z)$$

No need for finite differences or backpropagation—the gradient is analytical.

### 2. Gradients Guide Policy

The policy at state $s$ can be computed by finding:

$$\theta^* = \arg\max_\theta Q^+(s, \theta)$$

This is equivalent to following the gradient in parameter space:

$$\theta^* = \theta_0 + \alpha \nabla_\theta Q^+(s, \theta_0)$$

### 3. Gradients Enable Continuous Optimization

Because gradients exist and are smooth (for smooth kernels), we can use gradient-based optimization for action selection—even though the "action space" is infinite-dimensional!

### 4. Functional Derivatives Are Well-Defined

The Riesz representation theorem guarantees that all the functional derivatives we use in GRL (value derivatives, policy gradients, energy gradients) are well-defined elements of the RKHS.

---

## Common Misconceptions

### Misconception 1: "The Gradient Is a Vector"

**Reality:** In RKHS, the gradient is a **function**—an element of the same Hilbert space.

### Misconception 2: "RKHS Gradients Are Approximations"

**Reality:** RKHS gradients are **exact**—they are the Riesz representers defined by the inner product structure.

### Misconception 3: "We Need to Discretize to Compute Gradients"

**Reality:** For analytic kernels (RBF, Matérn, polynomial), gradients have closed-form expressions.

### Misconception 4: "Functional Derivatives Are Esoteric"

**Reality:** They're just the infinite-dimensional version of ordinary derivatives, made rigorous by the Riesz representation theorem.

---

## Summary

| Finite Dimensions | Infinite Dimensions (RKHS) |
|-------------------|----------------------------|
| Gradient is a vector $\nabla f \in \mathbb{R}^n$ | Gradient is a function $\nabla L \in \mathcal{H}$ |
| Inner product: $\langle \nabla f, v \rangle$ | Inner product: $\langle \nabla L, h \rangle_{\mathcal{H}}$ |
| Directional derivative via dot product | Directional derivative via RKHS inner product |
| Gradient descent in $\mathbb{R}^n$ | Gradient descent in $\mathcal{H}$ |

**The Riesz representation theorem says:** These are the same structure, just in different spaces.

---

## Key Takeaways

1. **Riesz Representer = Gradient in Function Space**
   - Every linear functional has a unique function that represents it via inner product
   - This function is the "gradient"

2. **RKHS Makes Gradients Tractable**
   - Reproducing property: $f(x) = \langle f, k(x, \cdot) \rangle$
   - Kernel sections $k(x, \cdot)$ are Riesz representers of point evaluations

3. **GRL's Reinforcement Field Is a Gradient Field**
   - $\nabla Q^+$ is the Riesz representer of value function changes
   - Policy inference follows this gradient to maximize value

4. **Notation:**
   - $\nabla Q^+$: The gradient (a function in RKHS)
   - $\langle \nabla Q^+, h \rangle$: Directional derivative in direction $h$
   - $\nabla_z k(z_i, z)$: Gradient of kernel (computable analytically)

---

## Further Reading

### Within This Tutorial

- **Chapter 2**: [RKHS Foundations](02-rkhs-foundations.md) — Inner products and reproducing property
- **Chapter 4**: [Reinforcement Field](04-reinforcement-field.md) — The gradient field in GRL
- **Chapter 5**: [Particle Memory](05-particle-memory.md) — How particles induce the value function

### Advanced Topics

- **Quantum-Inspired**: [Wavefunction Interpretation](../quantum_inspired/01a-wavefunction-interpretation.md) — State vectors vs. coordinate representations

### Mathematical References

- **Riesz Representation Theorem:**
  - Rudin, W. (1991). *Functional Analysis*. McGraw-Hill. (Chapter 6)
  - Reed & Simon (1980). *Functional Analysis*. Academic Press.

- **RKHS and Reproducing Property:**
  - Berlinet & Thomas-Agnan (2004). *Reproducing Kernel Hilbert Spaces in Probability and Statistics*. Springer.
  - Steinwart & Christmann (2008). *Support Vector Machines*. Springer.

- **Calculus of Variations:**
  - Gelfand & Fomin (1963). *Calculus of Variations*. Dover.

---

## Next Steps

In **Chapter 6: MemoryUpdate Algorithm**, we'll see how the Riesz representer structure enables efficient updates to the reinforcement field as new particles are added to memory.

**Spoiler:** Because gradients have explicit representations, we can compute value function updates analytically!

---

**Last Updated:** January 12, 2026
