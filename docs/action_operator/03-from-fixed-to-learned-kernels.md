# Part 3: From Fixed Kernels to Learned Kernels — The Feature Map Extension

**Series:** Action Operators in Generalized Reinforcement Learning  
**Part:** 3 of 3  
**Audience:** Self-reference and interview prep  
**Prerequisites:** Parts 1-2, basic kernel methods, familiarity with deep metric learning

---

## 1. Recap: The Kernel's Role in GRL0

In GRL0, the kernel function $k(x, x')$ is the foundational object. It defines:

- **The value function:** $Q^+(x) = \sum_i w_i k(x, x_i)$ — kernel superposition of particles
- **The reinforcement field:** $\nabla_x Q^+(x) = \sum_i w_i \nabla_x k(x, x_i)$ — gradient of the superposition
- **The similarity structure:** $k(x, x')$ determines which configurations are "neighbors" in augmented space
- **The RKHS geometry:** The inner product $\langle k(x, \cdot), k(x', \cdot) \rangle_{\mathcal{H}_k} = k(x, x')$ defines the native workspace

The kernel is simultaneously: a similarity measure, a basis function generator, and the definition of the agent's geometric universe. Everything flows from it.

But in GRL0, the kernel was *chosen a priori* — typically RBF with manually set lengthscale — and never updated during learning. This section traces why that's limiting and how to move beyond it.

---

## 2. Mercer's Theorem: The Feature Map Interpretation

The equation

$$k(x, x') = \langle \varphi(x), \varphi(x') \rangle_{\mathcal{F}}$$

is Mercer's theorem: any positive-definite kernel factorizes through some feature map $\varphi: \mathcal{X} \to \mathcal{F}$ into a (potentially infinite-dimensional) feature space $\mathcal{F}$.

For the RBF kernel, this feature map is infinite-dimensional — it maps inputs to a Hilbert space of Gaussian functions. For polynomial kernels, the feature map is finite and corresponds to all monomials up to a given degree.

The key insight is that **the kernel is doing implicit feature extraction**. When we compute $k(x, x')$, we are:

1. Mapping both inputs through the feature map $\varphi$
2. Computing their inner product (dot product) in feature space
3. Returning the result as a scalar similarity

The feature map is *fixed* once the kernel is chosen. An RBF kernel with lengthscale $\ell$ imposes a specific notion of similarity: two augmented states are similar if and only if they are close in Euclidean distance, scaled by $\ell$. This is a strong assumption about the structure of the value function.

---

## 3. The Metric Learning Analogy

This feature-map view reveals that GRL0's kernel is doing **metric learning with a frozen metric**. Let's make this precise.

### 3.1 What Metric Learning Does

In metric learning, we learn a distance function $d(x, x')$ (or equivalently, a similarity function) such that:

- **Semantically similar** inputs are mapped close together
- **Semantically dissimilar** inputs are mapped far apart

The standard approach: learn a feature map $\varphi_\psi$ (parameterized by $\psi$) such that

$$d_\psi(x, x') = \|\varphi_\psi(x) - \varphi_\psi(x')\|^2$$

is small for similar inputs and large for dissimilar ones. The similarity $k_\psi(x, x') = \langle \varphi_\psi(x), \varphi_\psi(x') \rangle$ is the dual view.

### 3.2 GRL0 as Frozen Metric Learning

In GRL0, the kernel $k(x, x')$ defines exactly this kind of similarity — but the metric is **frozen** at initialization:

- Two augmented states $x = (s, \theta)$ and $x' = (s', \theta')$ are "similar" iff $k(x, x') \approx 1$
- This similarity determines how evidence propagates: if $k(x, x') \approx 1$, then the value observed at $x'$ strongly influences the predicted value at $x$

The Gram matrix $K_{ij} = k(x_i, x_j)$ over accumulated particles acts as a **neighborhood graph induced by the kernel**. If $k(x_i, x_j)$ is large, the two particles are neighbors — their values should be consistent. The GP optimization finds weights $w_i$ that respect this neighborhood structure: the posterior mean $Q^+(x) = \sum_i w_i k(x, x_i)$ is the smoothest function (in the RKHS norm sense) that fits the observed values.

**This is the core property:** if $k(x, x')$ indicates high similarity, then the values $Q^+(x)$ and $Q^+(x')$ will be close, because the RKHS norm penalizes functions that vary rapidly between similar inputs. The kernel defines what "rapidly varying" means.

### 3.3 The Limitation

The frozen metric works well when the kernel's notion of similarity aligns with the true value structure. For RBF:

$$k_{\text{RBF}}(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

this means: two configurations are similar iff they are *Euclidean-close* in augmented space.

**When this breaks:**

- **Non-smooth value functions:** If the optimal policy has sharp decision boundaries (e.g., "go left if $x < 0$, go right if $x > 0$"), the RBF kernel smooths them out. Points near the boundary with very different optimal actions get treated as similar.

- **Irrelevant dimensions:** If some state dimensions don't affect the value (e.g., color of walls in a navigation task), the RBF kernel still uses them for similarity, wasting capacity.

- **Nonlinear feature interactions:** If value depends on interactions between state and action features (e.g., "action $\theta_1$ is good when state is in region A, bad in region B"), Euclidean distance in $(s, \theta)$ misses this structure.

- **High-dimensional states:** For pixel inputs, token embeddings, or graph features, Euclidean distance is nearly meaningless. All high-dimensional points are approximately equidistant (curse of dimensionality).

ARD (Automatic Relevance Determination) kernels — where each dimension gets its own lengthscale — partially address the irrelevant-dimensions problem but not the others.

---

## 4. GP-Based Spectral Clustering: GRL0's Post-Hoc Feature Discovery

GRL0 had an elegant partial solution to the fixed-kernel problem: **spectral clustering on the Gram matrix**.

### 4.1 The Mechanism

1. Fix kernel $k$, accumulate particles, fit GP posterior $Q^+$
2. Compute the Gram matrix $K_{ij} = k(x_i, x_j)$ over particles
3. Eigendecompose: $K = U \Lambda U^T$
4. Top eigenvectors $u_1, \ldots, u_p$ define **functional subspaces** in RKHS

The Gram matrix is a weighted adjacency matrix of the particle graph. Its eigenvectors reveal clusters — groups of particles that are mutually similar under $k$ and dissimilar from other groups. These clusters correspond to *behavioral modes*: configurations where the agent should act similarly.

### 4.2 Connection to Feature Maps

The eigendecomposition of $K$ is directly related to the feature map. Since $K_{ij} = \langle \varphi(x_i), \varphi(x_j) \rangle$, the eigendecomposition of $K$ is equivalent to **PCA in feature space** (this is exactly kernel PCA).

The top eigenvectors $u_k$ define the principal axes of variation in feature space. They are the directions along which the particle configurations vary most, as measured by the kernel.

### 4.3 What This Achieves and What It Doesn't

**Achieves:**
- Discovers latent structure in the value landscape
- Finds behavioral modes (clusters of similarly-valued configurations)
- Provides a low-dimensional representation of the particle ensemble
- Connects kernel methods to spectral graph theory

**Doesn't achieve:**
- The kernel $k$ is still fixed — eigendecomposition discovers structure *within* the geometry that $k$ defines, but can't discover structure that $k$ is blind to
- Post-hoc: features are discovered after learning, not during it
- Computational cost: eigendecomposition is $O(N^3)$ in the number of particles

This is *posterior-side* feature discovery: use a fixed prior (the kernel), fit the model, then analyze what the model learned. The alternative — *prior-side* feature discovery — is to learn the kernel itself.

---

## 5. Learned Kernels: Replacing $\varphi$ with $\varphi_\psi$

The natural extension: replace the fixed feature map with a **learned** one.

$$k_\psi(x, x') = \langle \varphi_\psi(x), \varphi_\psi(x') \rangle$$

where $\varphi_\psi: \mathcal{X} \to \mathbb{R}^D$ is a neural network with parameters $\psi$.

### 5.1 What This Preserves

The inner-product form guarantees that $k_\psi$ is a valid positive-definite kernel for any $\psi$. This means:

- **Particle memory survives:** $Q^+(x) = \sum_i w_i k_\psi(x, x_i)$ remains well-defined
- **RKHS structure survives:** conditional on frozen $\psi$, all the RKHS machinery applies
- **Reinforcement field survives:** $\nabla_x Q^+$ is still computable via the chain rule through $\varphi_\psi$
- **Energy-landscape policy survives:** $\theta \leftarrow \theta + \eta \nabla_\theta Q^+(s, \theta)$ still works

The particle-based, field-driven character of GRL is preserved. What changes is the *geometry* of the field — the learned kernel warps the space so that "similar" configurations are those that *should have similar values*, not those that happen to be Euclidean-close.

### 5.2 What This Relaxes

- **Representer theorem:** Holds only conditional on frozen $\psi$. When $\psi$ changes, the optimal weights $w_i$ must be recomputed. Training becomes alternating: (i) fit weights with $\psi$ frozen, (ii) update $\psi$ with weights frozen.

- **Stationarity:** As $\psi$ evolves, $Q^+(x)$ changes at *all* particles, not just the newly added one. Old experience gets reinterpreted under the new metric. This breaks GRL0's "store and forget" semantics.

- **Uncertainty calibration:** A fixed RBF kernel gives clean uncertainty: sparse particles → low kernel mass → low confidence. A learned kernel can map novel inputs to arbitrary locations in feature space, potentially creating false confidence.

### 5.3 Training Objectives for $\varphi_\psi$

Several approaches, each with different trade-offs:

**Value-aware (successor-feature-like):** Train $\varphi_\psi$ so that $Q^+$ becomes approximately linear in features:

$$Q^+(x) \approx \mathbf{w}^T \varphi_\psi(x)$$

This directly shapes the metric to be value-relevant. If two configurations have similar features, they *should* have similar values — by construction. Connection to successor features (Barreto 2017): $\varphi_\psi$ is learned to predict cumulative discounted features, making $Q^+$ linear in the reward weights.

**Contrastive:** Treat transitions with similar returns as positive pairs:

$$\mathcal{L}_\text{contrastive} = -\log \frac{\exp(\varphi_\psi(x)^T \varphi_\psi(x^+) / \tau)}{\sum_j \exp(\varphi_\psi(x)^T \varphi_\psi(x_j^-) / \tau)}$$

This pushes configurations with similar values together and different values apart in feature space. Standard SimCLR/MoCo machinery applies.

**Bellman-consistent:** TD loss on $Q^+$ propagates through $\varphi_\psi$:

$$\mathcal{L}_\text{TD} = \left[Q^+_\psi(x_t) - (r_t + \gamma \max_{\theta'} Q^+_\psi(s_{t+1}, \theta'))\right]^2$$

where $Q^+_\psi(x) = \sum_i w_i \langle \varphi_\psi(x), \varphi_\psi(x_i) \rangle$. Gradients flow through $\varphi_\psi$, shaping the metric to minimize TD error. This is closest to standard deep RL — the kernel adapts to make the value function easier to represent.

**Anti-collapse regularization:** All learned metric approaches risk representation collapse — $\varphi_\psi$ mapping everything to the same point minimizes many losses trivially. Regularizers: VICReg (variance + invariance + covariance), spectral normalization, or the observation that particles with opposite-sign weights provide a natural anti-collapse signal (if $w_i > 0$ and $w_j < 0$, mapping $x_i$ and $x_j$ to the same feature would make $Q^+$ trivially zero).

---

## 6. The Deep Kernel Learning Connection

The approach described above is a specific instance of **Deep Kernel Learning** (Wilson, Hinton, Salakhutdinov 2016): use a neural network as a feature extractor, and place a GP on top of the learned features.

$$k_\psi(x, x') = k_\text{base}(\varphi_\psi(x), \varphi_\psi(x'))$$

where $k_\text{base}$ is a standard kernel (e.g., RBF) applied in the learned feature space. This is slightly more general than the pure inner-product form — it allows a non-trivial base kernel on top of the features.

In the GRL context:

$$Q^+(x) = \sum_i w_i \, k_\text{base}(\varphi_\psi(x), \varphi_\psi(x_i))$$

The neural network handles the representation learning (mapping raw state-action pairs to meaningful features), and the kernel handles the field construction (superposition, interpolation, uncertainty). Each component does what it's best at.

---

## 7. Operator Algebra in Feature Space

This is where the learned kernel story connects back to the action operator formalization from Part 2.

### 7.1 The Equivariance Condition

If we have an action operator $\hat{O}_\theta: \mathcal{S} \to \mathcal{S}$, and a learned feature map $\varphi_\psi$, we can ask: **how does the operator act in feature space?**

If we can find a function $f_\theta$ such that:

$$\varphi_\psi(\hat{O}_\theta(s)) \approx f_\theta(\varphi_\psi(s))$$

then $f_\theta$ is the **feature-space representation** of the operator. The operator acts on states in $\mathcal{S}$; $f_\theta$ acts on features in $\mathbb{R}^D$.

### 7.2 Why This Matters

If $f_\theta$ is *simple* (e.g., linear: $f_\theta(z) = A_\theta z + b_\theta$), then the operator has a clean algebraic structure in feature space, even if it's nonlinear in state space. This means:

- **Composition is easy:** $f_{\theta_2} \circ f_{\theta_1}$ is just matrix multiplication (for linear $f$)
- **Value prediction is easy:** if $Q^+$ is linear in features and $f$ is linear, then $Q^+(s') = Q^+(\hat{O}_\theta(s)) \approx \mathbf{w}^T A_\theta \varphi_\psi(s) + \mathbf{w}^T b_\theta$
- **Planning is easy:** multi-step rollouts in feature space are iterated linear maps

### 7.3 Training for Equivariance

We can add an equivariance loss to the training objective:

$$\mathcal{L}_\text{equiv} = \|\varphi_\psi(\hat{O}_\theta(s)) - f_\theta(\varphi_\psi(s))\|^2$$

This encourages the feature map to "linearize" the operator. The learned representation makes the nonlinear dynamics *appear* linear — a learned analogue of Koopman theory, where a nonlinear dynamical system is represented as a linear operator on a lifted function space.

### 7.4 Connection to Koopman Theory

The Koopman operator is an infinite-dimensional linear operator that acts on *observables* (functions of state) rather than states:

$$\mathcal{K} g = g \circ \hat{O}$$

for any observable $g: \mathcal{S} \to \mathbb{R}$. The feature map $\varphi_\psi$ can be viewed as a finite-dimensional approximation to the Koopman eigenfunctions. If $\varphi_\psi$ is chosen such that the equivariance condition holds with linear $f_\theta$, then $\varphi_\psi$ spans a Koopman-invariant subspace.

This provides a principled bridge: **GRL0's RKHS ↔ learned features ↔ Koopman theory ↔ operator algebra**.

---

## 8. The Hybrid Architecture: Putting It Together

Combining Parts 1-3, the complete architecture looks like:

```
State s
  │
  ├── Operator policy π_ψ(s) → θ           [Definition 3 from Part 2]
  │
  ├── Operator generator Φ(θ, s) → Ô_θ     [Definition 2 from Part 2]
  │
  ├── State transition: s' = Ô_θ(s) + ξ    [Definition 1 from Part 2]
  │
  ├── Feature extraction: φ_ψ(s, θ) → z     [Learned kernel from Part 3]
  │
  ├── Value field: Q⁺(s,θ) = Σ wᵢ k(z, zᵢ) [GRL0 particle memory]
  │
  └── Reinforcement field: ∇_θ Q⁺          [GRL0 gradient field]
```

**What comes from GRL0:** Particle memory, kernel superposition, reinforcement field as gradient, energy landscape interpretation.

**What comes from Part 2:** Action operators, operator generator, operator policy, algebraic structure, energy regularization.

**What comes from learned kernels (Part 3):** Adaptive similarity metric, task-specific feature extraction, equivariant representations, Koopman connection.

The three tracks are complementary:

- GRL0 provides the *philosophy*: RL as field theory, experience as particles, policy as geometry
- The operator framework provides the *execution*: formal action definition, differentiable pipeline, algebraic structure
- Learned kernels provide the *representation*: adaptive similarity, scalable features, deep learning integration

---

## 9. Open Questions

### Convergence

Alternating between $\psi$ (feature map) and $w$ (particle weights) isn't guaranteed to converge. The optimization landscape changes every time $\psi$ updates, potentially destabilizing previously well-fitted particles. EMA targets (à la DQN/BYOL) or anchor losses on old particles may help.

### Particle Memory Under Representation Drift

When $\varphi_\psi$ changes, the similarity between *all* particle pairs changes. A particle that was once a close neighbor of the query might become distant, or vice versa. Options:

- **Re-embed at query time:** Store raw $(s_i, \theta_i)$, compute $\varphi_\psi(x_i)$ fresh each time. Correct but expensive.
- **Periodic re-embedding:** Update cached features every $N$ steps.
- **Anchor regularization:** Penalize $\|\varphi_\psi(x_i) - \varphi_{\psi_\text{old}}(x_i)\|$ on stored particles.

### Uncertainty Under Learned Kernels

Fixed RBF gives clean uncertainty: novel inputs are far from all particles in feature space, so the GP posterior variance is high. A learned kernel can map novel inputs to *any* location in feature space — potentially right next to existing particles — creating false confidence.

Solutions: GP head on learned features (exactly Wilson 2016), deep ensembles of $\varphi_\psi$, density estimation in feature space.

### Spectral Clustering with Learned Kernels

GRL0's spectral clustering on the Gram matrix $K$ can be applied to the *learned* Gram matrix $K_\psi$. The eigenstructure would then reflect learned similarity rather than Euclidean proximity. This could provide much richer behavioral modes — but the eigendecomposition would need to be recomputed as $\psi$ evolves.

A natural bridge: use spectral eigenvectors from a fixed kernel as *initialization* for $\varphi_\psi$, then fine-tune. This starts with GRL0's prior and relaxes it.

---

## 10. Summary: The Full Arc

| Stage | Representation | Feature Map | Limitation |
|-------|----------------|-------------|------------|
| Classical RL | $Q(s, a)$, discrete $a$ | None | No action structure |
| GRL0 | $Q^+(s, \theta)$, GP + fixed kernel | $\varphi$ fixed by kernel choice | Discretization of $\Theta$; frozen metric |
| GRL + operators | $Q(s, \theta)$, neural + operator $\Phi(\theta, s)$ | Not applicable (no kernel) | Loses particle/field semantics |
| GRL + learned kernel | $Q^+(s, \theta) = \sum w_i k_\psi(x, x_i)$ | $\varphi_\psi$ learned | Non-stationarity; collapse risk |
| Full hybrid | Operators + learned particle field | $\varphi_\psi$ with equivariance | Open research |

The progression is:

1. **GRL0:** Right philosophy (parametric actions, energy fields, particles), wrong execution mechanism (discretization)
2. **Part 2 (operator framework):** Right execution mechanism (action operators), different value representation (neural, not particle-based)
3. **Learned kernels:** Bridges the two — keeps GRL0's particle/field semantics while enabling the operator framework and adaptive representations

The full hybrid — operator-based actions with learned-kernel particle memory and equivariant feature maps — is the research frontier, with a minimum viable experiment still being scoped.

---

**Related documents:**
- Part 1: [The GRL0 Gap](./01-the-grl0-gap.md)
- Part 2: [Action Operator Formalization](./02-action-operator-formalization.md)
- [GRL0: Learning the Field Beyond GP](../GRL0/quantum_inspired/07-learning-the-field-beyond-gp.md)
- Functional fields notebook: `notebooks/field_series/02_functional_fields.ipynb`
