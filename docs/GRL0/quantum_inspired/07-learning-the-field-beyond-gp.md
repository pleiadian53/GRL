# Chapter 7: Learning the Reinforcement Field — Beyond Gaussian Processes

## Motivation

In [Chapter 6](06-agent-state-and-belief-evolution.md), we established that the agent's state is the reinforcement field $Q^+ \in \mathcal{H}_k$, and MemoryUpdate is the operator that evolves this state.

But **how exactly should we update the field given new experience?**

The original GRL paper uses a Gaussian Process (GP) perspective, where weights $w_i$ emerge from kernel-based inference. But **GP is not the only option!**

This chapter explores:

1. **Why GP is one choice among many** for learning $Q^+$
2. **Alternative learning mechanisms** (kernel ridge, online optimization, sparse methods, deep nets)
3. **When to use which approach** (trade-offs in scalability, sparsity, adaptivity)
4. **Amplitude-based learning** from quantum-inspired probability

**Key insight:** The state-as-field formalism is **agnostic to the learning mechanism**—you can swap the inference engine while preserving GRL's structure.

---

## 1. The Learning Problem: What Are We Optimizing?

### The Core Question

Given:

- Current field: $Q^+_t = \sum_i w_i^{(t)} k(z_i^{(t)}, \cdot)$
- New experience: $(s_t, a_t, r_t)$ (or TD target $y_t$)

**Find:** Updated field $Q^+_{t+1}$

**Constraint:** $Q^+_{t+1}$ should:

- Fit the new evidence
- Generalize via kernel structure
- Remain bounded/stable

---

### The Objective (General Form)

**Most learning mechanisms solve:**

$$\min_{Q^+ \in \mathcal{H}_k} \underbrace{\mathcal{L}(Q^+, \mathcal{D}_{t+1})}_{\text{fit data}} + \underbrace{\mathcal{R}(Q^+)}_{\text{regularization}}$$

where:

- $\mathcal{D}_{t+1} = \{(z_i, y_i)\}_{i=1}^{N+1}$ is accumulated experience
- $\mathcal{L}$ measures prediction error
- $\mathcal{R}$ controls complexity

**Different choices give different algorithms!**

---

## 2. Method 1: Gaussian Process Regression (Original GRL)

### The GP Perspective

**Model:** $Q^+(z) \sim \mathcal{GP}(\mu(z), k(z, z'))$

**Posterior mean after observing data:**

$$Q^+(z) = \sum_{i=1}^N \alpha_i k(z_i, z)$$

where $\boldsymbol{\alpha} = (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$

**Properties:**
- ✅ Probabilistic (gives uncertainty)
- ✅ Kernel-based (automatic generalization)
- ✅ Theoretically grounded
- ❌ Requires matrix inversion ($O(N^3)$)
- ❌ Full covariance matrix ($O(N^2)$ memory)

---

### When GP Makes Sense

**Good for:**
- Small-to-medium particle sets ($N < 10^4$)
- Batch updates (re-solve periodically)
- When you need calibrated uncertainty

**Not ideal for:**
- Large-scale online learning
- Extremely sparse solutions
- Real-time embedded systems

---

## 3. Method 2: Kernel Ridge Regression (Deterministic Cousin)

### The Regularized Least Squares View

**Objective:**

$$\min_{\mathbf{w}} \|\mathbf{K}\mathbf{w} - \mathbf{y}\|^2 + \lambda \mathbf{w}^T \mathbf{K} \mathbf{w}$$

**Solution:**

$$\mathbf{w} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$$

**Result:**

$$Q^+(z) = \sum_{i=1}^N w_i k(z_i, z)$$

---

### Comparison to GP

| Aspect | Gaussian Process | Kernel Ridge |
|--------|------------------|--------------|
| **Framework** | Probabilistic | Deterministic |
| **Solution** | $(\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$ | $(\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$ |
| **Uncertainty** | Yes (full covariance) | No |
| **Computation** | Same ($O(N^3)$) | Same ($O(N^3)$) |
| **Interpretation** | Posterior mean | Regularized fit |

**Practically equivalent for point predictions!**

**When to use:** If you don't need uncertainty estimates, kernel ridge is simpler.

---

## 4. Method 3: Online Convex Optimization (Incremental Updates)

### The Online Learning View

Instead of batch re-solving, **update weights incrementally:**

$$w_i^{(t+1)} = w_i^{(t)} - \eta_t \nabla_{w_i} \mathcal{L}_t$$

where $\mathcal{L}_t$ is loss on current experience.

---

### Stochastic Gradient Descent on Weights

**For TD learning:**

$$\mathcal{L}_t = \frac{1}{2} [Q^+_t(z_t) - y_t]^2$$

where $y_t = r_t + \gamma \max_{a'} Q^+_t(s_{t+1}, a')$ (TD target).

**Gradient:**

$$\nabla_{w_i} \mathcal{L}_t = [Q^+_t(z_t) - y_t] \cdot k(z_i, z_t)$$

**Update:**

$$w_i^{(t+1)} = w_i^{(t)} - \eta_t [Q^+_t(z_t) - y_t] k(z_i, z_t)$$

---

### Properties

✅ **Online:** No matrix inversion
✅ **Scalable:** $O(N)$ per update
✅ **Flexible:** Can use different loss functions (Huber, quantile)
❌ **No closed form:** Requires tuning learning rate $\eta_t$
❌ **Stability:** May need momentum, adaptive rates

---

### When to Use Online Updates

**Good for:**
- Large-scale continuous learning
- Non-stationary environments
- Real-time systems
- When batch re-solving is too expensive

**Example:** RL with millions of experiences—batch GP infeasible, online SGD works.

---

## 5. Method 4: Sparse Methods (Compact Memory)

### The Sparsity Objective

**Goal:** Learn $Q^+$ with **few active particles** (small $\|\mathbf{w}\|_0$)

**LASSO in kernel space:**

$$\min_{\mathbf{w}} \|\mathbf{K}\mathbf{w} - \mathbf{y}\|^2 + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \mathbf{w}^T \mathbf{K} \mathbf{w}$$

$\ell_1$ penalty encourages sparse $\mathbf{w}$ (many $w_i = 0$).

---

### Why Sparsity Matters

**Particle memory grows unbounded** without pruning:

- Every experience → new particle
- Memory: $O(N)$
- Computation: $O(N)$ per query

**With sparsity:**
- Active set: $\{i : w_i \neq 0\}$ is small
- Memory: $O(k)$ where $k \ll N$
- Computation: $O(k)$ per query

---

### Sparse GP Variants

**Inducing point methods:**
- Select $M$ representative particles (inducing points)
- Approximate $Q^+$ using only these $M$ particles
- Solve $(M \times M)$ system instead of $(N \times N)$

**Example: FITC, PITC, VFE**

**Compression ratio:** $M/N$ (e.g., 100 inducing points for 10,000 experiences)

---

### When to Use Sparse Methods

**Good for:**
- Memory-constrained systems (robots, embedded)
- Lifelong learning (unbounded experience streams)
- When most particles are redundant

**Trade-off:**
- Faster queries
- Less memory
- Slight approximation error

---

## 6. Method 5: Deep Neural Networks (Function Approximation)

### Replacing Kernels with Neural Nets

**Standard GRL:**

$$Q^+(z) = \sum_i w_i k(z_i, z)$$

**Neural GRL:**

$$Q^+(z) = Q_\theta(z)$$

where $Q_\theta$ is a deep neural network with parameters $\theta$.

---

### How to Train

**Objective:** Same TD learning

$$\mathcal{L} = \mathbb{E}[(Q_\theta(s, a) - y)^2]$$

where $y = r + \gamma \max_{a'} Q_\theta(s', a')$

**Update:** Gradient descent on $\theta$

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

---

### Hybrid: Neural Net + Particle Memory

**Keep the best of both:**

**Long-term memory (neural):**

$$Q_{\text{base}}(z) = Q_\theta(z)$$

**Short-term memory (particles):**

$$Q_{\text{episodic}}(z) = \sum_{i \in \text{recent}} w_i k(z_i, z)$$

**Combined:**

$$Q^+(z) = Q_\theta(z) + \beta \sum_{i \in \text{recent}} w_i k(z_i, z)$$

**Why this works:**
- Neural net: captures long-term patterns
- Particles: fast adaptation to new experiences
- Combination: benefits of both

---

### When to Use Deep Nets

**Good for:**
- High-dimensional states (images, text)
- When kernel methods don't scale
- Transfer learning (pre-trained representations)

**Not ideal for:**
- Low-data regimes (kernels better)
- When interpretability matters (particles clearer)
- Embedded systems (large models)

---

## 7. Method 6: Mixture of Experts (Multiple Local Fields)

### The Idea

Instead of one global field, maintain **multiple local fields** with a gating function:

$$Q^+(z) = \sum_{m=1}^M g_m(z) Q_m(z)$$

where:

- $Q_m(z)$ = expert $m$'s field
- $g_m(z)$ = gate (probability expert $m$ is active)
- $\sum_m g_m(z) = 1$

---

### How to Partition

**Option A: Geometric partition (augmented space)**

Gate based on kernel similarity to expert centers:

$$g_m(z) \propto \exp(\gamma \, k(z, c_m))$$

where $c_m$ is the center of expert $m$.

**Find centers via:**
- K-means clustering in augmented space
- Spectral clustering (align with concepts!)
- Online clustering as experiences arrive

---

**Option B: Concept-based partition (function space)**

Use concept subspaces from Chapter 5:

$$Q_m = P_{\mathcal{C}_m} Q^+$$

Gate by concept activation:

$$g_m(z) \propto \|P_{\mathcal{C}_m} k(z, \cdot)\|^2$$

**Interpretation:** "How much does query $z$ belong to concept $m$?"

**This is extremely elegant:** partition function space, not state space!

---

**Option C: State vs. action partition**

**State-based experts:** $g_m(z)$ depends only on $s$
- Good for environments with distinct regimes/modes
- Example: different rooms, different game phases

**Action-based experts:** $g_m(z)$ depends only on $a$
- Good for tool-using agents
- Example: "search expert," "calculator expert," "database expert"

**Joint experts:** $g_m(z)$ depends on $(s, a)$
- Most general, captures "this action works in this situation"
- Aligns with GRL's augmented space philosophy

---

### Training Mixture of Experts

**Option 1: Hard assignment**
- Assign each particle to one expert
- Train experts independently

**Option 2: Soft assignment (EM-style)**
- E-step: Compute gates $g_m(z)$
- M-step: Update each expert's weights with gate-weighted data

**Option 3: Joint training**
- Train gates and experts together
- Use sparse MoE (only activate top-k experts)

---

### When to Use MoE

**Good for:**
- Complex environments with distinct modes
- When single field is too simple
- Tool-using/API-calling agents
- Hierarchical decision-making

**Example:**
- Expert 1: Navigation in open space
- Expert 2: Navigation through doors
- Expert 3: Navigation in crowds
- Gate routes based on context

---

## 8. Amplitude-Based Learning (Quantum-Inspired)

### The Probability Amplitude View

**Standard ML:** Model probability directly

$$p(x) = f_\theta(x)$$

**Amplitude-based:** Model amplitude, derive probability

$$\psi(x) = f_\theta(x), \quad p(x) = |\psi(x)|^2$$

---

### Why This Is Interesting

**Allows interference:**
- Amplitudes can be negative or complex
- Combine amplitudes before squaring
- Constructive/destructive interference

**Example:**

$$\psi_{\text{total}}(x) = \psi_1(x) + \psi_2(x)$$

$$p(x) = |\psi_1(x) + \psi_2(x)|^2 \neq |\psi_1(x)|^2 + |\psi_2(x)|^2$$

**This captures correlations that direct probability models miss!**

---

### Existing Work

**Born Machines (Cheng et al. 2018):**
- Use quantum circuits to generate $\psi(x)$
- Sample from $p(x) = |\psi(x)|^2$
- For generative modeling

**Complex-valued neural networks:**
- Amplitude + phase
- Used in signal processing, audio, RF

**Quantum-inspired models:**
- Quantum cognition (decision theory)
- Quantum probability theory
- Order effects, context effects

---

### For GRL: Amplitude-Based Value Functions

**Proposal:** Treat $Q^+$ as an amplitude

$$Q^+(z) \in \mathbb{C}$$

**Policy from Born rule:**

$$\pi(a|s) \propto |Q^+(s, a)|^2$$

**Why interesting:**
- **Phase** can encode temporal structure, direction, context
- **Interference** between action options
- **Coherence** measures correlation strength

**This approach is explored in detail in [Chapter 3: Complex-Valued RKHS](03-complex-rkhs.md).**

---

### Research Opportunity

**What's underexplored:**

Amplitude-based **reinforcement learning** — treating value/policy as amplitude fields with:

- Interference effects
- Phase semantics
- Born rule for action selection

**GRL is positioned to explore this!**

---

## 9. Comparison Summary

| Method | Scalability | Uncertainty | Sparsity | Complexity |
|--------|-------------|-------------|----------|------------|
| **GP** | $O(N^3)$ | Yes | No | Low (matrix inversion) |
| **Kernel Ridge** | $O(N^3)$ | No | No | Low |
| **Online SGD** | $O(N)$ | No | No | Medium (tuning $\eta$) |
| **Sparse GP** | $O(M^3)$ | Yes | Yes | Medium (select inducing points) |
| **Deep NN** | $O(1)$ query | No | Implicit | High (architecture, hyperparams) |
| **MoE** | $O(M \cdot N/M)$ | Depends | Per expert | High (gating + experts) |
| **Amplitude** | Depends | Special | Depends | High (complex math) |

---

## 10. Decision Guide: Which Method to Use?

### For GRL v0 (Baseline)

**Recommendation:** Kernel ridge or sparse GP
- Maintains RKHS structure
- Computationally tractable
- Well-understood theory

---

### For Large-Scale Applications

**Recommendation:** Online SGD or hybrid (NN + particles)
- Scalable to millions of experiences
- Can use modern deep learning infrastructure
- Trade interpretability for capacity

---

### For Embedded/Real-Time Systems

**Recommendation:** Sparse methods
- Bounded memory: $O(M)$ with $M$ small
- Fast queries: $O(M)$
- Pruning/merging for adaptation

---

### For Research/Novel Contributions

**Recommendation:** MoE with concept-based gating or amplitude-based learning
- MoE: natural connection to hierarchical RL
- Amplitude: novel probability formulation for ML
- Both: strong theoretical foundations

---

## Summary

### Key Insights

1. **Learning $Q^+$ is a choice**
   - GP is elegant but not unique
   - Many alternatives preserve GRL structure

2. **Trade-offs matter**
   - Scalability: online SGD, deep nets
   - Sparsity: LASSO, inducing points
   - Structure: MoE, concept-based

3. **State-as-field is agnostic**
   - Can swap learning mechanisms
   - Preserves: state = field, query = projection, update = operator

4. **Amplitude-based learning is underexplored**
   - QM math can be used for ML/optimization
   - GRL is positioned to pioneer this for RL

---

### Key Equations

**General learning objective:**

$$\min_{Q^+} \mathcal{L}(Q^+, \mathcal{D}) + \mathcal{R}(Q^+)$$

**Online weight update:**

$$w_i^{(t+1)} = w_i^{(t)} - \eta [Q^+_t(z_t) - y_t] k(z_i, z_t)$$

**Mixture of experts:**

$$Q^+(z) = \sum_m g_m(z) Q_m(z)$$

**Amplitude-based policy:**

$$\pi(a|s) \propto |Q^+(s, a)|^2$$

---

## Further Reading

### Within This Series

- **[Chapter 3](03-complex-rkhs.md):** Complex-Valued RKHS (amplitude with phase)
- **[Chapter 5](05-concept-projections-and-measurements.md):** Concept Subspaces (for MoE gating)
- **[Chapter 6](06-agent-state-and-belief-evolution.md):** State Evolution (general framework)

### GRL Tutorials

- **[Tutorial Chapter 6](../tutorials/06-memory-update.md):** MemoryUpdate Algorithm

### Related Literature

**Gaussian Processes:**
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*.

**Online Learning:**
- Shalev-Shwartz (2011). "Online Learning and Online Convex Optimization."

**Sparse Methods:**
- Quiñonero-Candela & Rasmussen (2005). "A Unifying View of Sparse Approximate Gaussian Process Regression."

**Born Machines:**
- Cheng et al. (2018). "Quantum Generative Adversarial Learning." *PRL*.

**Mixture of Experts:**
- Jacobs et al. (1991). "Adaptive Mixtures of Local Experts." *Neural Computation*.
- Shazeer et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."

---

**Last Updated:** January 14, 2026
