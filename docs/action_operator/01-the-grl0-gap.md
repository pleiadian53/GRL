# Part 1: The GRL0 Gap — From Parametric Actions to the Operator Imperative

**Series:** Action Operators in Generalized Reinforcement Learning  
**Part:** 1 of 3  
**Audience:** Self-reference and interview prep  
**Prerequisites:** Familiarity with standard RL (MDP, Q-learning), basic functional analysis

---

## 1. What GRL0 Got Right

The original GRL paper (GRL0) introduced a radical departure from classical RL. Rather than treating actions as opaque symbols drawn from a finite set $\mathcal{A} = \{a_1, \ldots, a_k\}$, GRL0 proposed **parametric actions**: continuous parameters $\theta \in \Theta \subseteq \mathbb{R}^d$ that characterize a transformation of the state.

This was the core conceptual move:

| Aspect | Classical RL | GRL0 |
|--------|-------------|------|
| Action representation | Discrete symbol $a \in \mathcal{A}$ | Continuous parameter $\theta \in \Theta$ |
| Action space structure | Unstructured finite set | Continuous manifold |
| Value function domain | $Q(s, a)$ over state × index | $Q^+(s, \theta)$ over augmented space |
| Similarity between actions | Undefined | Kernel-induced: $k(\theta, \theta')$ |

The parametric action view opened up several innovations that were genuinely ahead of their time:

**1. Augmented State Space.** By forming $x = (s, \theta) \in \mathcal{X} = \mathcal{S} \times \Theta$, GRL0 treated state-action pairs as points in a joint continuous space. This enabled geometric reasoning about RL — actions weren't isolated choices but coordinates in a landscape.

**2. Energy Function.** The value field $Q^+(x) = Q^+(s, \theta)$ over augmented space could be re-read as an *energy landscape* via $E(x) = -Q^+(x)$ — the energy-based-model (EBM) convention in which **low energy = good**. Low-energy configurations (equivalently, high value) form basins of attraction; high-energy regions form barriers. The probabilistic reading $p(x) \propto e^{-E(x)}$ makes the intuition precise: low-energy states are plausible/certain, high-energy states implausible/uncertain.

**3. Reinforcement Field.** The negative gradient $-\nabla_x E(x)$ defined a vector field — the *reinforcement field* (a force field) — over augmented space. Policy improvement became gradient descent on energy: follow the field downhill toward lower-energy (higher-value) configurations. This collapsed value estimation, policy improvement, and exploration into a single geometric object.

**4. Experience Particles.** Transitions were stored as particles $(x_i, w_i)$ in augmented space, with the value function reconstructed via kernel superposition:
$$Q^+(x) = \sum_i w_i \, k(x, x_i)$$

**5. RKHS as Native Workspace.** Everything lived in a reproducing kernel Hilbert space. The GP posterior mean was guaranteed to be in RKHS. Functional gradients, inner products, and Riesz representers all applied rigorously.

These ideas anticipated energy-based models, score-based methods, neural fields, and diffusion-based policies by years.

---

## 2. Where GRL0 Hit a Wall

Despite the rich continuous machinery, GRL0 had to make a critical compromise at the point of *action execution*. The parametric action $\theta$ told the agent *where in parameter space* a good action lived, but not *what the action does to the state*.

The gap can be stated precisely:

> **GRL0 defined the value landscape $E(s, \theta)$ over parametric actions, but never formalized the mapping $\theta \mapsto s'$ — how action parameters transform the state.**

Without this mapping, GRL0 couldn't close the loop. The reinforcement field could point toward a region of $\Theta$ with high value, but executing an action still required the environment to interpret $\theta$. In practice, this forced two compromises:

### Compromise 1: Discretization of $\Theta$

To make the problem tractable with existing RL algorithms (SARSA, Q-learning), the continuous parameter space $\Theta$ had to be partitioned into bins:

$$\Theta \approx \{\theta_1, \theta_2, \ldots, \theta_m\}$$

The RF-SARSA algorithm (Section IV of the original paper) had a two-layer architecture:

- **Layer 1 (Primitive):** Standard SARSA on discrete state-action pairs — provides TD signals
- **Layer 2 (Field):** GP regression over augmented space — provides spatial generalization

But the primitive layer operated on a *discretized* action set. The continuous parameter space was partitioned, and Q-values were maintained for each partition. The field layer generalized across partitions via the kernel, but the control policy ultimately selected from finite options.

This defeated the purpose. The whole motivation was to escape discrete action sets, but the execution mechanism forced a return to them.

### Compromise 2: Environment-Mediated Transitions

Even when $\theta$ was interpreted continuously, the transition was still:

$$s' = T(s, \theta) \quad \text{(environment-defined)}$$

The agent had no say in *how* $\theta$ transforms $s$. The environment remained a black box that consumed action parameters and produced next states. The agent's role was still restricted to *selecting* parameters, not *constructing* transformations.

### The Structural Problem

The root cause was architectural. GRL0 had two powerful components that didn't connect:

```
Value side:         E(s, θ) = −Q⁺(s, θ) → −∇E → reinforcement field → "which θ is good"
Execution side:     θ → ??? → s'  (undefined / environment-dependent)
```

The value function knew which $\theta$ values were desirable at each state. But there was no formal mechanism to translate $\theta$ into a state transformation. The gap between "knowing what's good" and "doing what's good" was bridged only by heuristic discretization.

---

## 3. What Was Missing: Three Conceptual Pieces

Looking back, three things needed to be formalized to close the gap:

### Piece 1: Action as Transformation (Not Parameter)

GRL0 parameterized actions but didn't define what an action *is* functionally. The key insight: an action should be a **function on state space**:

$$\hat{O}: \mathcal{S} \to \mathcal{S}$$

Not a parameter that the environment interprets, but a transformation that the agent constructs and applies. The transition becomes:

$$s' = \hat{O}(s) + \xi$$

where $\xi$ is environmental noise. The agent *constructs* the transition; the environment provides only stochastic perturbation.

### Piece 2: Parameter-to-Operator Mapping (The Generator)

Once we define actions as operators, we need a formal bridge from parameters to operators:

$$\Phi: \Theta \times \mathcal{S} \to \mathcal{S}$$

For fixed $\theta$, this gives a specific operator $\hat{O}_\theta(s) = \Phi(\theta, s)$. This is the missing link — the **operator generator** — that converts GRL0's parametric actions into executable state transformations.

Different choices of $\Phi$ yield different operator families:
- $\Phi(\theta, s) = A_\theta s + b_\theta$ → affine operators
- $\Phi(\theta, s) = s + F_\theta(s)$ → field operators
- $\Phi(\theta, s) = s - \eta \nabla \phi_\theta(s)$ → potential operators

### Piece 3: Differentiable End-to-End Pipeline

With the generator in place, the full chain becomes differentiable:

$$s \xrightarrow{\pi_\psi} \theta \xrightarrow{\Phi} \hat{O}_\theta \xrightarrow{\text{apply}} s' = \hat{O}_\theta(s) + \xi$$

No discretization needed. Gradients flow from the reward signal through the transition, through the operator generator, back to the policy parameters $\psi$. The reinforcement field and the execution mechanism are unified.

---

## 4. The Proto-Operator in GRL0

It's worth noting that GRL0 *almost* had the operator concept. The original paper briefly mentioned "action operator" in the context of a matrix parameterization:

$$s' = A_\theta \cdot s$$

where $A_\theta$ is a matrix determined by action parameters. This is exactly the affine operator family (with $b = 0$). But it was introduced as an aside rather than developed as the central formalism. The paper didn't:

- Define a general operator space $\mathcal{O}$
- Establish algebraic structure (composition, identity, energy)
- Show how different operator families arise from different choices of $\Phi$
- Connect the operator formalism to the reinforcement field
- Prove that classical RL is a special case

The matrix multiplication was a specific instantiation, not a general framework. The leap from "actions can be parameterized as matrices" to "actions *are* operators on state space, and here is the full mathematical structure" is what Paper A accomplishes.

---

## 5. The GP-Based Spectral Clustering Connection

One of GRL0's underappreciated contributions was using GP-based spectral clustering to discover structure in the augmented space. The mechanism:

1. Fix kernel $k$, fit GP posterior $Q^+$ to particles
2. Compute the Gram matrix $K_{ij} = k(x_i, x_j)$ over accumulated particles
3. Eigendecompose $K$ to find coherent subspaces

The Gram matrix acts as a **neighborhood graph induced by the kernel**. If $k(x, x') \approx 1$, the two configurations are "neighbors" — they should have similar values. If $k(x, x') \approx 0$, they're independent. The kernel defines what "similar" means in augmented space.

The eigendecomposition discovers **functional subspaces** — groups of configurations that share similar value structure. This is post-hoc feature discovery: the kernel is fixed, but the spectral decomposition reveals latent structure.

This is deeply connected to the feature map interpretation. The Gram matrix $K$ can be written as:

$$K_{ij} = k(x_i, x_j) = \langle \varphi(x_i), \varphi(x_j) \rangle$$

The eigendecomposition of $K$ is equivalent to PCA in the feature space defined by $\varphi$. The top eigenvectors are the principal directions of variation — the most informative features for distinguishing augmented-state configurations.

But the feature map $\varphi$ is fixed by the choice of kernel. An RBF kernel imposes a specific notion of similarity (Euclidean distance scaled by lengthscale). If the true structure of the value function doesn't align with this similarity — e.g., if value depends on nonlinear combinations of state and action features — the fixed kernel will underfit.

This sets up the transition to Part 3, where we replace the fixed $\varphi$ with a learned $\varphi_\psi$.

---

## 6. Summary: The Three Gaps and Their Solutions

| Gap in GRL0 | Consequence | Solution (Paper A) |
|-------------|-------------|---------------------|
| No formal $\theta \mapsto s'$ mapping | Had to discretize $\Theta$ | **Operator Generator** $\Phi(\theta, s)$ |
| Actions interpreted by environment | Agent can't construct transitions | **Action Operators** $\hat{O}: \mathcal{S} \to \mathcal{S}$ |
| Fixed kernel limits similarity | Can't learn task-specific features | **Learned kernels** $k_\psi$ (Part 3) |

The first two gaps are closed by the action operator formalization (Part 2). The third gap — moving from fixed to learned kernels — opens a further extension that bridges GRL0's RKHS machinery to modern deep learning (Part 3).

---

**Next:** [Part 2: Action Operator Formalization](./02-action-operator-formalization.md) — the three definitions, concrete operator families, algebraic structure, and how they close the gap.

---

**Related documents:**
- GRL0 baseline concepts: `Projects/GRL/dev/GRL0/01-grl-baseline-concepts.md`
- Paper A formal definitions: `Projects/GRL/dev/papers/paper-a-theory/02-paper-a-formal-definitions.md`
- Reinforcement field (Section III): `Projects/GRL/dev/GRL0/reinforcement_field/01-RF-section-3.md`
- Learned kernels scope: `Projects/GRL/dev/GRL_extensions/learned_kernels/00-scope.md`
