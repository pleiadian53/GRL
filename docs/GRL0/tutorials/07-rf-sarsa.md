# Chapter 7: RF-SARSA (Functional TD Learning)

**Purpose**: Understand how GRL learns the reinforcement field through temporal-difference updates  
**Prerequisites**: Chapters 4-6 (Reinforcement Field, Particle Memory, MemoryUpdate)  
**Key Concepts**: RF-SARSA algorithm, two-layer learning, functional TD, field reshaping, belief-conditioned control

---

## Introduction

We've built up the conceptual foundations:

- The reinforcement field $Q^+(z)$ as a functional object in RKHS (Chapter 4)
- Experience particles as basis elements (Chapter 5)
- MemoryUpdate as belief evolution (Chapter 6)

But we haven't yet addressed the fundamental question: **How does the agent learn $Q^+$ from experience?**

This chapter introduces **RF-SARSA** (Reinforcement Field SARSA), the core learning algorithm. The name might suggest "SARSA with kernels," but this would be misleading. RF-SARSA is fundamentally different:

|| Classical SARSA | RF-SARSA (GRL) |
|---|---|---|
| **Updates** | Scalar $Q(s,a)$ values | Particle weights defining global functional field |
| **Learning target** | State-action value | Geometry of the entire landscape |
| **Policy** | $\arg\max_a Q(s,a)$ | Field-based inference via energy minimization |
| **Generalization** | Via function approximation | Geometric propagation through kernels |

**Core insight**: RF-SARSA doesn't learn Q-values—it **reshapes the energy landscape** from which actions are inferred.

---

## 1. What Makes RF-SARSA Different?

### 1.1 Classical SARSA: A Reminder

Standard SARSA (State-Action-Reward-State-Action) updates Q-values using the temporal difference (TD) error:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma Q(s', a') - Q(s, a)\right]$$

**Interpretation**:

- **Target**: $r + \gamma Q(s', a')$ (one-step return using actual next action)
- **Error**: Difference between target and current estimate
- **Update**: Move current estimate toward target

**Key property**: On-policy learning (learns about the policy being executed).

---

### 1.2 RF-SARSA: Functional TD in RKHS

RF-SARSA maintains the TD learning principle but applies it to **basis functions** in RKHS, not to table entries.

**What changes**:

1. **Primary updates**: Particle weights $w_i$ that define the field $Q^+(z) = \sum_i w_i k(z, z_i)$
2. **Geometric propagation**: TD signal affects not just one location but a neighborhood (via kernel)
3. **Belief representation**: The field $Q^+$ (equivalently, particle memory $\Omega$) is the agent's state
4. **Action inference**: Policy emerges from field queries, not direct optimization

**Critical point**: RF-SARSA is **not** learning a policy—it's reshaping a landscape from which policy is inferred.

---

### 1.3 Two Learning Processes at Different Levels

RF-SARSA couples two learning processes:

**Primitive Layer** (Discrete):

- Maintains $Q(s, a)$ estimates over discrete state-action pairs
- Uses standard SARSA updates
- Provides **temporal grounding** (immediate feedback from environment)
- Supplies **reinforcement signals** (TD errors)

**Field Layer** (Continuous):

- Generalizes estimates over continuous augmented space $(s, \theta)$
- Uses Gaussian Process Regression (GPR) for interpolation
- Performs **policy inference** exclusively
- Receives **reinforcement** through experience particles

**Key relationship**:

- Primitive layer → generates TD evidence
- Field layer → performs action inference
- MemoryUpdate → connects them (Algorithm 1)

---

## 2. The RF-SARSA Algorithm (Informal Overview)

Before the formal specification, here's the conceptual flow:

**Initialization**:

1. Initialize kernel hyperparameters $\theta$ (e.g., via ARD on initial particles)
2. Initialize primitive $Q(s, a)$ arbitrarily
3. Initialize particle memory $\Omega$ (possibly empty)

**Each episode**:

1. Start in state $s_0$

**Each step**:

2. **Field-based action inference**:

   - For each candidate action $a^{(i)}$, form augmented state $z^{(i)} = (s, a^{(i)})$
   - Query field: $Q^+(z^{(i)}) = \mathbb{E}[Q^+ \mid \Omega, k, \theta]$ via GPR
   - Select action via policy: $a \sim \pi(a \mid s)$ based on $Q^+$ values
   
3. **Environment interaction**:

   - Execute $a$, observe reward $r$ and next state $s'$
   
4. **Next action inference**:

   - Repeat step 2 for state $s'$ to select $a'$
   
5. **Primitive SARSA update**:

   - Compute TD error: $\delta = r + \gamma Q(s', a') - Q(s, a)$
   - Update: $Q(s, a) \leftarrow Q(s, a) + \alpha \delta$
   
6. **Particle reinforcement** (Algorithm 1: MemoryUpdate):

   - Form particle: $\omega = (z, Q(s, a))$ where $z = (s, a)$
   - Update memory: $\Omega \leftarrow \text{MemoryUpdate}(\omega, \delta, k, \tau, \Omega)$
   
7. **Advance**: $s \leftarrow s'$, $a \leftarrow a'$

**Periodic**:

- Every $T$ steps, re-estimate kernel hyperparameters $\theta$ via ARD

---

## 3. Formal Specification

### 3.1 Notation

**Environment**:

- $s \in \mathcal{S}$: primitive environment state
- $a^{(i)} \in \mathcal{A}$: discrete action ($i = 1, \ldots, n$)
- $r \in \mathbb{R}$: reward
- $\gamma \in [0, 1]$: discount factor

**Parametric action representation**:

- $M$: parametric action model
- $f_{A^+}: a^{(i)} \mapsto x_a^{(i)} \in \mathbb{R}^{d_a}$: action encoder
- $f_A: x_a \mapsto a$: action decoder

**Augmented space**:

- $x_s \in \mathbb{R}^{d_s}$: state features
- $x_a \in \mathbb{R}^{d_a}$: action parameters
- $z = (x_s, x_a) \in \mathbb{R}^{d_s + d_a}$: augmented state-action point

**Value functions**:

- $Q(s, a)$: primitive action-value (base SARSA learner)
- $Q^+(z)$: field-based value estimate (via GPR on $\Omega$)
- $E(z) := -Q^+(z)$: energy (lower is better)

**Memory and kernel**:

- $\Omega = \{(z_i, q_i)\}_{i=1}^N$: particle memory (GPR training set)
- $k(z, z'; \theta)$: kernel function with hyperparameters $\theta$
- $\tau \in [0, 1]$: association threshold

**Parameters**:

- $\alpha$: SARSA learning rate
- $T$: ARD update period
- $\beta$: policy temperature (for Boltzmann policy)

---

### 3.2 Algorithm: RF-SARSA

**Inputs**:

- Kernel function $k(\cdot, \cdot; \theta)$
- Parametric action model $M$
- ARD update period $T$
- Initial particle memory $\Omega_0$
- Association threshold $\tau$
- Learning rate $\alpha$, discount $\gamma$, policy temperature $\beta$

**Initialization**:

1. **Estimate kernel hyperparameters**:

   - If $\Omega_0 \neq \emptyset$: Run ARD on $\Omega_0$ to get $\theta$
   - Else: Initialize $\theta = \theta_0$ from prior

2. **Initialize primitive Q-function**:

   - $Q(s, a) \leftarrow 0$ for all $(s, a)$ (or small random values)

3. **Initialize particle memory**:

   - $\Omega \leftarrow \Omega_0$

4. **Initialize step counter**:

   - $t_{\text{ARD}} \leftarrow 0$

---

**For each episode**:

5. **Initialize state**:

   - Observe initial state $s_0$
   - Set $s \leftarrow s_0$

6. **Initial action selection**:

   - For each $a^{(i)} \in \mathcal{A}$:

     - Encode: $x_a^{(i)} \leftarrow f_{A^+}(a^{(i)})$
     - Form augmented: $z^{(i)} \leftarrow (x_s(s), x_a^{(i)})$
     - Field query: $Q^+(z^{(i)}) \leftarrow \text{GPR-Predict}(z^{(i)}; \Omega, k, \theta)$
   - Policy: $a \sim \pi(\cdot \mid s)$ where $\pi(a^{(i)} \mid s) \propto \exp(\beta Q^+(z^{(i)}))$

---

**For each step of episode**:

7. **Periodic kernel hyperparameter update**:

   - If $t_{\text{ARD}} \bmod T = 0$:

     - Re-run ARD on $\Omega$ to update $\theta$
     - (Optional) Increase $T$ to reduce update frequency over time
   - $t_{\text{ARD}} \leftarrow t_{\text{ARD}} + 1$

8. **Environment interaction**:

   - Execute action $a$
   - Observe reward $r$ and next state $s'$

9. **Next action inference** (field query):

   - For each $a'^{(j)} \in \mathcal{A}$:

     - Encode: $x_a'^{(j)} \leftarrow f_{A^+}(a'^{(j)})$
     - Form augmented: $z'^{(j)} \leftarrow (x_s(s'), x_a'^{(j)})$
     - Field query: $Q^+(z'^{(j)}) \leftarrow \text{GPR-Predict}(z'^{(j)}; \Omega, k, \theta)$
   - Policy: $a' \sim \pi(\cdot \mid s')$ where $\pi(a'^{(j)} \mid s') \propto \exp(\beta Q^+(z'^{(j)}))$

10. **Primitive SARSA update** (generates TD evidence):

    - Compute TD error:
      $$\delta \leftarrow r + \gamma Q(s', a') - Q(s, a)$$
    - Update primitive Q-function:
      $$Q(s, a) \leftarrow Q(s, a) + \alpha \delta$$
    - Store updated value: $q \leftarrow Q(s, a)$

11. **Particle reinforcement** (Algorithm 1: MemoryUpdate):

    - Form experience particle:
      $$\omega \leftarrow (z, q) \quad \text{where} \quad z = (x_s(s), f_{A^+}(a))$$
    - Update particle memory:
      $$\Omega \leftarrow \text{MemoryUpdate}(\omega, \delta, k, \tau, \Omega)$$
      (See Chapter 6 for MemoryUpdate details)

12. **State-action transition**:

    - $s \leftarrow s'$, $a \leftarrow a'$

13. **Termination check**:

    - If $s$ is terminal, end episode
    - Else, return to step 7

---

### 3.3 Helper Function: GPR-Predict

**Gaussian Process Regression prediction**:

Given particle memory $\Omega = \{(z_i, q_i)\}_{i=1}^N$, kernel $k$, and hyperparameters $\theta$, predict the field value at query point $z$:

$$Q^+(z) = \sum_{i=1}^N \alpha_i k(z, z_i; \theta)$$

where coefficients $\alpha = (\alpha_1, \ldots, \alpha_N)^\top$ solve:

$$(K + \sigma_n^2 I) \alpha = q$$

with:

- $K_{ij} = k(z_i, z_j; \theta)$: kernel matrix
- $q = (q_1, \ldots, q_N)^\top$: stored values
- $\sigma_n^2$: noise variance (hyperparameter)

**In practice**: Use efficient GPR libraries (e.g., GPyTorch, scikit-learn's `GaussianProcessRegressor`).

**Computational note**: For large $N$, use sparse GP approximations (e.g., inducing points, random features).

---

## 4. How RF-SARSA Works: The Three Forces

RF-SARSA succeeds by balancing three forces:

### 4.1 Temporal Credit Assignment (Primitive SARSA)

**Role**: Ground the field in actual experienced returns.

**Mechanism**: SARSA provides **temporally accurate** value estimates:

- TD error $\delta = r + \gamma Q(s', a') - Q(s, a)$ measures prediction error
- Bootstrapping from $Q(s', a')$ propagates future returns backward
- On-policy learning ensures values reflect the behavior policy

**Why this matters**: Without temporal grounding, the field would have no connection to true returns—it would generalize nonsense.

---

### 4.2 Geometric Generalization (GPR)

**Role**: Spread value information across similar configurations.

**Mechanism**: Kernel similarity defines "nearness":

- $k(z, z')$ large → $z$ and $z'$ are similar → should have similar $Q^+$ values
- GP regression interpolates smoothly between particles
- Predictions come with uncertainty estimates ($\sigma^2(z)$)

**Why this matters**: The agent visits a tiny fraction of augmented space—generalization is essential for learning.

---

### 4.3 Adaptive Geometry (ARD)

**Role**: Learn which dimensions matter.

**Mechanism**: Automatic Relevance Determination (ARD) adjusts kernel lengthscales:

- Large lengthscale → dimension is irrelevant → smooth over it
- Small lengthscale → dimension is critical → pay attention to variations

**Example**: In a reaching task, gripper orientation might be irrelevant initially but critical when grasping—ARD adapts.

**Why this matters**: The agent doesn't know a priori which action parameters are important—ARD discovers this from data.

---

### 4.4 The Virtuous Cycle

These three forces create a virtuous cycle:

```
1. SARSA updates primitive Q(s,a) using TD → accurate local estimates
2. Particles (z, Q(s,a)) added to Ω → new training data
3. GPR generalizes across Ω → smooth field Q+(z)
4. Policy queries Q+(z) → explores intelligently
5. New experiences → refine SARSA estimates
6. ARD adapts kernel → focuses on relevant dimensions
→ Loop back to step 1
```

**Key insight**: The field $Q^+$ is **not optimized**—it **emerges** from the interaction of these forces.

---

## 5. Connection to the Principle of Least Action

With Chapter 03a fresh in mind, we can now see RF-SARSA through the lens of physics.

### 5.1 RF-SARSA as Action Minimization

Recall from Chapter 03a that optimal trajectories minimize the action functional:

$$S[\tau] = \int_0^T \left[E(z_t) + \frac{1}{2\lambda}\|\dot{z}_t\|^2\right] dt$$

**RF-SARSA implements this**:

1. **Energy term**: $E(z) = -Q^+(z)$ is learned via TD updates
   - High $Q^+$ → low energy → good configuration
   - SARSA signals reshape $Q^+$ to reflect actual returns
   
2. **Kinetic term**: $\frac{1}{2\lambda}\|\dot{z}\|^2$ is implicitly encoded by the kernel
   - Smooth kernels (e.g., RBF) penalize rapid changes
   - Particles propagate weights to neighbors (Algorithm 1: MemoryUpdate)
   - Result: Smooth $Q^+$ landscape

3. **Policy**: Boltzmann distribution $\pi \propto \exp(Q^+/\lambda)$ minimizes expected action
   - Temperature $\lambda$ controls exploration
   - Sampling via Langevin dynamics (gradient flow on $Q^+$)

**The deep connection**: RF-SARSA is not an ad-hoc algorithm—it's **learning the energy landscape** from which the least-action principle determines optimal trajectories!

---

### 5.2 Why This Matters for Learning

**Smoothness as regularization**:

- RF-SARSA favors smooth $Q^+$ fields (via kernel smoothness)
- Equivalent to penalizing rapid action changes (kinetic term)
- Natural Occam's razor: simple policies preferred

**Physical intuition**:

- Particles are like "mass distributions" creating a potential landscape
- Agent follows "trajectories" that minimize action
- MemoryUpdate reshapes the landscape based on experience

**Modern perspective**:

- This is **energy-based learning** (Chapter 3)
- Policy emerges from **gradient flow** on learned energy (Langevin dynamics)
- RF-SARSA anticipates modern score-based / diffusion-based RL methods

---

## 6. Worked Example: 1D Navigation (Continued)

Let's extend the 1D navigation example from Chapter 6 to show how RF-SARSA learns.

### 6.1 Problem Setup

**Environment**:

- State $s \in [0, 10]$: position on a line
- Goal: $s = 10$ (reward $r = +10$)
- Obstacle: region $[4, 6]$ (reward $r = -5$ if entered)
- Actions: $a \in \{\text{left}, \text{right}\}$ with parametric "step size" $\theta \in [0, 1]$
  - left: $s' = s - 2\theta$
  - right: $s' = s + 2\theta$

**Augmented space**: $z = (s, \theta) \in [0, 10] \times [0, 1]$

**Kernel**: RBF kernel $k(z, z') = \exp(-\|z - z'\|^2 / (2\ell^2))$ with lengthscale $\ell = 0.5$

---

### 6.2 Initial State (Episode 1, Step 1)

**Agent at $s = 2$**:

**Primitive Q-function** (initialized to zero):

- $Q(2, \text{left}) = 0$
- $Q(2, \text{right}) = 0$

**Particle memory**: $\Omega = \emptyset$ (no particles yet)

**Field prediction** (no particles → default prior mean = 0):

- $Q^+(2, \text{left}, 0.5) = 0$
- $Q^+(2, \text{right}, 0.5) = 0$

**Action selection** (random, since all Q-values equal):

- Select $a = \text{right}$, $\theta = 0.5$

---

### 6.3 First Experience

**Execute action**:

- $s' = 2 + 2(0.5) = 3$, $r = 0$ (no reward yet)

**Next action** (random again):

- $a' = \text{right}$, $\theta' = 0.5$

**SARSA update** (assume $\gamma = 0.9$, $\alpha = 0.1$):
$$\delta = 0 + 0.9 \cdot 0 - 0 = 0$$
$$Q(2, \text{right}) = 0 + 0.1 \cdot 0 = 0$$

**Particle**: $\omega = ((2, 0.5), 0)$ (augmented state, updated Q-value)

**MemoryUpdate**:

- $\Omega$ is empty → create new particle
- $\Omega = \{((2, 0.5), 0)\}$

**Field now**: $Q^+(z) = 0 \cdot k(z, (2, 0.5))$ (still zero, but structure is building)

---

### 6.4 After Many Steps: Goal Reached

**Episode 1 concludes**: Agent reaches $s = 10$ after 5 steps, receives $r = +10$.

**Final step SARSA update**:
$$\delta = 10 + 0 - 0 = 10 \quad (\text{terminal, so } Q(s', a') = 0)$$
$$Q(9, \text{right}) = 0 + 0.1 \cdot 10 = 1.0$$

**Particle**: $\omega = ((9, 0.5), 1.0)$

**MemoryUpdate**: Adds particle to $\Omega$, propagates positive weight to neighbors.

**Key moment**: Particles with positive values now exist near the goal!

---

### 6.5 Episode 2: Field-Guided Exploration

**Agent at $s = 2$ again**:

**Field prediction** (now informed by particles):

- Particles exist at $(9, 0.5)$, $(7, 0.5)$, etc. with positive weights
- $Q^+(2, \text{right}, 0.5)$ > $Q^+(2, \text{left}, 0.5)$ (right leads toward goal)

**Policy**: Boltzmann with $\beta = 1$:
$$\pi(\text{right} \mid 2) = \frac{e^{Q^+(2, \text{right}, 0.5)}}{e^{Q^+(2, \text{right}, 0.5)} + e^{Q^+(2, \text{left}, 0.5)}}$$

**Result**: Agent more likely to choose right (toward goal).

---

### 6.6 Long-Term Behavior

**After 100 episodes**:

- Particle memory $\Omega$ contains ~500 particles
- Positive-value particles cluster in paths leading to goal
- Negative-value particles mark obstacle region
- ARD has learned: position $s$ is critical, step size $\theta$ less so (larger lengthscale for $\theta$)

**Field $Q^+$**:

- High values: paths from any $s$ toward goal, avoiding obstacle
- Low values: paths toward obstacle or away from goal

**Policy**:

- Smooth, deterministic path from any start state to goal
- Naturally avoids obstacle (low $Q^+$ region)

**This is emergence**: The agent never explicitly computed an optimal path—it emerged from RF-SARSA's three forces!

---

## 7. Why RF-SARSA Is NOT Standard SARSA with Kernels

### 7.1 Common Misconception

**误 interpretation**: "RF-SARSA is just SARSA using kernel function approximation instead of a table."

**Why this is wrong**:

| Aspect | Kernel Function Approximation | RF-SARSA |
|---|---|---|
| **Updates** | Global weight vector $w$ via SGD | Particle ensemble $\Omega$ via MemoryUpdate |
| **Prediction** | $Q(s,a) = w^\top \phi(s,a)$ (linear) | $Q^+(z) = \sum_i \alpha_i k(z, z_i)$ (GPR) |
| **Memory** | Fixed basis $\phi$ | Growing/evolving particle set |
| **Geometry** | Fixed feature space | Adaptive (ARD on kernel) |

**Kernel FA** learns a weight vector. **RF-SARSA** shapes a functional field.

---

### 7.2 What RF-SARSA Actually Is

RF-SARSA is closer to:

**Galerkin methods** (functional analysis):

- Approximate solutions in a finite-dimensional subspace of an infinite-dimensional space
- RF-SARSA: subspace spanned by kernel sections $k(z_i, \cdot)$

**Interacting particle systems** (statistical physics):

- Particles influence each other through pairwise interactions
- RF-SARSA: MemoryUpdate propagates weights through kernel interactions

**Energy-based models** (modern ML):

- Learn energy function, sample from $p \propto \exp(-E)$
- RF-SARSA: Learn $E = -Q^+$, policy is Boltzmann distribution

**Belief-state RL** (POMDPs):

- Maintain belief over states, condition policy on belief
- RF-SARSA: $\Omega$ is belief representation, policy conditioned on $Q^+(\cdot; \Omega)$

---

## 8. Relation to Modern RL Methods

RF-SARSA anticipated several ideas that became mainstream later:

### 8.1 Kernel Temporal Difference Learning

**Kernel TD** (Engel et al., 2005):

- Apply TD learning in RKHS
- Use kernel trick for function approximation

**RF-SARSA connection**:

- Also uses kernels for TD learning
- But operates in augmented space $(s, \theta)$
- Couples with particle-based memory management

---

### 8.2 Energy-Based RL

**Modern EBMs** (e.g., Diffusion Q-learning, 2023):

- Represent policies/values as energy functions
- Sampling via Langevin dynamics

**RF-SARSA connection**:

- Energy interpretation $E = -Q^+$ (Chapter 3)
- Policy via Boltzmann distribution (Chapter 03a)
- Implicitly performs gradient flow on learned energy

---

### 8.3 Model-Based RL via GPs

**GP-based model learning** (Deisenroth et al., 2015):

- Learn forward dynamics $p(s' \mid s, a)$ via GP
- Plan using learned model

**RF-SARSA connection**:

- GP over augmented space provides implicit forward model
- Soft state transitions emerge from kernel similarity (Chapter 8, upcoming)
- No explicit dynamics model, but captures uncertainty

---

### 8.4 Neural Processes

**Neural Processes** (Garnelo et al., 2018):

- Learn a distribution over functions from context set
- Condition predictions on context

**RF-SARSA connection**:

- $\Omega$ is the context set (particle memory)
- $Q^+(z; \Omega)$ is the conditional prediction
- GPR is a (non-parametric) neural process!

---

## 9. Implementation Notes

### 9.1 Choosing Hyperparameters

**Kernel lengthscale $\ell$**:

- Too small: overfitting, no generalization
- Too large: over-smoothing, loss of detail
- **Solution**: Use ARD to learn per-dimension lengthscales

**Association threshold $\tau$**:

- Too low: all particles associate, slow computation
- Too high: no association, memory explosion
- **Rule of thumb**: $\tau \approx 0.1$ (10% correlation threshold)

**SARSA learning rate $\alpha$**:

- Standard RL tuning: start $\alpha = 0.1$, decay over time
- Should be larger than typical Q-learning (on-policy is more stable)

**Policy temperature $\beta$**:

- High $\beta$ (low temperature): greedy, exploitation
- Low $\beta$ (high temperature): stochastic, exploration
- **Schedule**: Exponential decay, e.g., $\beta_t = \beta_0 \cdot 1.01^t$

---

### 9.2 Computational Complexity

**Per-step costs**:

1. **Field query** (policy inference): $O(Nn)$
   - $N$: number of particles
   - $n$: number of discrete actions
   - GPR prediction: $O(N)$ per query (after precomputing $(K + \sigma_n^2 I)^{-1}q$)

2. **MemoryUpdate**: $O(N)$
   - Compute associations: $O(N)$
   - Update weights: $O(|\mathcal{N}|)$ where $|\mathcal{N}| \ll N$

3. **ARD** (every $T$ steps): $O(N^3)$
   - Solve GP regression: $O(N^3)$ (matrix inversion)
   - **Mitigation**: Use sparse GP methods, or increase $T$ over time

**Total**: $O(Nn + N + N^3/T) \approx O(Nn)$ for reasonable $T$, $N$.

**Scaling**: For large $N$ (>1000), use:

- Sparse GPs (inducing points)
- Random Fourier features
- Amortized inference (neural network)

---

### 9.3 Sparse GP Approximations

**Problem**: GP regression is $O(N^3)$ in number of particles.

**Solution**: Sparse GP (Quiñonero-Candela & Rasmussen, 2005):

- Choose $M \ll N$ inducing points
- Approximate $Q^+$ using only these points
- Reduces complexity to $O(M^2 N)$

**In RF-SARSA**:

- Select inducing points from particle memory (e.g., k-means)
- Update inducing points periodically

**Implementation**: Use GPyTorch or GPflow with `InducingPointStrategy`.

---

### 9.4 Python Pseudocode

```python
class RFSARSA:
    def __init__(self, kernel, action_model, alpha=0.1, gamma=0.9, beta=1.0, tau=0.1, T_ard=100):
        self.kernel = kernel  # e.g., RBF kernel
        self.action_model = action_model  # encodes/decodes actions
        self.alpha = alpha  # SARSA learning rate
        self.gamma = gamma  # discount
        self.beta = beta  # policy temperature
        self.tau = tau  # association threshold
        self.T_ard = T_ard  # ARD period
        
        self.Q_table = {}  # primitive Q(s,a)
        self.particles = []  # particle memory Ω = [(z, q), ...]
        self.t_ard = 0
    
    def gpr_predict(self, z):
        """Predict Q+(z) using GPR on particle memory."""
        if len(self.particles) == 0:
            return 0.0  # prior mean
        
        # Kernel vector: k(z, z_i) for all particles
        k_vec = np.array([self.kernel(z, p[0]) for p in self.particles])
        
        # GPR prediction (assuming precomputed alpha coefficients)
        return k_vec @ self.alpha_gpr
    
    def policy(self, s, actions):
        """Boltzmann policy over actions based on Q+ field."""
        q_values = []
        for a in actions:
            x_a = self.action_model.encode(a)
            z = np.concatenate([s, x_a])
            q_plus = self.gpr_predict(z)
            q_values.append(q_plus)
        
        q_values = np.array(q_values)
        probs = np.exp(self.beta * q_values)
        probs /= probs.sum()
        
        return np.random.choice(actions, p=probs)
    
    def update(self, s, a, r, s_next, a_next):
        """RF-SARSA update: primitive SARSA + MemoryUpdate."""
        # Primitive SARSA update
        q_current = self.Q_table.get((s, a), 0.0)
        q_next = self.Q_table.get((s_next, a_next), 0.0)
        
        delta = r + self.gamma * q_next - q_current
        q_new = q_current + self.alpha * delta
        
        self.Q_table[(s, a)] = q_new
        
        # Form particle
        x_a = self.action_model.encode(a)
        z = np.concatenate([s, x_a])
        particle = (z, q_new)
        
        # MemoryUpdate (Algorithm 1)
        self.particles = memory_update(
            particle, delta, self.kernel, self.tau, self.particles
        )
        
        # Periodic ARD update
        self.t_ard += 1
        if self.t_ard % self.T_ard == 0:
            self.update_kernel_hyperparameters()
    
    def update_kernel_hyperparameters(self):
        """Run ARD to update kernel lengthscales."""
        # Extract (z, q) from particles
        Z = np.array([p[0] for p in self.particles])
        q = np.array([p[1] for p in self.particles])
        
        # Fit GP with ARD (optimize lengthscales)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        
        kernel = RBF(length_scale=np.ones(Z.shape[1]), length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gp.fit(Z, q)
        
        # Update kernel with learned lengthscales
        self.kernel.set_lengthscales(gp.kernel_.length_scale)
        
        # Recompute GPR coefficients
        K = self.kernel.matrix(Z, Z)
        self.alpha_gpr = np.linalg.solve(K + 1e-6 * np.eye(len(Z)), q)
```

---

## 10. Strengths and Limitations

### 10.1 Strengths

**1. Generalization in Continuous Action Spaces**
- Natural handling of parametric actions
- Smooth interpolation across action parameters

**2. Uncertainty Quantification**
- GP provides variance $\sigma^2(z)$ (not shown above for simplicity)
- Can guide exploration (e.g., UCB: $Q^+ + \kappa \sigma$)

**3. Adaptive Metric Learning**
- ARD discovers relevant dimensions automatically
- No manual feature engineering required

**4. Interpretability**
- Particles are interpretable (experienced configurations)
- Field visualization shows value landscape

**5. Theoretical Grounding**
- Connection to RKHS theory (Chapters 2, 4)
- Least action principle (Chapter 03a)
- POMDP interpretation (Chapter 8, upcoming)

---

### 10.2 Limitations

**1. Computational Cost**
- $O(N^3)$ for GP regression (ARD step)
- Mitigated by sparse GPs, but still slower than deep RL

**2. Memory Growth**
- Particle memory grows over time
- Needs pruning/consolidation strategies (Chapter 06a)

**3. Discrete Action Assumption**
- Algorithm as stated requires discrete action set for field queries
- **Solution**: Continuous optimization via Langevin sampling (Chapter 03a)

**4. On-Policy Learning**
- SARSA is on-policy (learns about behavior policy)
- Slower than off-policy (Q-learning, DQN)
- **Tradeoff**: More stable, safer for risk-sensitive domains

**5. Hyperparameter Sensitivity**
- Requires tuning $\alpha$, $\beta$, $\tau$, $T$
- ARD helps but doesn't eliminate manual tuning

---

## 11. Summary

### 11.1 What RF-SARSA Does

RF-SARSA is **not** a policy learning algorithm. It is a **functional reinforcement mechanism** that:

1. **Grounds** value estimates temporally via primitive SARSA ($Q(s,a)$)
2. **Generalizes** them spatially via GP regression over particles ($Q^+(z)$)
3. **Propagates** them geometrically via MemoryUpdate (Algorithm 1)
4. **Adapts** the metric via ARD (kernel hyperparameter learning)

**Policy emerges** as a consequence of this process (via field queries), not as its direct goal.

---

### 11.2 Key Conceptual Insights

**Two-layer architecture**:

- Primitive layer (SARSA) → temporal grounding
- Field layer (GPR) → spatial generalization

**Belief-state interpretation**:

- Particle memory $\Omega$ = agent's belief state
- MemoryUpdate = belief update operator
- Policy = belief-conditioned action inference

**Physics grounding**:

- Energy landscape $E = -Q^+$ learned from experience
- Boltzmann policy minimizes expected action (Chapter 03a)
- Smooth trajectories emerge from kinetic regularization (kernel smoothness)

---

### 11.3 Connection to the Big Picture

**Part I: Reinforcement Fields** (where we are now):

- Chapter 4: What is the reinforcement field? (functional object in RKHS)
- Chapter 5: How is it represented? (particles as basis elements)
- Chapter 6: How does memory evolve? (MemoryUpdate as belief transition)
- **Chapter 7** (this chapter): **How is the field learned?** (RF-SARSA as functional TD)

**Coming next**:

- Chapter 8: What emerges from this? (soft state transitions, uncertainty)
- Chapter 9: How to interpret this? (POMDP view, belief-based control)
- Chapter 10: Putting it all together (complete GRL system)

---

## 12. Key Takeaways

1. **RF-SARSA couples two learning processes**: primitive SARSA (temporal grounding) + GPR (spatial generalization)

2. **It's not SARSA with kernels**: Updates particle ensemble, not weight vector; reshapes functional field, not table entries

3. **Three forces enable learning**: temporal credit (SARSA), geometric generalization (GP), adaptive geometry (ARD)

4. **Policy is inferred, not learned**: Field queries via GPR → Boltzmann sampling → actions

5. **Physics-grounded**: Energy landscape from least action principle; smooth trajectories from kinetic regularization

6. **Belief-state formulation**: $\Omega$ is belief, MemoryUpdate is belief update, policy is belief-conditioned

7. **Scalable with approximations**: Sparse GPs, random features, or neural networks for large-scale problems

8. **Anticipated modern methods**: Kernel TD, energy-based RL, GP-based model RL, neural processes

---

## 13. Further Reading

**Original RF-SARSA**:

- Chiu & Huber (2022), Section IV. [arXiv:2208.04822](https://arxiv.org/abs/2208.04822)

**Kernel Temporal Difference Learning**:

- Engel, Y., Mannor, S., & Meir, R. (2005). "Reinforcement learning with Gaussian processes." *ICML*.
- Xu, X., et al. (2007). "Kernel-based least squares policy iteration for reinforcement learning." *IEEE TNNLS*.

**Path Integral Control (connection to Least Action)**:

- Theodorou, E., Buchli, J., & Schaal, S. (2010). "A generalized path integral control approach to reinforcement learning." *JMLR*.

**Gaussian Processes for RL**:

- Deisenroth, M. P., & Rasmussen, C. E. (2011). "PILCO: A model-based and data-efficient approach to policy search." *ICML*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

**Energy-Based RL**:

- Haarnoja, T., et al. (2017). "Reinforcement learning with deep energy-based policies." *ICML* (SQL).
- Ajay, A., et al. (2023). "Is conditional generative modeling all you need for decision making?" *ICLR* (Diffusion-QL).

---

**[← Back to Chapter 06a: Advanced Memory Dynamics](06a-advanced-memory-dynamics.md)** | **[Next: Chapter 08 (Coming Soon) →]()**

**[Related: Chapter 03a - Least Action Principle](03a-least-action-principle.md)** | **[Related: Chapter 06 - MemoryUpdate](06-memory-update.md)**

---

**Last Updated**: January 14, 2026
