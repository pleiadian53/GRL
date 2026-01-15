# Chapter 6: MemoryUpdate — The Agent's Belief Transition

**Purpose:** Understand how GRL agents evolve their understanding of the world  
**Prerequisites:** Chapters 4 (Reinforcement Field), 5 (Particle Memory)  
**Key Concepts:** Belief update, particle evolution, kernel association, functional memory

---

## Why This Chapter Matters

In Chapter 5, we learned that **memory is a functional representation**—a weighted set of experience particles that induce a reinforcement field.

But how does this representation **change** as the agent experiences new things?

This chapter introduces **Algorithm 1: MemoryUpdate**—the core mechanism that:
- Converts new experiences into particles
- Associates them with existing knowledge
- Reshapes the energy landscape
- Maintains a tractable particle set

**Critical insight:**

> MemoryUpdate is not just "memory management"—it is the **belief-state transition operator** of GRL, expressed in RKHS rather than probability space.

Everything downstream (policy inference, reinforcement propagation, soft transitions, POMDP) emerges from this single operation.

---

## The Problem: How Should an Agent Update Its Beliefs?

### Traditional RL Approach

In standard RL, when the agent observes $(s_t, a_t, r_t, s_{t+1})$, it updates:

1. **Value function** via Bellman backup:
   $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

2. **Experience buffer** (if using replay):
   $$\text{buffer} \leftarrow \text{buffer} \cup \{(s, a, r, s')\}$$

3. **Policy** (if policy gradient):
   $$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)$$

These are **three separate operations** on **three separate structures**.

### GRL Approach

In GRL, when the agent observes $(s_t, \theta_t, r_t)$ (remember: actions are parameters $\theta$), it performs **one operation**:

$$\text{MemoryUpdate}: \mathcal{M} \rightarrow \mathcal{M}'$$

This single operation **simultaneously**:
- Updates the agent's belief about the value landscape
- Incorporates new experience into memory
- Reshapes the policy (implicitly, via the field)
- Maintains uncertainty representation

**How?** By operating at the **functional level** in RKHS, not the parametric level.

---

## What MemoryUpdate Does (Conceptual View)

Before diving into the algorithm, let's understand its four core operations:

### 1. Particle Instantiation

**What:** Convert experience $(s_t, \theta_t, r_t)$ into an augmented state-action particle

$$z_{new} = (s_t, \theta_t)$$

**Why:** This particle is a **hypothesis** about what matters—a point in augmented space where something significant happened.

**Not:** Just storing a tuple. The particle becomes a **functional basis element** in RKHS.

---

### 2. Geometric Association

**What:** Compute how similar the new experience is to existing particles

$$a_i = k(z_{new}, z_i) \quad \text{for each } z_i \in \mathcal{M}$$

where $k(\cdot, \cdot)$ is the RKHS kernel over augmented space.

**Why:** This determines:
- How much the new experience **confirms** existing beliefs
- How far its influence should **spread** in the landscape
- Which particles should be **adjusted** in response

**Not:** Nearest-neighbor lookup. Association is **soft**, **global**, and **geometry-aware**.

---

### 3. Energy Functional Maintenance

**What:** Assign a weight $w_{new}$ to the new particle, and optionally adjust weights of associated particles

$$w_{new} = g(r_t, \{a_i\})$$

where $g(\cdot)$ maps reinforcement evidence to an energy contribution.

**Why:** Particle weights determine the shape of the energy landscape:

$$E(z) = -Q^+(z) = -\sum_i w_i k(z_i, z)$$

Adjusting weights **reshapes** this landscape to reflect new evidence.

**Not:** Direct value update. We're modifying the **functional representation**, not tabular entries.

---

### 4. Memory Management

**What:** Prune, merge, or decay particles to maintain bounded complexity

**Why:** Without this, the particle set grows unboundedly and computational cost explodes.

**How:**
- Attenuate particles with small $|w_i|$ (low evidence)
- Merge particles with high mutual similarity (redundant)
- Discard particles below relevance threshold (forgetting)

**Not:** Simple replay buffer eviction. This is **functional regularization** + **model selection**.

---

## Algorithm 1: MemoryUpdate (Formal Specification)

Now let's see the actual algorithm with precise notation.

---

### Input

| Symbol | Meaning | Type |
|--------|---------|------|
| $\mathcal{M}$ | Current particle memory | $\{(z_i, w_i)\}_{i=1}^N$ |
| $z_i = (s_i, \theta_i)$ | Augmented state-action particle | $z_i \in \mathcal{Z} = \mathcal{S} \times \Theta$ |
| $w_i$ | Particle weight (energy contribution) | $w_i \in \mathbb{R}$ |
| $(s_t, \theta_t, r_t)$ | New experience tuple | $(s_t, \theta_t) \in \mathcal{Z}, r_t \in \mathbb{R}$ |
| $k(\cdot, \cdot)$ | RKHS kernel over augmented space | $k: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ |

### Output

| Symbol | Meaning |
|--------|---------|
| $\mathcal{M}'$ | Updated particle memory |

---

### Algorithm Steps

**Step 1: Particle Construction**

Form the new augmented particle:

$$z_{new} = (s_t, \theta_t)$$

---

**Step 2: Kernel-Based Association**

For each existing particle $z_i \in \mathcal{M}$, compute similarity:

$$a_i = k(z_{new}, z_i)$$

This creates an **association vector**:

$$\mathbf{a} = [a_1, a_2, \ldots, a_N]^T$$

---

**Step 3: Energy Contribution Initialization**

Assign an initial weight to $z_{new}$, proportional to the reinforcement signal:

$$w_{new} = g(r_t, \mathbf{a})$$

**Common choices for $g(\cdot)$:**

1. **Direct reinforcement:** $w_{new} = r_t$
2. **Normalized:** $w_{new} = \frac{r_t}{\sum_i a_i + \epsilon}$
3. **Novelty-weighted:** $w_{new} = r_t \cdot (1 - \max_i a_i)$

The paper uses option 1 (direct) for simplicity, but options 2-3 can improve performance.

---

**Step 4: Memory Integration**

Insert the new particle into memory:

$$\mathcal{M} \leftarrow \mathcal{M} \cup \{(z_{new}, w_{new})\}$$

At this point, the energy functional becomes:

$$E(z) = -\sum_{i=1}^{N} w_i k(z_i, z) - w_{new} k(z_{new}, z)$$

---

**Step 5: Association-Driven Adjustment** (Optional)

For particles with strong association ($a_i > \varepsilon$), adjust their weights:

$$w_i \leftarrow w_i + \lambda \cdot a_i \cdot w_{new}$$

where:

- $\lambda \in [0, 1]$ controls propagation strength
- $a_i$ determines how much particle $i$ should respond
- $w_{new}$ provides the signal to propagate

**Interpretation:** New evidence **propagates geometrically** to related experiences, enabling generalization without Bellman backups.

---

**Step 6: Memory Regularization**

Apply one or more of the following:

**a) Weight decay:**

$$w_i \leftarrow \gamma_w \cdot w_i \quad \text{for all } i$$

where $\gamma_w \in (0, 1]$ (e.g., 0.99).

**b) Particle merging:**

If $k(z_i, z_j) > \tau_{merge}$ for some pair $(i, j)$:

$$z_{merged} = \frac{w_i z_i + w_j z_j}{w_i + w_j}, \quad w_{merged} = w_i + w_j$$

Remove $(z_i, w_i)$ and $(z_j, w_j)$; add $(z_{merged}, w_{merged})$.

**c) Particle pruning:**

Remove particles with $|w_i| < \tau_{prune}$.

---

**Step 7: Return Updated Memory**

$$\text{return } \mathcal{M}'$$

---

## Line-by-Line Interpretation

Let's unpack what each step **really means** for the agent.

### Step 1: Particle Construction

**Plain language:**

> "This experience is a hypothesis about what actions matter in what states."

**Why particle form?**

Because in RKHS, every particle $z_i$ induces a **kernel section** $k(z_i, \cdot)$—a function in $\mathcal{H}_k$ that carries influence throughout the augmented space.

**Geometric view:**

You're adding a "probe" into the value landscape—a point that will shape the field around it.

---

### Step 2: Kernel Association

**Plain language:**

> "How compatible is this experience with what I already believe?"

**Not asking:** "Is this the same state?" or "Is this the same action?"

**Asking:** "How much does this experience **resonate** with existing knowledge?"

**Example (Gaussian RBF kernel):**

$$a_i = \exp\left(-\frac{\|z_{new} - z_i\|^2}{2\sigma^2}\right)$$

- If $z_{new} \approx z_i$: $a_i \approx 1$ (high similarity)
- If $z_{new}$ far from $z_i$: $a_i \approx 0$ (low similarity)
- Intermediate distances: $a_i \in (0, 1)$ (partial similarity)

This is **soft association**—every particle contributes, weighted by distance.

---

### Step 3: Energy Contribution Initialization

**Plain language:**

> "How strongly should this experience shape the landscape?"

**Not:** "What is the value of $(s, a)$?"

**But:** "How much **evidence** does this experience provide?"

**Sign matters:**
- If $r_t > 0$: $w_{new} > 0$ → positive particle → **attracts** in field
- If $r_t < 0$: $w_{new} < 0$ → negative particle → **repels** in field

**Magnitude matters:**
- Large $|r_t|$: Strong influence
- Small $|r_t|$: Weak influence

---

### Step 4: Memory Integration

**Plain language:**

> "The agent's state of knowledge has changed."

After this step, the energy functional is:

$$E(z) = -\sum_{i=1}^{N+1} w_i k(z_i, z)$$

This is the **new belief state** of the agent—a functional representation of all experiences so far.

**Key insight:** Memory grows **functionally**, not discretely. The agent doesn't just "remember more tuples"—it maintains a richer **functional prior** for policy inference.

---

### Step 5: Association-Driven Adjustment

**Plain language:**

> "New evidence doesn't just affect the new particle—it propagates to related experiences."

This is **subtle and powerful**.

**Example:**

Suppose:
- Agent has particle $z_1 = (s_1, \theta_1)$ with $w_1 = +2.0$
- New experience: $z_{new} = (s_1, \theta_2)$ with $r_t = +3.0$
- Association: $a_1 = k(z_{new}, z_1) = 0.8$ (high similarity)

After Step 5:

$$w_1 \leftarrow 2.0 + \lambda \cdot 0.8 \cdot 3.0 = 2.0 + 2.4\lambda$$

If $\lambda = 0.5$, then $w_1 = 3.2$.

**Interpretation:** The new positive experience at $z_{new}$ **reinforces** the nearby particle $z_1$, even though they have different action parameters.

This is how GRL **generalizes without Bellman backups**—evidence spreads through kernel geometry.

---

### Step 6: Memory Regularization

**Plain language:**

> "Keep the particle set tractable and adaptive."

**Why each sub-operation matters:**

**a) Weight decay:**
- Implements **forgetting** (older evidence matters less)
- Prevents weight explosion
- Allows adaptation in non-stationary environments

**b) Particle merging:**
- Removes **redundancy** (two particles in nearly the same location)
- Maintains **functional expressiveness** without wasting particles
- Natural form of **model compression**

**c) Particle pruning:**
- Removes **low-evidence particles** (noise or outliers)
- Bounds computational cost
- Implements **Occam's razor** (prefer simpler representations)

---

## Worked Example: 1D Navigation

Let's see MemoryUpdate in action with a concrete example.

### Setup

- **State space:** $\mathcal{S} = [0, 10]$ (1D position)
- **Action space:** $\Theta = [-1, +1]$ (velocity parameter)
- **Kernel:** Gaussian RBF, $k(z, z') = \exp(-\|z - z'\|^2 / 2)$
- **Goal:** $s = 10$ (reward $r = +10$)
- **Current memory:** $\mathcal{M} = \{(z_1, w_1), (z_2, w_2)\}$

where:

- $z_1 = (8.0, +0.5)$, $w_1 = +5.0$
- $z_2 = (3.0, -0.3)$, $w_2 = -2.0$

### New Experience

Agent at $s_t = 9.0$, takes action $\theta_t = +0.8$, reaches goal, receives $r_t = +10.0$.

---

### Step-by-Step Execution

**Step 1: Particle Construction**

$$z_{new} = (9.0, 0.8)$$

---

**Step 2: Kernel Association**

Compute similarity to existing particles:

$$a_1 = k(z_{new}, z_1) = k\Big((9.0, 0.8), (8.0, 0.5)\Big)$$

Distance:

$$\|z_{new} - z_1\| = \sqrt{(9.0 - 8.0)^2 + (0.8 - 0.5)^2} = \sqrt{1.0 + 0.09} = 1.044$$

Similarity:

$$a_1 = \exp(-1.044^2 / 2) = \exp(-0.545) \approx 0.58$$

Similarly:

$$\|z_{new} - z_2\| = \sqrt{(9.0 - 3.0)^2 + (0.8 + 0.3)^2} = \sqrt{36 + 1.21} = 6.10$$

$$a_2 = \exp(-6.10^2 / 2) = \exp(-18.6) \approx 0.0$$

**Association vector:** $\mathbf{a} = [0.58, 0.0]^T$

**Interpretation:** The new experience is **similar** to $z_1$ (both near the goal with positive actions), but **dissimilar** to $z_2$ (far from goal, negative action).

---

**Step 3: Energy Contribution**

Using direct reinforcement:

$$w_{new} = r_t = +10.0$$

---

**Step 4: Memory Integration**

$$\mathcal{M} \leftarrow \{(z_1, +5.0), (z_2, -2.0), (z_{new}, +10.0)\}$$

Energy functional:

$$E(z) = -5.0 \cdot k(z, z_1) + 2.0 \cdot k(z, z_2) - 10.0 \cdot k(z, z_{new})$$

(Recall: $E = -Q^+$, so negative weights attract, positive repel in energy terms.)

---

**Step 5: Association-Driven Adjustment** (with $\lambda = 0.5$)

Only $z_1$ has significant association ($a_1 = 0.58 > \varepsilon$):

$$w_1 \leftarrow 5.0 + 0.5 \cdot 0.58 \cdot 10.0 = 5.0 + 2.9 = 7.9$$

Updated memory:

$$\mathcal{M} = \{(z_1, +7.9), (z_2, -2.0), (z_{new}, +10.0)\}$$

**Interpretation:** The new positive experience **reinforces** $z_1$ because they're geometrically close—generalization via kernel association!

---

**Step 6: Memory Regularization** (simplified)

Apply weight decay with $\gamma_w = 0.99$:

$$w_1 \leftarrow 0.99 \cdot 7.9 = 7.82$$
$$w_2 \leftarrow 0.99 \cdot (-2.0) = -1.98$$
$$w_{new} \leftarrow 0.99 \cdot 10.0 = 9.90$$

Final memory:

$$\mathcal{M}' = \{(z_1, +7.82), (z_2, -1.98), (z_{new}, +9.90)\}$$

---

### What Changed?

**Before MemoryUpdate:**

Energy landscape had:
- Moderate attraction near $(8.0, 0.5)$ (weight $+5.0$)
- Weak repulsion near $(3.0, -0.3)$ (weight $-2.0$)

**After MemoryUpdate:**

Energy landscape has:
- **Stronger attraction** near $(8.0, 0.5)$ (weight $+7.82$) ← reinforced by new evidence
- Still weak repulsion near $(3.0, -0.3)$ (weight $-1.98$) ← unchanged (no association)
- **Strong new attraction** near $(9.0, 0.8)$ (weight $+9.90$) ← new high-value region

**Policy implication:** Agent will now strongly prefer positive velocities when near the goal (generalization from one good experience to a neighborhood!).

---

## Why MemoryUpdate Is Instrumental (Not Just Representational)

Earlier we said MemoryUpdate is the **belief-state transition operator**. Let's unpack why it's **instrumental** for everything downstream.

### 1. MemoryUpdate Enables Policy Inference

The policy is defined as:

$$\pi(\theta | s) \propto \exp(\beta \cdot Q^+(s, \theta))$$

But $Q^+$ is determined by particle memory:

$$Q^+(s, \theta) = \sum_i w_i k((s, \theta), z_i)$$

So **changing $\mathcal{M}$ directly changes the policy**—no separate policy update needed.

**Example from above:** After the update, $\pi(\theta | s=9)$ will strongly favor $\theta \approx +0.8$ because $w_{new} = +9.90$ creates high $Q^+$ there.

---

### 2. MemoryUpdate Implements Reinforcement Propagation

In traditional RL, reinforcement propagates via Bellman backups:

$$Q(s, a) \gets r + \gamma \max_{a'} Q(s', a')$$

In GRL, reinforcement propagates via **kernel association** (Step 5):

$$w_i \leftarrow w_i + \lambda \cdot a_i \cdot w_{new}$$

**Advantages:**
- **Soft:** Propagation strength depends on similarity $a_i$
- **Global:** All associated particles update simultaneously
- **Geometry-aware:** Propagation respects the RKHS metric

---

### 3. MemoryUpdate Induces Soft State Transitions

Classical RL assumes deterministic or explicitly stochastic state transitions:

$$s_{t+1} \sim P(\cdot | s_t, a_t)$$

GRL has **no explicit transition model**. Instead, transitions are **implicitly soft** due to kernel overlap.

**How?**

When the agent observes $(s_t, \theta_t, r_t)$, it creates particle $z_t = (s_t, \theta_t)$. This particle associates with **multiple** existing particles via $a_i = k(z_t, z_i)$, each representing a different "hypothesis" about what state-action is relevant.

The agent maintains **belief over multiple hypotheses simultaneously**—emergent uncertainty from geometry!

---

### 4. MemoryUpdate Is the POMDP Belief Update

In a POMDP, belief updates follow:

$$b'(s') = \frac{O(o | s') \sum_s T(s' | s, a) b(s)}{\text{normalizer}}$$

In GRL, particle memory $\mathcal{M}$ **is** the belief state:

$$b(z) \propto \sum_i w_i k(z, z_i)$$

MemoryUpdate transitions:

$$b_t(z) \xrightarrow{\text{MemoryUpdate}} b_{t+1}(z)$$

But instead of Bayesian filtering over a discrete state space, we're doing **functional belief evolution in RKHS**.

**Key insight:** POMDP structure emerges **for free** from the particle representation—no need to specify observation or transition models!

---

## Connection to Other GRL Components

Let's see how MemoryUpdate connects to the rest of the GRL framework.

### MemoryUpdate + Reinforcement Field

**Chapter 4:** The reinforcement field is:

$$Q^+(z) = \sum_i w_i k(z_i, z)$$

**MemoryUpdate modifies:**
- Particle locations $\{z_i\}$ (by adding new particles)
- Particle weights $\{w_i\}$ (by adjusting in response to new evidence)

**Result:** The field **reshapes** to reflect new experience—peaks form in high-value regions, valleys in low-value regions.

---

### MemoryUpdate + Particle Memory

**Chapter 5:** Memory is a **functional representation** of experience.

**MemoryUpdate implements:**
- **Dynamic basis expansion:** Adding new particles = adding new basis functions
- **Weight adjustment:** Changing the functional prior
- **Basis compression:** Merging/pruning to maintain tractability

**Result:** Memory stays **expressive** (can represent complex fields) yet **bounded** (doesn't explode).

---

### MemoryUpdate + RF-SARSA

**Chapter 7 (next):** RF-SARSA provides the **reinforcement signal** $r_t$ that enters MemoryUpdate.

**Two-layer system:**
1. **RF-SARSA (outer loop):** Decides *what reinforcement signal* to send
2. **MemoryUpdate (inner loop):** Reshapes the field in response

**Analogy:** RF-SARSA is the "teacher" providing feedback; MemoryUpdate is the "learner" updating beliefs.

---

## Common Misconceptions

### Misconception 1: "MemoryUpdate Is Just Experience Replay"

**Reality:** Experience replay stores tuples and samples them for training. MemoryUpdate converts experiences into **functional basis elements** that directly shape the value landscape.

**Key difference:** Replay is passive storage; MemoryUpdate is active belief evolution.

---

### Misconception 2: "Kernel Association Is Just Nearest-Neighbor"

**Reality:** Kernel association is:
- **Soft:** All particles contribute, weighted by similarity
- **Global:** Associations computed with all particles simultaneously
- **Differentiable:** Enables smooth generalization

Nearest-neighbor is:
- **Hard:** Only the closest point matters
- **Local:** No information about the broader landscape
- **Discontinuous:** Small changes in input cause jumps

---

### Misconception 3: "MemoryUpdate Only Affects Memory"

**Reality:** MemoryUpdate **simultaneously affects**:
- The value landscape (via particle weights)
- The policy (via $Q^+(s, \theta)$)
- Belief uncertainty (via particle diversity)
- Generalization (via kernel propagation)

It's the **state transition operator** of the agent.

---

### Misconception 4: "Step 5 (Association-Driven Adjustment) Is Optional"

**Truth:** Step 5 is optional **in implementation**, but it's **conceptually central** to GRL.

**Why:** Without Step 5, GRL reduces to kernel regression with independent samples. With Step 5, GRL becomes a **belief propagation system** where evidence spreads through geometry.

**Recommendation:** Include Step 5 for best performance, especially in continuous domains where generalization is critical.

---

## Practical Considerations

### 1. Computational Complexity

**Per update cost:**

- **Step 2 (Association):** $O(N)$ kernel evaluations
- **Step 5 (Adjustment):** $O(N)$ weight updates (if all $a_i > \varepsilon$)
- **Step 6 (Regularization):** $O(N^2)$ for merging (pairwise comparison), $O(N)$ for pruning

**Total:** $O(N)$ for association + adjustment, $O(N^2)$ for merging.

**Optimization:**
- Use **sparse association:** Only update particles with $a_i > \varepsilon$
- Use **KD-trees** or **ball trees** to find high-association particles quickly
- Merge periodically (every $K$ steps) rather than every update

---

### 2. Hyperparameters

| Parameter | Meaning | Typical Range | Effect of Increasing |
|-----------|---------|---------------|----------------------|
| $\sigma$ (kernel bandwidth) | Similarity scale | $[0.1, 10]$ | More particles associate (smoother field) |
| $\lambda$ (propagation) | Step 5 coupling strength | $[0, 1]$ | Stronger evidence propagation |
| $\gamma_w$ (decay) | Weight decay rate | $[0.95, 1.0]$ | Faster forgetting |
| $\tau_{merge}$ | Merge threshold | $[0.8, 0.99]$ | More aggressive merging |
| $\tau_{prune}$ | Prune threshold | $[0.01, 0.5]$ | More aggressive pruning |

**Tuning guidance:**
- Start with $\sigma \approx \text{characteristic state-action distance}$
- Use $\lambda = 0.5$ as default (moderate propagation)
- Set $\gamma_w = 0.99$ for stationary environments, $0.95$ for non-stationary
- Tune $\tau_{merge}$ and $\tau_{prune}$ based on memory budget

---

### 3. Implementation Tips

**Efficient kernel computation:**

For Gaussian RBF:

```python
def kernel(z1, z2, sigma=1.0):
    dist_sq = np.sum((z1 - z2)**2)
    return np.exp(-dist_sq / (2 * sigma**2))
```

**Sparse association:**

```python
def update_memory(memory, z_new, r_t, epsilon=0.1, lambda_prop=0.5):
    # Step 2: Compute associations
    associations = [kernel(z_new, z_i) for z_i, w_i in memory]
    
    # Step 3: Initialize weight
    w_new = r_t  # Direct reinforcement
    
    # Step 4: Add to memory
    memory.append((z_new, w_new))
    
    # Step 5: Update associated particles (sparse)
    for i, (a_i, (z_i, w_i)) in enumerate(zip(associations, memory[:-1])):
        if a_i > epsilon:
            memory[i] = (z_i, w_i + lambda_prop * a_i * w_new)
    
    return memory
```

**Vectorized version:**

```python
import numpy as np

def update_memory_vectorized(Z, W, z_new, r_t, sigma=1.0, lambda_prop=0.5):
    """
    Z: (N, d) array of particle locations
    W: (N,) array of particle weights
    z_new: (d,) new particle location
    r_t: scalar reward
    """
    # Step 2: Compute all associations at once
    dists_sq = np.sum((Z - z_new[None, :])**2, axis=1)
    A = np.exp(-dists_sq / (2 * sigma**2))  # (N,) association vector
    
    # Step 3 + 4: Add new particle
    w_new = r_t
    Z = np.vstack([Z, z_new[None, :]])
    W = np.append(W, w_new)
    
    # Step 5: Update associated particles
    W[:-1] += lambda_prop * A * w_new
    
    return Z, W
```

---

## Visualization: Field Evolution

Let's see how the reinforcement field evolves through MemoryUpdate.

### Initial State (2 particles)

```
Particle memory:
  z_1 = (8.0, +0.5), w_1 = +5.0
  z_2 = (3.0, -0.3), w_2 = -2.0

Field Q^+(s, θ):
      θ
    1 |           ⊕ (peak near z_1)
    0 |------------------------
   -1 |     ⊖ (valley near z_2)
      |________________________
      0    3    6    9    10   s
```

**Legend:**
- ⊕ High value region (positive particle)
- ⊖ Low value region (negative particle)

---

### After New Experience (3 particles)

**New experience:** $(s=9, θ=0.8, r=+10)$

```
Particle memory:
  z_1 = (8.0, +0.5), w_1 = +7.9  ← reinforced!
  z_2 = (3.0, -0.3), w_2 = -2.0  ← unchanged
  z_new = (9.0, +0.8), w_new = +10.0  ← new peak!

Field Q^+(s, θ):
      θ
    1 |          ⊕⊕ (stronger peak, two nearby particles)
    0 |------------------------
   -1 |     ⊖
      |________________________
      0    3    6    9    10   s
```

**What changed:**
- Peak near goal **strengthened** (two particles now: $z_1$ and $z_{new}$)
- Peak **widened** (kernel overlap creates broader attraction)
- Policy will strongly prefer $\theta \approx 0.5$-$0.8$ when $s \approx 8$-$9$

---

## Summary

### MemoryUpdate Is...

| What It Is | What It Does |
|------------|--------------|
| **Belief-state transition operator** | Updates agent's functional representation of the world |
| **RKHS operation** | Adds basis functions, adjusts weights in function space |
| **Generalization mechanism** | Spreads evidence through kernel geometry |
| **Functional regularizer** | Maintains bounded, expressive particle sets |

### Key Equations

**Energy functional after update:**

$$E(z) = -\sum_{i \in \mathcal{M}'} w_i k(z_i, z)$$

**Association vector:**

$$a_i = k(z_{new}, z_i)$$

**Association-driven weight update:**

$$w_i \leftarrow w_i + \lambda \cdot a_i \cdot w_{new}$$

---

## Key Takeaways

1. **MemoryUpdate is not "just memory"**
   - It's the agent's state transition operator
   - It simultaneously affects value, policy, and belief

2. **Kernel association enables soft generalization**
   - Evidence spreads to geometrically similar experiences
   - No Bellman backups needed

3. **Particles are functional bases**
   - Adding a particle = adding a basis function to RKHS
   - Adjusting weights = reshaping the field

4. **Regularization is essential**
   - Without pruning/merging, complexity explodes
   - Implements forgetting and model selection

5. **MemoryUpdate enables everything downstream**
   - Policy inference: via $Q^+(s, \theta)$ induced by particles
   - Soft transitions: via kernel overlap
   - POMDP: via functional belief representation

---

## Looking Ahead

In the next chapter, we'll introduce **RF-SARSA**—the algorithm that determines *what reinforcement signal* to send to MemoryUpdate.

**Spoiler:** RF-SARSA implements **temporal-difference learning in RKHS**, enabling the agent to bootstrap value estimates from the reinforcement field itself.

Together, MemoryUpdate and RF-SARSA form a **two-layer learning system**:
- **RF-SARSA:** Computes TD error in function space
- **MemoryUpdate:** Reshapes the field in response

This is how GRL learns without explicit value function approximation or policy gradients!

---

## Further Reading

### Within This Tutorial

- **Chapter 2**: [RKHS Foundations](02-rkhs-foundations.md) — Kernel inner products and function spaces
- **Chapter 4**: [Reinforcement Field](04-reinforcement-field.md) — The functional landscape shaped by particles
- **Chapter 4a**: [Riesz Representer](04a-riesz-representer.md) — Gradients in function space
- **Chapter 5**: [Particle Memory](05-particle-memory.md) — Memory as functional representation

### Next Chapters

- **[Chapter 6a](06a-advanced-memory-dynamics.md)**: Advanced Memory Dynamics (supplement) — Practical improvements beyond hard thresholds
- **Chapter 7**: RF-SARSA (next) — Temporal-difference learning in RKHS
- **Chapter 8**: Soft State Transitions — Emergent uncertainty from kernel geometry
- **Chapter 9**: POMDP Interpretation — Belief-based view of GRL

### Advanced Topics

For principled alternatives to hard threshold $\tau$:

**[Chapter 06a: Advanced Memory Dynamics →](06a-advanced-memory-dynamics.md)**

Practical improvements:
- **Top-k Adaptive Neighbors** — Density-aware, no global threshold
- **Surprise-Gated Consolidation** — Data-driven, bounded memory growth
- **Hybrid Approach** — Combines both methods
- Code examples and decision guides

For full theoretical treatment:
- **[Chapter 07: Learning Beyond GP](../quantum_inspired/07-learning-the-field-beyond-gp.md)** — Alternative learning mechanisms
- **[Chapter 08: Memory Dynamics](../quantum_inspired/08-memory-dynamics-formation-consolidation-retrieval.md)** — Formation, consolidation, retrieval operators

### Original Paper

- **Section IV-A**: Experience Association and Particle Evolution
- **Algorithm 1**: MemoryUpdate (original specification)

**arXiv:** [Generalized Reinforcement Learning: Experience Particles, Action Operator, Reinforcement Field, Memory Association, and Decision Concepts](https://arxiv.org/abs/2208.04822)

---

**Last Updated:** January 14, 2026
