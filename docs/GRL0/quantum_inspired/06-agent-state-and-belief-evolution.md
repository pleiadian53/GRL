# Chapter 6: The Agent's State and Belief Evolution

## Motivation

Throughout this series, we've discussed "the state" $Q^+$, projections like $Q^+(s, a)$, and operations like MemoryUpdate. But what exactly **is** the agent's state in GRL?

This is not a philosophical question—it's a precise technical question with important implications:

- What object encodes the agent's knowledge?
- What changes when the agent learns?
- What stays fixed during inference?

This chapter provides definitive answers by clarifying:

1. **The agent's state** = particle memory = reinforcement field
2. **Belief evolution** = MemoryUpdate as state transition operator
3. **The role of weights** = implicit GP-derived coefficients, not learned parameters
4. **Three distinct operations** = fixing state, querying state, evolving state

**This resolves a common confusion:** mixing up the state (what the agent knows) with observations (what the agent queries).

---

## 1. What Is "The State" in GRL?

### The Question

In traditional RL, "state" usually means environment state $s \in \mathcal{S}$.

But in GRL, we have multiple candidates:

- Environment state $s$?
- Augmented point $z = (s, a)$?
- Field value $Q^+(s, a)$?
- Action projection $Q^+(s, \cdot)$?
- The entire field $Q^+$?

**Which one is "the agent's state"?**

---

### The Answer: The Entire Field

**The agent's state is the reinforcement field:**

$$Q^+ = \sum_{i=1}^N w_i k(z_i, \cdot) \in \mathcal{H}_k$$

**Why the entire field?**

Because this object **completely specifies** the agent's beliefs about value across all state-action combinations.

**Equivalent representation:** Particle memory

$$\Omega = \{(z_i, w_i)\}_{i=1}^N$$

**Key equation:**

$$\Omega \Longleftrightarrow Q^+$$

These are **two views of the same object:**

- $\Omega$ = discrete representation (particles)
- $Q^+$ = continuous representation (field)

---

### Why Not Something Smaller?

**Q: Why isn't $Q^+(s, \cdot)$ the state?**

**A:** Because $Q^+(s, \cdot)$ is a **projection** of the state onto a particular subspace, not the state itself.

**Analogy to QM:**

| Quantum Mechanics | GRL |
|-------------------|-----|
| State: $\|\psi\rangle \in \mathcal{H}$ | State: $Q^+ \in \mathcal{H}_k$ |
| Wavefunction: $\psi(x) = \langle x \| \psi \rangle$ | Field value: $Q^+(s, a) = \langle Q^+, k((s,a), \cdot) \rangle$ |
| Position representation | Augmented space representation |

**The wavefunction $\psi(x)$ is not the state—it's a coordinate representation of the state.**

**Same in GRL:** $Q^+(s, a)$ is not the state—it's a coordinate representation (amplitude) of the state.

---

### Particle Memory IS the State

**Critical insight:**

$$\text{The particle set } \{(z_i, w_i)\}_{i=1}^N \text{ completely determines } Q^+$$

**What the particles encode:**

| Component | Meaning | Type |
|-----------|---------|------|
| $z_i = (s_i, a_i)$ | Where experience occurred | Position in $\mathcal{Z} = \mathcal{S} \times \Theta$ |
| $w_i$ | Evidence strength at $z_i$ | Real number (positive or negative) |
| $k(z_i, \cdot)$ | Kernel section | Basis function in $\mathcal{H}_k$ |

**From particles to field:**

$$Q^+ = \sum_{i=1}^N w_i k(z_i, \cdot)$$

**This representation is complete!** You can compute $Q^+(z)$ for any $z$:

$$Q^+(z) = \langle Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k} = \sum_{i=1}^N w_i k(z_i, z)$$

---

### What About $Q^+(z_i)$ at the Particle Locations?

**Question:** "Should we store $Q^+(z_i)$ as part of the particle?"

**Answer:** **No, it's redundant!**

$Q^+(z_i)$ is **computable** from the particles:

$$Q^+(z_i) = \sum_{j=1}^N w_j k(z_j, z_i)$$

**So the particle representation is:**

$$\text{Particle } i: (z_i, w_i)$$

NOT:

$$\text{Particle } i: (z_i, w_i, Q^+(z_i)) \leftarrow \text{redundant!}$$

**What $w_i$ represents:**

- **Original paper:** Fitness contribution
- **Modern framing:** Energy contribution (negative fitness: $E(z_i) = -w_i k(z_i, z_i)$)
- **Mathematically:** RKHS expansion coefficient

---

## 2. Three Distinct Operations

Now that we know the state is $Q^+$ (equivalently: $\Omega$), let's clarify three operations that are often confused:

### Operation A: Fixing the Belief State

**At time $t$, given particle memory $\Omega_t$:**

$$\Omega_t = \{(z_i^{(t)}, w_i^{(t)})\}_{i=1}^{N_t}$$

**This fixes the belief state:**

$$Q^+_t = \sum_{i=1}^{N_t} w_i^{(t)} k(z_i^{(t)}, \cdot)$$

**Meaning:** "Conditional on the current memory, the agent's knowledge is $Q^+_t$."

**This is NOT learning—it's just stating what the current belief is.**

---

### Operation B: Querying the State (Inference)

**Given fixed $Q^+_t$, compute:**

$$Q^+_t(s, a) = \langle Q^+_t, k((s,a), \cdot) \rangle_{\mathcal{H}_k}$$

or action wavefunction (from Chapter 4):

$$\psi_{s,t}(a) = Q^+_t(s, a)$$

or concept activation (from Chapter 5):

$$A_{k,t} = \|P_k Q^+_t\|^2$$

**Key point:** These operations **do not change $Q^+_t$!**

They are:

- Queries
- Projections
- Evaluations
- Inferences

**Analogy:** Computing $\psi(x) = \langle x | \psi \rangle$ doesn't change $|\psi\rangle$.

**This is pure inference, no learning.**

---

### Operation C: Evolving the State (Learning via MemoryUpdate)

**MemoryUpdate transforms the belief state:**

$$\mathcal{U}: Q^+_t \mapsto Q^+_{t+1}$$

or equivalently:

$$\Omega_t \xrightarrow{\text{MemoryUpdate}} \Omega_{t+1}$$

**What can change:**

- **Add particles:** $\Omega_{t+1} = \Omega_t \cup \{(z_{new}, w_{new})\}$
- **Update weights:** $w_i^{(t+1)} = w_i^{(t)} + \Delta w_i$
- **Merge particles:** Combine nearby particles into one
- **Prune particles:** Remove low-influence particles

**Result:** New belief state $Q^+_{t+1} \neq Q^+_t$

**This IS learning!**

---

### Summary Table

| Operation | Changes $Q^+$? | Purpose |
|-----------|----------------|---------|
| **A. Fix state** | No (just specify current state) | Define what agent knows |
| **B. Query state** | No (projection/evaluation) | Action selection, concept activation |
| **C. Evolve state** | Yes (belief update) | Learning from experience |

**Critical distinction:**

> **Between MemoryUpdate events, $Q^+$ is fixed. During MemoryUpdate, $Q^+$ evolves.**

---

## 3. Two Time Scales

This gives GRL a natural separation of time scales:

### Slow Time Scale: Belief Evolution

**MemoryUpdate events:** $t = 0, 1, 2, \ldots$

**State transitions:**

$$Q^+_0 \xrightarrow{\mathcal{U}_1} Q^+_1 \xrightarrow{\mathcal{U}_2} Q^+_2 \xrightarrow{\mathcal{U}_3} \cdots$$

**This is learning.**

**Frequency:** Every episode, or every $K$ steps, or based on novelty

---

### Fast Time Scale: Inference

**Between $t$ and $t+1$, $Q^+_t$ is fixed.**

**Agent performs many queries:**

- Evaluate $Q^+_t(s_1, a)$ for action selection at $s_1$
- Evaluate $Q^+_t(s_2, a)$ for action selection at $s_2$
- Compute concept activation $A_{k,t}$
- Sample from policy $\pi_t(a|s) \propto \exp(\beta Q^+_t(s, a))$

**This is inference.**

**Frequency:** Every step, or multiple times per step

---

### Why This Matters

**Separation of concerns:**

- **Learning:** Happens via MemoryUpdate (slow)
- **Acting:** Happens via inference (fast)
- **No gradient descent** mixing learning and inference

**Computational efficiency:**

- Don't recompute entire field for every action
- Cache kernel evaluations between updates
- Amortize expensive operations (merging, pruning)

**Theoretical clarity:**

- Clean POMDP interpretation (belief state = $Q^+$)
- Well-defined state transition operator ($\mathcal{U}$)
- No ambiguity about "what changed"

---

## 4. The Role of Weights: Implicit, Not Learned

### Common Misconception

**Misconception:** "The weights $w_i$ are learned parameters, like neural network weights."

**Reality:** The weights are **implicit coefficients** determined by the GP posterior, not explicit optimization variables.

---

### How Weights Arise

**In Gaussian Process regression:**

Given data $\mathcal{D} = \{(z_i, y_i)\}$ and kernel $k$, the posterior mean is:

$$\mu(z) = \mathbf{k}(z)^T (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$$

where:

- $\mathbf{k}(z) = [k(z_1, z), \ldots, k(z_N, z)]^T$
- $\mathbf{K}_{ij} = k(z_i, z_j)$
- $\mathbf{y} = [y_1, \ldots, y_N]^T$

**This can be written as:**

$$\mu(z) = \sum_{i=1}^N w_i k(z_i, z)$$

where $\mathbf{w} = (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$.

**The weights $w_i$ are not learned—they're computed from the data and kernel!**

---

### In GRL

**Similar structure:**

$$Q^+(z) = \sum_{i=1}^N w_i k(z_i, z)$$

**The weights arise from:**

1. **Experience accumulation:** Each $(z_i, r_i)$ contributes
2. **Kernel propagation:** Overlap spreads influence
3. **TD updates:** Temporal difference signals adjust weights

**They are NOT:**

- Gradient descent parameters
- Explicitly optimized
- Independent of the kernel structure

**They ARE:**

- **State variables** (part of the belief state)
- **Functionally determined** by experience and kernel
- **Evidence coefficients** (strength of belief at each particle)

---

### Representer Theorem Connection

**The representer theorem** says:

> In RKHS, any function minimizing a regularized loss can be written as a finite sum over data points:

$$f^* = \sum_{i=1}^N \alpha_i k(x_i, \cdot)$$

**In GRL:**

- Data points = experience particles $z_i$
- Coefficients = weights $w_i$
- Function = reinforcement field $Q^+$

**So the particle representation is not arbitrary—it's the optimal form given the RKHS structure!**

---

## 5. MemoryUpdate as Belief Transition Operator

### Formal Definition

**MemoryUpdate is an operator:**

$$\mathcal{U}: \mathcal{H}_k \to \mathcal{H}_k$$

$$Q^+_t \mapsto Q^+_{t+1}$$

**In particle coordinates:**

$$\mathcal{U}: \{(z_i^{(t)}, w_i^{(t)})\} \mapsto \{(z_j^{(t+1)}, w_j^{(t+1)})\}$$

---

### What MemoryUpdate Does (Algorithm 1)

From Tutorial Chapter 6, MemoryUpdate performs:

**Step 1: Particle instantiation**

Given experience $(s_t, a_t, r_t)$, create:

$$z_{new} = (s_t, a_t), \quad w_{new} = f(r_t)$$

where $f(\cdot)$ maps reinforcement to weight (e.g., $f(r) = r$, or $f(r) = -r$ for energy).

---

**Step 2: Kernel association**

Compute similarity to existing particles:

$$a_i = k(z_{new}, z_i^{(t)})$$

---

**Step 3: Weight propagation (optional)**

For particles with high association:

$$w_i^{(t+1)} = w_i^{(t)} + \lambda \cdot a_i \cdot w_{new}$$

**This is "experience association"—evidence spreads through kernel geometry!**

---

**Step 4: Memory integration**

$$\Omega_{t+1} = \Omega_t \cup \{(z_{new}, w_{new})\}$$

---

**Step 5: Structural consolidation**

- **Merge:** Combine particles with $k(z_i, z_j) > \tau_{merge}$
- **Prune:** Remove particles with $|w_i| < \tau_{prune}$
- **Decay:** $w_i^{(t+1)} = \gamma w_i^{(t)}$ for all $i$

---

**Result:**

$$Q^+_{t+1} = \sum_{j=1}^{N_{t+1}} w_j^{(t+1)} k(z_j^{(t+1)}, \cdot)$$

**This is a discrete, explicit state transition!**

---

### Connection to Gaussian Process Updates

**MemoryUpdate can be viewed as:**

> **GP posterior update expressed in particle (inducing point) coordinates**

**Standard GP update:**

1. Observe new data: $(z_{new}, y_{new})$
2. Update posterior: $p(f | \mathcal{D}_{t+1}) \propto p(y_{new} | f, z_{new}) \cdot p(f | \mathcal{D}_t)$

**GRL equivalent:**

1. Observe new experience: $(z_{new}, r_{new})$
2. Update particle memory: $\Omega_{t+1}$ (via MemoryUpdate)
3. Resulting field: $Q^+_{t+1}$

**Key difference:** GRL also includes:

- Weight propagation (kernel association)
- Structural consolidation (merge/prune)

**These are not standard GP operations, but natural extensions for lifelong learning!**

---

## 6. Experience Association: What It Really Is

### The Original Paper's Description

Section IV-A describes "experience association"—new experience affects nearby particles through kernel overlap.

**Formalizing experience association:**

---

### Experience Association as Operator

**Experience association is the weight propagation step in MemoryUpdate:**

$$\mathcal{A}: (Q^+_t, z_{new}, w_{new}) \mapsto Q^+_t + \Delta Q^+$$

where:

$$\Delta Q^+ = \sum_{i: a_i > \varepsilon} (\lambda \cdot a_i \cdot w_{new}) \, k(z_i, \cdot)$$

**In words:**

- New evidence at $z_{new}$ with strength $w_{new}$
- Propagates to associated particles $z_i$ (where $a_i = k(z_{new}, z_i) > \varepsilon$)
- Strength of propagation: $\lambda \cdot a_i \cdot w_{new}$

---

### Why This Differs from Standard GP

**Standard GP:** Each data point contributes independently

$$\mu(z) = \sum_i \alpha_i k(z_i, z)$$

where $\alpha_i$ depends only on observation $y_i$ and regularization.

**GRL with experience association:** Data points influence each other's weights

$$w_i^{(t+1)} = w_i^{(t)} + \lambda \sum_{j: \text{new}} a_{ij} w_j^{\text{new}}$$

**This is a form of:**

- **Soft credit assignment** (not just local TD error)
- **Geometric belief propagation** (through kernel metric)
- **Non-local update** (affects multiple particles simultaneously)

---

### Connection to Kernel-Based Message Passing

**This is similar to:**

- Kernel mean embedding updates
- Belief propagation in continuous spaces
- Kernel density estimation with adaptive weights

**But GRL's version is unique because:**

- Weights can be positive or negative (not just probabilities)
- Propagation is kernel-weighted (not uniform or discrete)
- Updates are compositional (new evidence builds on old)

---

## 7. Reconciling with Quantum Mechanics

### The QM Analogy, Precisely Stated

**In quantum mechanics:**

**State:** $|\psi\rangle \in \mathcal{H}$ (Hilbert space vector)

**Evolution:** Unitary operators (between measurements)

$$|\psi(t)\rangle = e^{-iHt/\hbar} |\psi(0)\rangle$$

**Measurement:** Projects onto observable eigenspace

$$p(\lambda) = |\langle \lambda | \psi \rangle|^2$$

**State "fixed":** Between measurements

---

**In GRL:**

**State:** $Q^+ \in \mathcal{H}_k$ (RKHS vector) ≡ particle memory $\Omega$

**Evolution:** MemoryUpdate operator (between inference queries)

$$Q^+_{t+1} = \mathcal{U}(Q^+_t, \text{experience}_t)$$

**Measurement:** Projects onto query subspaces

$$Q^+(s, a) = \langle Q^+, k((s,a), \cdot) \rangle$$

**State "fixed":** Between MemoryUpdate events

---

### The Parallel Is Structural, Not Metaphorical

| Aspect | Quantum Mechanics | GRL |
|--------|-------------------|-----|
| **State space** | Hilbert space $\mathcal{H}$ | RKHS $\mathcal{H}_k$ |
| **State vector** | $\|\psi\rangle$ | $Q^+$ (or $\Omega$) |
| **Basis** | $\{\|x\rangle\}$ | $\{k(z, \cdot)\}$ |
| **Coordinate rep** | $\psi(x) = \langle x \| \psi \rangle$ | $Q^+(z) = \langle Q^+, k(z, \cdot) \rangle$ |
| **Evolution** | Hamiltonian $\hat{H}$ | MemoryUpdate $\mathcal{U}$ |
| **Measurement** | Observable $\hat{O}$ | Projection $P_k$ or query |
| **Time scales** | Between measurements: fixed | Between updates: fixed |

**This is not poetry—it's the same mathematical structure!**

---

## 8. Practical Implications

### For Implementation

**Representation choice:**

Store particles, not the full field:

```python
class BeliefState:
    def __init__(self):
        self.particles = []  # List of (z_i, w_i)
        
    def query(self, z_query):
        """Compute Q^+(z_query) from particles"""
        return sum(w_i * kernel(z_i, z_query) 
                   for z_i, w_i in self.particles)
    
    def update(self, experience):
        """MemoryUpdate: evolve belief state"""
        z_new, r_new = experience
        w_new = r_new  # or more complex mapping
        
        # Particle instantiation
        self.particles.append((z_new, w_new))
        
        # Experience association (weight propagation)
        for i, (z_i, w_i) in enumerate(self.particles[:-1]):
            a_i = kernel(z_new, z_i)
            if a_i > epsilon:
                self.particles[i] = (z_i, w_i + lambda_prop * a_i * w_new)
        
        # Structural consolidation
        self.merge_particles()
        self.prune_particles()
```

---

### For Efficiency

**Between MemoryUpdate:**

- Cache kernel evaluations
- Precompute Gram matrix if needed
- Use sparse representations for large particle sets

**During MemoryUpdate:**

- Only update associated particles (threshold $\varepsilon$)
- Merge periodically, not every step
- Use KD-trees for fast nearest-neighbor finding

---

### For Interpretation

**Visualize belief evolution:**

```python
# Track field value at specific points over time
history = []
for t in range(T):
    state_t = agent.belief_state.query(z_test)
    history.append(state_t)
    
    # Agent acts, observes, learns
    experience = agent.interact(env)
    agent.belief_state.update(experience)

# Plot belief evolution
plt.plot(history)
plt.xlabel('Time (MemoryUpdate events)')
plt.ylabel('Q^+(z_test)')
plt.title('Belief Evolution at Test Point')
```

---

## Summary

### Key Concepts

1. **The Agent's State**

   - State = reinforcement field $Q^+ \in \mathcal{H}_k$
   - Equivalently: particle memory $\Omega = \{(z_i, w_i)\}$
   - Complete representation: particles determine field

2. **Three Operations**

   - **Fix state:** Specify current belief (Operation A)
   - **Query state:** Compute projections/evaluations (Operation B)
   - **Evolve state:** MemoryUpdate (Operation C)

3. **Two Time Scales**

   - **Slow:** Learning via MemoryUpdate ($Q^+_t \to Q^+_{t+1}$)
   - **Fast:** Inference via queries ($Q^+_t(s, a)$, fixed $Q^+_t$)

4. **Weights Are Implicit**

   - Not learned parameters
   - GP-derived coefficients
   - State variables, not optimization variables

5. **MemoryUpdate as Operator**

   - Belief state transition: $\mathcal{U}: Q^+_t \mapsto Q^+_{t+1}$
   - Includes: instantiation, association, consolidation
   - Experience association = weight propagation

6. **QM Parallel**

   - Same structure: state vector in Hilbert space
   - Evolution via operators
   - Fixed between update/measurement events

---

### Key Equations

**State specification:**

$$\Omega = \{(z_i, w_i)\}_{i=1}^N \Longleftrightarrow Q^+ = \sum_{i=1}^N w_i k(z_i, \cdot)$$

**Query (inference):**

$$Q^+(z) = \langle Q^+, k(z, \cdot) \rangle = \sum_i w_i k(z_i, z)$$

**Evolution (learning):**

$$\mathcal{U}: Q^+_t \mapsto Q^+_{t+1}$$

**Experience association:**

$$w_i^{(t+1)} = w_i^{(t)} + \lambda \cdot k(z_{new}, z_i) \cdot w_{new}$$

---

### What This Clarifies

**For theory:**

- Rigorous definition of "the state"
- Clean separation of learning and inference
- Well-defined belief evolution operator
- Precise QM parallel

**For implementation:**

- What to store (particles)
- What to compute (queries)
- When to update (MemoryUpdate events)
- How to optimize (caching, sparse ops)

**For Part II (Section V):**

- Concept activation operates on fixed $Q^+$
- Concept evolution tracks $A_k(t)$ over MemoryUpdate events
- Clean distinction between concept inference and concept learning

---

## Further Reading

### Within This Series

- **[Chapter 2](02-rkhs-basis-and-amplitudes.md):** RKHS Basis and Amplitudes
- **[Chapter 4](04-action-and-state-fields.md):** Action and State Projections
- **[Chapter 5](05-concept-projections-and-measurements.md):** Concept Subspaces

### GRL Tutorials

- **[Tutorial Chapter 5](../tutorials/05-particle-memory.md):** Particle Memory
- **[Tutorial Chapter 6](../tutorials/06-memory-update.md):** MemoryUpdate Algorithm

### Related Literature

**Gaussian Processes:**

- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Qui & Candela (2005). "Sparse Gaussian Processes using Pseudo-inputs." *NIPS*.

**Belief-State RL:**

- Kaelbling et al. (1998). "Planning and Acting in Partially Observable Stochastic Domains."
- Ross et al. (2008). "Online Planning Algorithms for POMDPs."

**Kernel Methods:**

- Schölkopf & Smola (2002). *Learning with Kernels*. MIT Press.
- Berlinet & Thomas-Agnan (2004). *Reproducing Kernel Hilbert Spaces in Probability and Statistics*.

---

**Last Updated:** January 14, 2026
