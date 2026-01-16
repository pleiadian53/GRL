# Chapter 4: Slicing the Reinforcement Field — Action, State, and Concept Projections

## Motivation

In [Chapter 2](02-rkhs-basis-and-amplitudes.md), we learned that the reinforcement field $Q^+$ is a **state in RKHS**, and $Q^+(z)$ is its coordinate representation at point $z = (s, a)$.

But $z$ has **two components**: state $s$ and action $a$.

This raises a natural question:

> **Can we slice $Q^+$ along just one dimension—fixing either state or action—to get different "views" of the same field?**

**Answer:** Yes! And this reveals a powerful structure that enables:

- **Action wavefunctions:** The landscape of possible actions at a given state
- **State wavefunctions:** The applicability of an action across states
- **Concept projections:** Hierarchical abstractions via subspace projections

This chapter shows how **one state $Q^+$ gives rise to multiple coordinate representations**, each useful for different purposes.

---

## 1. The Foundation: One State, Many Projections

### Quantum Mechanics Parallel

In QM, you have **one state** $|\psi\rangle$, but you can view it in different bases:

**Position representation:**

$$\psi(x) = \langle x | \psi \rangle$$

**Momentum representation:**

$$\tilde{\psi}(p) = \langle p | \psi \rangle$$

**Key point:** Same state, different coordinate systems.

---

### GRL Analogue

In GRL, you have **one field** $Q^+ \in \mathcal{H}_k$, but you can project it onto different subspaces:

**Full augmented space:**

$$Q^+(s, a) = \langle Q^+, k((s,a), \cdot) \rangle_{\mathcal{H}_k}$$

**Action slice (fix state):**

$$\psi_s(a) = Q^+(s, a) \quad \text{for fixed } s$$

**State slice (fix action):**

$$\phi_a(s) = Q^+(s, a) \quad \text{for fixed } a$$

**Concept projection (subspace):**

$$Q^+_{\text{concept}} = P_k Q^+$$

**Key insight:** These are **not different states**—they're different **projections** of the same state $Q^+$.

---

### The Critical Distinction

Before we proceed, let's be absolutely clear about what's what:

| Object | Type | Role |
|--------|------|------|
| $Q^+ \in \mathcal{H}_k$ | **State** (the thing) | The agent's knowledge, encoded as RKHS element |
| $k(z, \cdot)$ | **Basis element** (frame) | Representational scaffolding, fixed |
| $Q^+(z)$ | **Coordinate** (view) | Value of state in direction $k(z, \cdot)$ |

**Crucial:**

> **Basis vectors are fixed questions; the state is the system's answer.**

When you compute $Q^+(s, a)$, you're not creating a new state—you're asking: **"What does the current state look like from this point of view?"**

---

## 2. Action Wavefunction: $\psi_s(a) = Q^+(s, a)$

### Definition

**Fix the state** $s$ and let action $a$ vary:

$$\psi_s: \Theta \to \mathbb{R}$$
$$\psi_s(a) := Q^+(s, a)$$

This gives you a **function over the action space** for a specific state.

---

### What It Represents

$\psi_s(a)$ is an **amplitude field over actions**, not a policy yet.

**It encodes:**

- Compatibility between past experience and candidate actions
- Multi-modality (multiple plausible actions can have high amplitude)
- Smooth generalization across actions via kernel overlap

**Visual intuition:**

```
ψ_s(a)
  ^
  |        *         *
  |       ***       ***
  |      *****     *****
  |     *******   *******
  +--------------------------> a
      a₁        a₂
```

Two peaks suggest two good actions at state $s$.

---

### From Amplitude to Policy

To get a policy, apply the Boltzmann transformation:

$$\pi(a | s) = \frac{\exp(\beta \, \psi_s(a))}{\int_{\Theta} \exp(\beta \, \psi_s(a')) da'}$$

**Analogy to QM:**

- Wavefunction $\psi_s(a)$ → amplitude landscape
- Boltzmann rule → measurement process (like Born rule)

---

### Practical Use Cases

#### **1. Continuous Control**

**Standard RL:** Pick $a^* = \arg\max_a Q(s, a)$ (one point)

**With action wavefunction:** See the **full landscape**

- Multiple peaks → multiple strategies
- Smooth ridges → continuous manifolds of good actions
- Natural for robotics and motor control

**Example:** Robot grasping
- $\psi_s(a)$ shows all viable grips
- Controller can smoothly interpolate between them

---

#### **2. Action Substitution**

If two actions have high kernel overlap, they appear as **neighboring peaks** in $\psi_s(a)$.

**This gives graded replaceability:**

- Action $a_1$ unavailable? Use nearby $a_2$ with similar amplitude
- No need for discrete "backup plans"—structure is inherent

**Example:** Tool use
- "Hammer" unavailable → "mallet" is nearby in action space
- Amplitude field shows compatibility automatically

---

#### **3. Exploration Strategies**

**Standard:** $\epsilon$-greedy (uniform noise) or Boltzmann (temperature)

**With action wavefunction:**

- Sample from high-amplitude **regions** (not just max)
- Preserves structure (won't explore clearly bad actions)
- Natural multi-modal exploration

**Example:** Game playing
- Multiple openings have high amplitude
- Explore all of them proportional to their support

---

#### **4. Option Discovery**

**Observation:** Peaks in $\psi_s(a)$ correspond to **natural options**

**Algorithm:**

1. Compute $\psi_s(a)$ for visited states $s$
2. Cluster peaks across states
3. Each cluster = a discovered option

**Why this works:** Options are actions that "make sense together" across states—exactly what amplitude clustering captures.

---

## 3. State Wavefunction: $\phi_a(s) = Q^+(s, a)$

### Definition

Now reverse the roles: **fix the action** $a$ and let state $s$ vary:

$$\phi_a: \mathcal{S} \to \mathbb{R}$$
$$\phi_a(s) := Q^+(s, a)$$

This gives you a **function over the state space** for a specific action.

---

### What It Represents

$\phi_a(s)$ is an **applicability amplitude**—it answers:

> "Where in state space does this action **make sense**?"

**High values mean:**

- This action has historically aligned well with similar states
- Kernel overlap supports generalization here
- Preconditions (implicit) are satisfied

**Low values mean:**

- Action is structurally incompatible with this region
- Either never tried here, or tried and failed

**Visual intuition:**

```
State space:

  s₂ |  ···  ███████  ···   ← φ_a(s) high
     |       ███████
  s₁ |       ███████
     +------------------------
         Region where action a works
```

---

### Why This Is Novel

**Standard RL:** Actions are arbitrary choices—any action can be taken anywhere (modulo physical constraints).

**GRL with state wavefunction:** Actions have **applicability regions**—some actions "belong" in certain states.

**This is closer to human reasoning:**

> "This tool works **here**, not **there**."

---

### Practical Use Cases

#### **1. Implicit Precondition Learning**

**Without symbolic rules,** learn where actions apply.

**Example:** Robotics
- "Open door" action: $\phi_{\text{open}}(s)$ high when door is closed and reachable
- "Pour liquid" action: $\phi_{\text{pour}}(s)$ high when container is held upright

**No hand-coded logic—just learned structure!**

---

#### **2. Affordance Maps**

In robotics, $\phi_a(s)$ **is literally an affordance map**.

**Affordance:** "What can I do here?"

**GRL answer:** For each action $a$, $\phi_a(s)$ shows where it affords interaction.

**Example:** Navigation
- $\phi_{\text{walk}}(s)$: high on floors, low on walls
- $\phi_{\text{grasp}}(s)$: high near objects, low in empty space

---

#### **3. Safety Constraints**

**Define:** An action $a$ is **unsafe** at state $s$ if $\phi_a(s) < \tau$ (threshold).

**This gives implicit safety without explicit rules:**

- Low amplitude → historically bad outcomes
- Safe policy: only consider actions with $\phi_a(s) > \tau$

**Example:** Autonomous driving
- $\phi_{\text{accelerate}}(s)$: low when obstacle ahead
- Learned from experience, not programmed

---

#### **4. World Model Filtering**

**Setup:** Model-based RL with a world model predicting future states

**Challenge:** Model might predict physically possible but **practically infeasible** states

**Solution:** Filter by action applicability
- World model: "You could be in state $s'$"
- GRL: "But action $a$ doesn't work there" (low $\phi_a(s')$)
- Keep only reachable states

---

#### **5. Skill Discovery**

**Observation:** Actions with **compact support** (high $\phi_a$ in small region) are natural skills.

**Algorithm:**

1. Compute $\phi_a(s)$ for all actions
2. Find actions with localized high-amplitude regions
3. Each compact region defines a skill

**Why:** Skills are actions that work in specific contexts—exactly what state wavefunctions capture.

---

## 4. Concept Subspace Projections: $P_k Q^+$

### Beyond Pointwise Projections

So far, we've projected onto **pointwise bases** $k(z, \cdot)$.

Now let's project onto **subspaces** discovered by spectral clustering (Section V of the original paper).

---

### Definition

Suppose spectral clustering identifies an eigenspace:

$$\mathcal{C}_k \subset \mathcal{H}_k$$

spanned by eigenfunctions $\{\phi_1, \ldots, \phi_m\}$ associated with a functional cluster.

**Project the field onto this subspace:**

$$P_k Q^+ = \sum_{i=1}^m \langle Q^+, \phi_i \rangle \phi_i$$

This is **still the same state**, just viewed through a **coarse lens**.

---

### What This Represents

A concept-level amplitude field answers:

> "How strongly does the current situation activate this **concept**?"

**Not:**

- ❌ A discrete label
- ❌ A hard cluster assignment
- ❌ A symbolic rule

**But:**

- ✅ A **graded, geometric activation**

- ✅ Smooth interpolation between concepts
- ✅ Hierarchical structure

---

### Connection to Concepts

**Traditional clustering:** Assign data point to one cluster

**GRL concepts:** Compute activation of each concept subspace

$$\text{activation}_k = \|P_k Q^+\|_{\mathcal{H}_k}$$

**This is a functional, not discrete, view of concepts.**

---

### Practical Use Cases

#### **1. Hierarchical Decision Making**

**High-level:** Concept activations determine strategy

**Low-level:** Action wavefunction $\psi_s(a)$ implements strategy

**Example:** Navigation
- High-level concept: "In doorway" (concept subspace)
- Low-level action: "Turn left" (action wavefunction)
- Concept modulates action field

---

#### **2. Interpretability**

Concept activation curves over time are **interpretable trajectories**:

```
Concept activation
  ^
  | C₁ ──────╮
  |          ╰─ C₂ ─────╮
  |                     ╰─ C₃
  +-------------------------> time
```

"Agent transitioned from concept C₁ to C₂ to C₃"

**No symbolic labels needed—just geometric structure!**

---

#### **3. Transfer Learning**

**Observation:** Concept subspaces are **more stable** than raw particles.

**Why:** Concepts capture abstract structure, not specific experiences.

**Transfer algorithm:**

1. Learn concept subspaces in source task
2. Transfer subspaces to target task
3. Re-learn particle weights within subspaces

**This is natural compositional transfer.**

---

## 5. Unifying View: One State, Multiple Projections

### The Common Structure

All three cases share the same mathematical structure:

1. There is **one GRL state** $Q^+ \in \mathcal{H}_k$
2. You never clone it or create variants
3. You only **project**, **slice**, or **coarse-grain** it

**This mirrors quantum mechanics exactly:**

| Quantum Mechanics | GRL (RKHS) |
|-------------------|------------|
| State vector $\|\psi\rangle$ | Reinforcement field $Q^+$ |
| Basis $\|x\rangle$ | Kernel basis $k(z, \cdot)$ |
| Wavefunction $\psi(x) = \langle x \| \psi \rangle$ | Field value $Q^+(z) = \langle Q^+, k(z, \cdot) \rangle$ |
| Observable (position, momentum) | Projection (action, state, concept) |

---

### Projection Operations Summary

| Projection Type | Operation | Result | Use Case |
|----------------|-----------|--------|----------|
| **Full field** | $Q^+(s, a)$ | Value at point $(s, a)$ | Standard RL value |
| **Action slice** | $\psi_s(a) = Q^+(s, a)$ | Action landscape at $s$ | Continuous control, exploration |
| **State slice** | $\phi_a(s) = Q^+(s, a)$ | Applicability of $a$ across states | Preconditions, affordances, safety |
| **Concept subspace** | $P_k Q^+$ | Concept activation | Hierarchical RL, interpretability, transfer |

---

### Why This Matters

**Once you see this structure, the QM analogy stops being decorative and starts being **constraining**—in a good way.**

**It tells you:**

- How to define new projections (other subspaces?)
- Why actions and states are **dual** under projection
- How to compose operations (project, then project again?)

---

## 6. Action-State Duality

### The Symmetry

Notice the beautiful symmetry:

**Action wavefunction:**

$$\psi_s(a) = Q^+(s, a) \quad \text{(fix } s\text{, vary } a\text{)}$$

**State wavefunction:**

$$\phi_a(s) = Q^+(s, a) \quad \text{(fix } a\text{, vary } s\text{)}$$

**These are the same function, just indexed differently!**

---

### Deeper Implications

This suggests actions and states might be more symmetric than traditional RL assumes.

**Standard RL:**

- States = where you are
- Actions = what you do
- Asymmetric roles

**GRL:**

- States and actions are **coordinates** in augmented space $\mathcal{Z} = \mathcal{S} \times \Theta$
- Projections along either dimension are equally valid
- **Symmetric structure**

**This could enable:**

- **Dual learning:** Learn about actions by exploring states, and vice versa
- **Compositional policies:** Combine state-based and action-based representations
- **Transfer:** Actions discovered in one state space transfer to another

---

## 7. From Projections to Operators (Preview)

### The Next Level

So far, we've discussed **projections**—static views of $Q^+$.

**Natural question:** What **transforms** $Q^+$?

**Answer:** **Operators!**

**Examples:**

- **MemoryUpdate:** Operator that updates $Q^+$ given new experience
- **Action conditioning:** Operator that shifts field based on action
- **Abstraction:** Operator that projects onto concept subspaces

**This is where dynamics live—not in the state, but in the operators that transform it.**

---

### Operator Formalism (Preview)

In QM, observables are Hermitian operators $\hat{O}$.

In GRL, we can define:

**Memory update operator:**

$$\hat{M}(e): \mathcal{H}_k \to \mathcal{H}_k$$
$$Q^+ \mapsto Q^+ + w_{new} k(z_{new}, \cdot)$$

**Policy operator:**

$$\hat{\Pi}_s: \mathcal{H}_k \to L^2(\Theta)$$
$$Q^+ \mapsto \psi_s(a) = Q^+(s, a)$$

**This will be explored in future chapters!**

---

## Summary

### Key Concepts

1. **One State, Many Views**

   - $Q^+ \in \mathcal{H}_k$ is the state
   - $Q^+(z), \psi_s(a), \phi_a(s), P_k Q^+$ are projections

2. **Action Wavefunction** $\psi_s(a) = Q^+(s, a)$
   - Amplitude field over actions at state $s$
   - Use: continuous control, exploration, option discovery

3. **State Wavefunction** $\phi_a(s) = Q^+(s, a)$
   - Applicability of action $a$ across states
   - Use: preconditions, affordances, safety, skills

4. **Concept Projections** $P_k Q^+$
   - Subspace projections onto concept eigenspaces
   - Use: hierarchical RL, interpretability, transfer

5. **Action-State Duality**

   - Symmetric roles in augmented space
   - Enables dual learning and compositional policies

---

### Key Equations

**Action wavefunction:**

$$\psi_s(a) = Q^+(s, a) = \sum_i w_i k((s_i, a_i), (s, a))$$

**State wavefunction:**

$$\phi_a(s) = Q^+(s, a) = \sum_i w_i k((s_i, a_i), (s, a))$$

**Concept projection:**

$$P_k Q^+ = \sum_{j=1}^m \langle Q^+, \phi_j \rangle_{\mathcal{H}_k} \phi_j$$

**Policy from action wavefunction:**

$$\pi(a|s) = \frac{\exp(\beta \, \psi_s(a))}{\int_{\Theta} \exp(\beta \, \psi_s(a')) da'}$$

---

### What This Enables

**Theoretical:**

- Rigorous projection formalism
- Action-state duality
- Operator-based dynamics

**Practical:**

- Continuous control with full action landscapes
- Implicit precondition and affordance learning
- Hierarchical RL via concept activations
- Natural skill and option discovery
- Safety via applicability constraints

---

## Further Reading

### Within This Series

- **[Chapter 1a](01a-wavefunction-interpretation.md):** State Vector vs. Wavefunction
- **[Chapter 2](02-rkhs-basis-and-amplitudes.md):** RKHS Basis and Amplitudes
- **[Chapter 5 (future)](05-operators-and-dynamics.md):** Operators on the Reinforcement Field

### GRL Tutorials

- **[Tutorial Chapter 4](../tutorials/04-reinforcement-field.md):** Reinforcement Field Basics
- **[Tutorial Chapter 5](../tutorials/05-particle-memory.md):** Particle Memory

### Related Work

**Eigenoptions:**

- Machado et al. (2017). "A Laplacian Framework for Option Discovery." *ICML*.

**Affordances in RL:**

- Khetarpal et al. (2020). "What can I do here? A theory of affordances in reinforcement learning." *ICML*.

**Hierarchical RL:**

- Sutton et al. (1999). "Between MDPs and semi-MDPs: A framework for temporal abstraction in RL."

**Operator Formalism:**

- Barreto et al. (2017). "Successor Features for Transfer in Reinforcement Learning." *NIPS*.

---

**Last Updated:** January 14, 2026
