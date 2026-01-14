# Chapter 2: RKHS Basis, Kernel Amplitudes, and Why GRL Doesn't Need Normalization

## Motivation

In [Chapter 1a](01a-wavefunction-interpretation.md), we learned that quantum mechanics represents states in Hilbert space, with wavefunctions as coordinate representations in a chosen basis.

In [Chapter 1](01-rkhs-quantum-parallel.md), we stated that GRL's reinforcement field $Q^+ \in \mathcal{H}_k$ is analogous to a quantum state, with the field value $Q^+(z)$ analogous to a wavefunction $\psi(x)$.

This raises three fundamental questions:

1. **Basis:** What is the "basis" in RKHS, and how is it chosen?
2. **Amplitudes:** How do kernel evaluations relate to probability amplitudes?
3. **Probabilities:** Does GRL need normalized probabilities like quantum mechanics?

This chapter answers all three and reveals a surprising insight: **GRL combines QM's amplitude geometry with energy-based models' unnormalized inference**.

---

## 1. The RKHS Basis: Kernel Sections

### Quantum Mechanics Review

In QM, the position basis is:

$$\{|x\rangle : x \in \mathbb{R}\}$$

Each $|x\rangle$ is a basis vector representing "particle definitely at position $x$."

**To get coordinates:** Project the state $|\psi\rangle$ onto a basis vector:

$$\psi(x) = \langle x | \psi \rangle$$

**Key insight:** Choosing a specific $x$ doesn't define the whole basis—it **selects one basis vector** from the continuum.

---

### RKHS Analogue

In GRL, the RKHS $\mathcal{H}_k$ has an implicit basis (technically, a **frame**) given by **kernel sections**:

$$\{k(z, \cdot) : z \in \mathcal{Z}\}$$

where $\mathcal{Z} = \mathcal{S} \times \Theta$ is the augmented state-action space.

**What is $k(z, \cdot)$?**

For each point $z \in \mathcal{Z}$, the function $k(z, \cdot): \mathcal{Z} \to \mathbb{R}$ is an element of $\mathcal{H}_k$.

**Analogy:**

| Quantum Mechanics | GRL (RKHS) |
|-------------------|------------|
| $\|x\rangle$ = basis vector for position $x$ | $k(z, \cdot)$ = basis vector (frame element) for point $z$ |
| Position basis: $\{\|x\rangle : x \in \mathbb{R}\}$ | Kernel basis: $\{k(z, \cdot) : z \in \mathcal{Z}\}$ |
| State: $\|\psi\rangle \in \mathcal{H}$ | State: $Q^+ \in \mathcal{H}_k$ |
| Wavefunction: $\psi(x) = \langle x \| \psi \rangle$ | Field value: $Q^+(z) = \langle Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$ |

---

### To Get Coordinates: Evaluate the Inner Product

Given the state $Q^+ \in \mathcal{H}_k$ (determined by particles), to find its "coordinate" at query point $z$:

$$Q^+(z) = \langle Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$$

**What's happening:**

1. The **state** $Q^+$ is already fixed (by the particle memory)
2. Choosing $z$ **selects one frame element** $k(z, \cdot)$
3. The inner product gives the **amplitude** of $Q^+$ along that direction

**This is structurally identical to** $\psi(x) = \langle x | \psi \rangle$!

---

### So Is the Basis Determined by Choosing $z$?

**Short answer:** Yes—but **locally**, not globally.

**Long answer:**

The kernel-induced basis $\{k(z, \cdot)\}$ exists for **all** $z \in \mathcal{Z}$, just like the position basis exists for all $x \in \mathbb{R}$.

When you choose a specific $z$:
- You're not "defining the basis"
- You're **selecting which basis element** to project onto

**Analogy:** Asking "What is $\psi(3.2)$?" doesn't define the position basis—it just asks for the amplitude along the $|3.2\rangle$ direction.

Same here: $Q^+(z_0)$ asks for the amplitude along the $k(z_0, \cdot)$ direction.

---

## 2. Kernel Amplitudes: Overlap Without Probabilities

### Quantum Mechanics: Amplitudes → Probabilities

In QM:

**Primitive:** Complex amplitude $\psi(x)$

**Derived:** Probability via Born rule

$$p(x) = |\psi(x)|^2$$

**Normalization:**

$$\int_{-\infty}^{\infty} |\psi(x)|^2 dx = 1$$

**Key point:** Amplitude is fundamental; probability is derived by squaring and normalizing.

---

### RKHS: Amplitudes Without Probabilities

In RKHS, a kernel evaluation:

$$k(z_i, z) = \langle k(z_i, \cdot), k(z, \cdot) \rangle_{\mathcal{H}_k}$$

already looks like an **overlap amplitude**!

**But important differences:**

| Property | Quantum Mechanics | RKHS / GRL |
|----------|-------------------|------------|
| Values | Complex: $\psi(x) \in \mathbb{C}$ | Real: $Q^+(z) \in \mathbb{R}$ |
| Normalization | Required: $\int \|\psi(x)\|^2 dx = 1$ | Not required |
| Interpretation | Probability amplitude | Score / value / energy |

**Could you normalize?** Yes! You could define:

$$p(z) = \frac{|Q^+(z)|^2}{\int_{\mathcal{Z}} |Q^+(z')|^2 dz'}$$

This is mathematically valid—**and in fact, Born machines and quantum-inspired generative models do exactly this**.

---

### But GRL Doesn't Need This Step

**Why not?**

Because GRL uses $Q^+(z)$ for **decision-making and control**, not for sampling or probability estimation.

**Policy inference:**

$$\pi(\theta | s) \propto \exp(\beta \, Q^+(s, \theta))$$

**Action selection:**

$$\theta^* = \arg\max_\theta Q^+(s, \theta)$$

**Both use relative values, not normalized probabilities.**

This brings us to the third question...

---

## 3. Energy-Based Models: Unnormalized Scores Are Enough

### The EBM Perspective

Energy-based models (EBMs) define a probability distribution:

$$p(x) = \frac{1}{Z} \exp(-E(x))$$

where $Z = \int \exp(-E(x')) dx'$ is the partition function.

**The key insight:**

> **In practice, we never compute $Z$!**

Why? Because inference, optimization, and control only require **relative energies**:

**Comparing two options:**

$$\frac{p(x_1)}{p(x_2)} = \frac{\exp(-E(x_1))}{\exp(-E(x_2))} = \exp(E(x_2) - E(x_1))$$

The partition function $Z$ cancels out!

**Optimization:**

$$x^* = \arg\min_x E(x)$$

No normalization needed—just find the minimum.

**Control:**

$$u^* = \arg\min_u E(x, u)$$

Again, relative values are sufficient.

---

### GRL's Position in the Landscape

GRL does the same thing:

**The field $Q^+(z)$** acts as an **unnormalized score** / **negative energy**:

$$E(z) = -Q^+(z)$$

**Policy (Boltzmann distribution):**

$$\pi(\theta | s) \propto \exp(\beta \, Q^+(s, \theta))$$

No explicit normalization is computed—the policy is normalized implicitly when actions are sampled or probabilities are queried.

**Optimization (greedy action):**

$$\theta^* = \arg\max_\theta Q^+(s, \theta)$$

Only relative values matter.

---

### Where Does GRL Sit?

| Framework | Primitive | Normalized? | Use Case |
|-----------|-----------|-------------|----------|
| **Quantum Mechanics** | Amplitude $\psi(x)$ | Yes (Born rule) | Predictions about measurements |
| **Born Machines** | Amplitude $\psi(x)$ | Yes | Generative modeling |
| **Energy-Based Models** | Energy $E(x)$ | No | Optimization, inference |
| **GRL** | Field $Q^+(z)$ | No | Control, decision-making |

**The key insight:**

> **GRL borrows the amplitude geometry from quantum mechanics  
> and the unnormalized inference logic from energy-based models.**

This combination keeps GRL mathematically clean **without forcing probability normalization**.

---

## 4. Three Interpretations, One Object

The reinforcement field $Q^+$ can be viewed from three complementary perspectives:

### Perspective 1: Hilbert Space State

$$Q^+ \in \mathcal{H}_k$$

A vector in the RKHS, determined by particle memory:

$$Q^+ = \sum_{i=1}^N w_i \, k(z_i, \cdot)$$

**This is the abstract object**—basis-independent.

---

### Perspective 2: Amplitude Field (QM-like)

$$Q^+(z) = \langle Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$$

The coordinate representation in the kernel-induced basis.

**This gives the field value at each point**—the "wavefunction" of the field.

---

### Perspective 3: Energy / Score Function (EBM-like)

$$E(z) = -Q^+(z)$$

An unnormalized score used for inference and control.

**This enables decision-making without normalization.**

---

**Nothing is inconsistent here**—they're just different **readouts** of the same state, like:
- Abstract vector $\mathbf{v}$
- Cartesian coordinates $[v_x, v_y, v_z]$
- Potential energy $U(\mathbf{v})$

All describe the same object from different angles.

---

## 5. What GRL Does (and Doesn't) Claim

### GRL Does NOT Claim:

- ❌ $Q^+(z)$ is a probability
- ❌ Probabilities are derived via Born's rule $p(z) \propto |Q^+(z)|^2$
- ❌ Normalization is required for inference

### GRL DOES Claim:

- ✅ $Q^+$ is an element of a Hilbert space (RKHS)
- ✅ Kernel evaluations act as overlap amplitudes
- ✅ Action selection uses unnormalized scores (like EBMs)
- ✅ This unifies RKHS geometry, QM-style state representation, and EBM inference

---

### A Precise Statement

> **Although kernel evaluations resemble probability amplitudes, GRL does not require normalized probabilities. Inference and control are performed via unnormalized energy-like scores, consistent with modern energy-based models.**

This keeps the framework mathematically clean **and** modern.

---

## 6. Could We Define Probabilities? (Optional Extension)

**Your question:** "I think kernel functions have a similar property, but to construct valid probability, the square of kernel needs to be normalized."

**Answer:** Yes! You **could** define probabilities in RKHS. There are multiple options:

### Option 1: Born Rule on Kernel Amplitudes

$$p(z) = \frac{|Q^+(z)|^2}{\int_{\mathcal{Z}} |Q^+(z')|^2 dz'}$$

**This works!** Born machines use this approach.

**For GRL:**
- Pro: Direct QM analogy
- Con: Requires computing the normalization integral
- Con: Real-valued amplitudes don't give interference effects

---

### Option 2: Boltzmann Distribution (What GRL Uses)

$$\pi(\theta | s) = \frac{\exp(\beta Q^+(s, \theta))}{\int_{\Theta} \exp(\beta Q^+(s, \theta')) d\theta'}$$

**This is what GRL implicitly uses for policy!**

**Advantages:**
- Standard in RL (Boltzmann exploration)
- Temperature parameter $\beta$ controls exploration
- No need to compute normalization explicitly (ratio trick)

---

### Option 3: Complex-Valued RKHS (Advanced)

If you extend RKHS to **complex values**, you get:
- Probability amplitudes: $\psi(z) \in \mathbb{C}$
- Born rule: $p(z) = |\psi(z)|^2$
- **Interference effects** from phase!

**This is exactly what you explore in** `02-complex-rkhs.md`!

---

## 7. Three Roles of "Choosing a Point"

Let's unify the three concepts by understanding what "choosing $z$" means in each context:

| Concept | Question Answered | Example |
|---------|-------------------|---------|
| **Basis selection** | "Along which direction am I looking?" | $k(z, \cdot)$ is the direction |
| **Amplitude** | "How much does the state align with that direction?" | $Q^+(z) = \langle Q^+, k(z, \cdot) \rangle$ |
| **Probability** | "How often would I observe this if I sampled?" | $p(z) \propto \exp(\beta Q^+(z))$ (optional) |

**GRL needs the first two, not the third.**

---

## 8. Why This Matters for Your Framework

### Conceptual Clarity

Understanding that GRL works with **unnormalized amplitudes** (like EBMs) rather than **normalized probabilities** (like QM):

1. **Explains why no partition function is needed**
2. **Justifies the QM analogy** (geometry) without **overcommitting** (Born rule)
3. **Positions GRL correctly** in the modern ML landscape (alongside EBMs, score-based models)

---

### Future Extensions

If you want to develop normalized probability formulations:

**Option A:** Born rule approach (for generative modeling)

$$p(z) \propto |Q^+(z)|^2$$

**Option B:** Complex RKHS (for interference and phase semantics)

$$\psi(z) \in \mathbb{C}, \quad p(z) = |\psi(z)|^2$$

Both are natural extensions explored in later chapters!

---

### Practical Implications

**For implementation:**
- No need to compute partition functions
- Use relative values for action selection
- Policy normalization handled implicitly

**For theory:**
- Clean mathematical framework
- Rigorous Hilbert space structure
- Flexible enough for future probability extensions

---

## Summary

### Key Insights

1. **RKHS Basis:**
   - Kernel sections $\{k(z, \cdot)\}$ form a continuous frame
   - Choosing $z$ selects one frame element, like choosing $x$ selects $|x\rangle$ in QM
   - The state $Q^+$ exists independently; $Q^+(z)$ is its projection onto $k(z, \cdot)$

2. **Kernel Amplitudes:**
   - $Q^+(z)$ acts like a wavefunction (coordinate representation)
   - But GRL doesn't require normalization (unlike QM)
   - Similar to EBMs working with unnormalized energies

3. **Three Interpretations:**
   - Abstract: $Q^+ \in \mathcal{H}_k$ (Hilbert space state)
   - Coordinate: $Q^+(z)$ (amplitude field)
   - Inference: $-Q^+(z)$ (energy score)

4. **Why No Normalization:**
   - Decision-making uses relative values
   - Partition function cancels in ratios
   - Same principle as modern EBMs

5. **Could Define Probabilities:**
   - Born rule: $p(z) \propto |Q^+(z)|^2$ (possible, not required)
   - Boltzmann: $\pi(\theta|s) \propto \exp(\beta Q^+(s,\theta))$ (what GRL uses)
   - Complex RKHS: $p(z) = |\psi(z)|^2$ with interference (future extension)

---

## Key Equations

**RKHS basis (frame):**

$$\{k(z, \cdot) : z \in \mathcal{Z}\} \subset \mathcal{H}_k$$

**State in RKHS:**

$$Q^+ = \sum_{i=1}^N w_i \, k(z_i, \cdot) \in \mathcal{H}_k$$

**Coordinate representation (amplitude):**

$$Q^+(z) = \langle Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$$

**Energy interpretation:**

$$E(z) = -Q^+(z)$$

**Policy (Boltzmann):**

$$\pi(\theta | s) \propto \exp(\beta \, Q^+(s, \theta))$$

**Analogy to QM:**

| Quantum | RKHS |
|---------|------|
| $\|\psi\rangle \in \mathcal{H}$ | $Q^+ \in \mathcal{H}_k$ |
| $\psi(x) = \langle x \| \psi \rangle$ | $Q^+(z) = \langle Q^+, k(z, \cdot) \rangle$ |
| Basis: $\{\|x\rangle\}$ | Frame: $\{k(z, \cdot)\}$ |

---

## Further Reading

### Within This Tutorial Series

- **[Chapter 1](01-rkhs-quantum-parallel.md):** RKHS-Quantum Structural Parallel
- **[Chapter 1a](01a-wavefunction-interpretation.md):** What Is a Wavefunction?
- **[Chapter 3 (complex)](../quantum_inspired/03-complex-rkhs.md):** Complex-Valued RKHS and Probability Amplitudes

### Back to GRL Tutorials

- **[Tutorial Chapter 2](../tutorials/02-rkhs-foundations.md):** RKHS Foundations
- **[Tutorial Chapter 4](../tutorials/04-reinforcement-field.md):** Reinforcement Field

### Related Concepts

- **Energy-Based Models:** LeCun et al. (2006). "A Tutorial on Energy-Based Learning"
- **Born Machines:** Cheng et al. (2018). "Quantum Generative Adversarial Learning"
- **RKHS and Kernel Methods:** Berlinet & Thomas-Agnan (2004). "Reproducing Kernel Hilbert Spaces"

---

**Last Updated:** January 14, 2026
