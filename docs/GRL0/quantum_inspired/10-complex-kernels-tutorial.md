# Chapter 10: Leveraging Complex-Valued Kernels for GRL — A Practical Tutorial

**Prerequisites:** [Chapter 03 — Complex-Valued RKHS](03-complex-rkhs.md) (theory),
[Chapter 01a — Wavefunction Interpretation](01a-wavefunction-interpretation.md),
[Chapter 07 — Learning the Field Beyond GP](07-learning-the-field-beyond-gp.md).

**Companion notebook:** _planned_ `notebooks/field_series/03_complex_fields.ipynb`.

**Status:** 🔬 Tutorial draft — theory is established in Chapter 03; this document
focuses on how to actually use complex kernels in practice.

---

## 0. What This Tutorial Does

Chapter 03 established that we *can* move GRL from a real-valued reinforcement
field $Q^+(z) \in \mathbb{R}$ to a complex-valued wavefunction
$\Psi(z) \in \mathbb{C}$. This chapter answers the follow-up question:

> **Given that I have a complex kernel, what concretely changes in the GRL
> pipeline, and when do I gain something from it?**

We walk through five practical patterns, with pseudocode, gradient derivations,
pitfalls, and when *not* to reach for a complex kernel.

---

## 1. The Minimum Viable Change

The entire GRL machinery goes through with one substitution. The real-valued
reinforcement field

$$Q^+(z) = \sum_i w_i\, k(z, z_i), \qquad w_i \in \mathbb{R},\ k(z, z_i) \in \mathbb{R}$$

becomes the complex wavefunction

$$\Psi(z) = \sum_i c_i\, k_{\mathbb{C}}(z, z_i), \qquad c_i \in \mathbb{C},\ k_{\mathbb{C}}(z, z_i) \in \mathbb{C}$$

with $c_i = w_i e^{i \phi_i}$ packing magnitude and phase. The field value at
$z$ (analogue of $Q^+$) is the **Born rule**:

$$V(z) \;=\; |\Psi(z)|^2 \;=\; \Psi^*(z)\,\Psi(z) \;\in\; \mathbb{R}_{\ge 0}$$

Everything downstream — gradients, policy inference, memory update — generalizes
straightforwardly, but with a few subtleties that are worth making explicit.

### What you store, what you compute

| Quantity | Real-valued GRL | Complex-valued GRL |
|---|---|---|
| Particle | $(z_i, w_i)$ | $(z_i, w_i, \phi_i)$ or directly $(z_i, c_i)$ |
| Kernel evaluation | $k(z, z_i) \in \mathbb{R}$ | $k_{\mathbb{C}}(z, z_i) \in \mathbb{C}$ |
| Field | $Q^+(z) \in \mathbb{R}$ | $\Psi(z) \in \mathbb{C}$ |
| Value (for policy) | $Q^+(z)$ directly | $V(z) = \lvert\Psi(z)\rvert^2$ |
| Gradient for policy | $\nabla_a Q^+(s, a)$ | $\nabla_a \lvert\Psi(s, a)\rvert^2$ |

The only structural change is **"what does the policy read off the field?"** In
real GRL you ascend $Q^+$ directly. In complex GRL you ascend the squared
magnitude. This is what enables interference.

---

## 2. Worked Example: Two Particles, Three Phase Choices

Take two particles on the real line at $z_1 = -1, z_2 = +1$, both with magnitude
$w_1 = w_2 = 1$, and a Gaussian kernel with $\ell = 1$:

$$k_{\mathbb{C}}(z, z_i) = e^{-(z - z_i)^2 / 2\ell^2} \cdot e^{i\phi_i}$$

(Here we absorb each particle's phase into its own kernel evaluation for
simplicity; equivalently one can set $\phi_i$ on the complex weight $c_i$.)

**Case A — Aligned phases ($\phi_1 = \phi_2 = 0$).** Reduces to real GRL.
$\Psi(z) = e^{-(z+1)^2/2} + e^{-(z-1)^2/2}$, giving a smooth landscape with a
single bump-like region between the particles. Constructive interference
throughout.

**Case B — Opposite phases ($\phi_1 = 0$, $\phi_2 = \pi$).** Here
$c_2 = -1$. At the midpoint $z = 0$, the two kernel contributions have equal
magnitude and opposite sign, so $\Psi(0) = 0$ and hence $V(0) = 0$. A **node**
forms between them — the value landscape has a valley at the midpoint even
though both particles have positive magnitude.

**Case C — Quadrature ($\phi_1 = 0$, $\phi_2 = \pi/2$).** At the midpoint,
$\Psi(0) = k(0, z_1) + i\, k(0, z_2)$. The magnitudes add in quadrature:
$|\Psi(0)|^2 = k(0, z_1)^2 + k(0, z_2)^2$ — **no interference** at the
midpoint, just magnitude addition.

The same two particles produce three qualitatively different value landscapes
depending on phase. **Phase is a knob that shapes the reinforcement landscape
without changing what events occurred.**

### Pseudocode

```python
import numpy as np

def complex_rbf(z, z_i, ell=1.0):
    """Real Gaussian amplitude; phase carried by the particle weight."""
    return np.exp(-np.sum((z - z_i)**2) / (2 * ell**2))

def psi(z, particles, ell=1.0):
    """Complex wavefunction at z."""
    out = 0.0 + 0.0j
    for p in particles:
        c = p["w"] * np.exp(1j * p["phi"])
        out += c * complex_rbf(z, p["z"], ell)
    return out

def value(z, particles, ell=1.0):
    """Born-rule value |Psi(z)|^2."""
    amp = psi(z, particles, ell)
    return (amp.conjugate() * amp).real  # == abs(amp)**2
```

For visualization, always plot **magnitude** $|\Psi(z)|$ and **phase**
$\arg \Psi(z)$ separately. A contour plot of $V(z) = |\Psi|^2$ alone hides
the interference structure.

---

## 3. Gradients in the Complex Setting

The GRL policy ascends the value landscape:
$a \leftarrow a + \eta \nabla_a V(s, a)$. With $V = |\Psi|^2$, the gradient is

$$\nabla_z V(z) \;=\; \nabla_z \big(\Psi^*(z)\,\Psi(z)\big) \;=\; 2 \, \mathrm{Re}\!\left\{ \Psi^*(z)\, \nabla_z \Psi(z) \right\}$$

This is standard Wirtinger calculus. The **direction** of steepest ascent is
not just "up the kernel," it is modulated by the complex conjugate of the
current amplitude — the gradient rotates as the phase rotates.

### Explicit form for a complex Gaussian kernel

With $k_{\mathbb{C}}(z, z_i) = k_{\mathrm{RBF}}(z, z_i) \cdot e^{i\phi_i(z)}$
and the simplest case of constant per-particle phase $\phi_i$:

$$\nabla_z \Psi(z) \;=\; \sum_i c_i\, \nabla_z k_{\mathrm{RBF}}(z, z_i) \, e^{i\phi_i} \;=\; -\sum_i c_i \, e^{i\phi_i} \frac{z - z_i}{\ell^2}\, k_{\mathrm{RBF}}(z, z_i)$$

Plug into $\nabla_z V = 2\,\mathrm{Re}\{\Psi^* \nabla_z \Psi\}$ and you have a
closed-form policy gradient. **This is the direct analogue of the real-valued
`build_gradient` function in Notebook 2** — same structure, one extra complex
multiplication per term.

```python
def grad_value(z, particles, ell=1.0):
    """
    Gradient of V(z) = |Psi(z)|^2 with respect to z.
    Returns a real-valued gradient vector.
    """
    amp = psi(z, particles, ell)           # complex scalar
    grad_psi = np.zeros_like(z, dtype=complex)
    for p in particles:
        c = p["w"] * np.exp(1j * p["phi"])
        disp = z - p["z"]
        kern = np.exp(-np.dot(disp, disp) / (2 * ell**2))
        grad_psi += c * (-disp / ell**2) * kern
    return 2.0 * np.real(np.conjugate(amp) * grad_psi)
```

If you are working in PyTorch/JAX with complex dtypes, `autograd` handles all
of this — you just need to call `.abs().pow(2)` on the field and let the
framework differentiate. In a reference NumPy implementation, the closed-form
above is preferable because it's numerically stable and obvious to inspect.

---

## 4. Five Patterns for Leveraging Phase

This is the core of the chapter: *when* does the complex extension pay off?
Five concrete patterns, each with a one-line recipe and a caveat.

### Pattern 1 — Temporal credit assignment via rotating phase

**Recipe:** At experience time $t_i$, store particle with phase
$\phi_i = \omega\, t_i$ for some angular frequency $\omega$.

**What you get:** Experiences close in time interfere constructively; those
separated by half a period cancel. The value landscape naturally emphasizes
recent experience without an explicit exponential discount.

**When it helps:** Environments with non-stationary reward structure or
episodic tasks where recency matters but exact decay is unknown.

**Caveat:** The choice of $\omega$ is a hyperparameter. Too small and
everything is constructive (no decoherence); too large and adjacent
experiences cancel spuriously. A useful heuristic: set
$\omega = \pi / T_{\text{relevant}}$ where $T_{\text{relevant}}$ is the
horizon of interest.

### Pattern 2 — Multi-task / multi-context separation via discrete phase

**Recipe:** For context $c \in \{0, 1, \dots, C-1\}$, set
$\phi_i = 2\pi\, c_i / C$. Same-context particles align; different-context
particles separate around the phase circle.

**What you get:** Task-separated value landscapes in a single shared memory.
Queries from context $c$ naturally read out values aligned with that context's
phase; other contexts contribute with scrambled phase and partially cancel.

**When it helps:** Meta-RL, continual learning, or any setting where you
want one particle memory to serve many related tasks without explicit
segregation.

**Caveat:** With $C$ well-spaced phases, cross-context interference averages
toward zero but does not vanish exactly unless you project explicitly. For
hard separation, use a phase-mask projection at query time (see §7).

### Pattern 3 — Directional preference via position-dependent phase

**Recipe:** For navigation-style problems where the relevant structure is a
direction in state-action space, define
$k_{\mathbb{C}}(z, z') = k_{\mathrm{RBF}}(z, z') \exp\!\bigl(i\, \mathbf{n} \cdot (z - z')\bigr)$
for a chosen direction vector $\mathbf{n}$.

**What you get:** Particles "in front of" the agent (aligned with $\mathbf{n}$)
interfere constructively; particles "behind" interfere destructively. The
value landscape develops an implicit flow.

**When it helps:** Flow-field policies, asymmetric tasks where forward
motion is preferred, learning from demonstrations with directional bias.

**Caveat:** This kernel is no longer invariant under translation in the
direction of $\mathbf{n}$. That's usually the point, but it breaks some of
the closed-form nice properties (e.g., the equilibrium distribution of a
random-walk policy).

### Pattern 4 — Learned phase via a small network

**Recipe:** Train a small network $\phi_\psi: \mathcal{Z} \to [0, 2\pi)$ that
predicts phase from (possibly auxiliary) features. The forward model looks
like
$k_{\mathbb{C}}(z, z') = k_{\mathrm{RBF}}(z, z') \exp\!\bigl(i[\phi_\psi(z) - \phi_\psi(z')]\bigr)$.

**What you get:** A data-adaptive phase structure that discovers whatever
interference pattern actually helps on your task. Combines with the
learned-kernel direction from
[`dev/GRL_extensions/learned_kernels/00-scope.md`](../../../dev/GRL_extensions/learned_kernels/00-scope.md).

**When it helps:** When none of Patterns 1–3 obviously applies but the task
seems to have latent temporal or contextual structure.

**Caveat:** Phase is periodic; naive regression will fight the wrap-around.
Parameterize as $(\cos, \sin)$ pair and recover $\phi$ via `atan2`. Also:
anti-collapse regularization is needed just as for the learned-kernel case
(trivially $\phi_\psi \equiv \text{const}$ recovers the real kernel).

### Pattern 5 — Nodes as "forbidden regions"

**Recipe:** To create a hard "do not go here" region at location $z_-$
**without** introducing a negative-weight particle (which just creates a
valley), introduce two particles of equal magnitude and phase difference
$\pi$ straddling $z_-$. The destructive interference zeros out the value
on the line between them.

**What you get:** A **node** — a region where $|\Psi|^2 = 0$ — which is
qualitatively different from a region of low value. Gradients vanish at a
node, and the node is persistent under uniform scaling.

**When it helps:** Modeling hard constraints (obstacles, infeasible
operator parameters) through the field itself rather than external masks.

**Caveat:** Nodes are zero-measure structures. They are visible in
$|\Psi|^2$ but fragile under smoothing or noise. Not appropriate for soft
constraints — use negative weights for those.

---

## 5. Reading Out a Policy

In real-valued GRL, the policy is gradient ascent on $Q^+(s, a)$ over the
action parameter $a$. In complex-valued GRL there are three reasonable
choices, and which one you pick matters.

### Choice 1 — Ascend $|\Psi|^2$ directly (recommended default)

$$a \leftarrow a + \eta\, \nabla_a |\Psi(s, a)|^2 = a + 2\eta\, \mathrm{Re}\!\left\{\Psi^*(s, a)\, \nabla_a \Psi(s, a)\right\}$$

This is the natural generalization and preserves the Born-rule interpretation.
**Default unless you have a specific reason otherwise.**

### Choice 2 — Boltzmann policy on $|\Psi|^2$

$$\pi(a \mid s) \propto \exp\!\bigl(\beta\, |\Psi(s, a)|^2\bigr)$$

Useful for stochastic policies and for the path-integral connection in
Chapter 09. Samples are drawn from the squared-amplitude distribution, which
is exactly the Born rule in RL clothing.

### Choice 3 — Project onto a chosen phase slice

For multi-context scenarios (Pattern 2), compute the **projected field**
$\tilde\Psi_c(z) = e^{-i \phi_c} \Psi(z)$ and ascend $\mathrm{Re}\{\tilde\Psi_c\}$.
This reads out only the value aligned with context $c$, ignoring particles
whose phase is orthogonal to $\phi_c$.

```python
def project_on_phase(z, particles, phi_target, ell=1.0):
    """
    Read out the value component aligned with phase phi_target.
    For context-conditional policies.
    """
    amp = psi(z, particles, ell)
    return np.real(np.exp(-1j * phi_target) * amp)
```

### Which to use when

| Setup | Recommended readout |
|---|---|
| Single task, phase is a structural prior | Choice 1 — $\lvert\Psi\rvert^2$ |
| Multi-task, phase encodes context | Choice 3 — phase-projected |
| Stochastic exploration / path-integral view | Choice 2 — Boltzmann on $\lvert\Psi\rvert^2$ |
| Temporal credit assignment (Pattern 1) | Choice 1; phase decoheres automatically |

---

## 6. Learning the Complex Field

The Part I MemoryUpdate algorithm adds a particle and updates weights to fit
a target value. The complex extension generalizes to **complex particles**
with complex weights, but the training signal is usually still real-valued
(rewards). So you need a loss on $|\Psi|^2$, not on $\Psi$ directly.

### Loss functions

**Option A — Born-rule MSE (no phase supervision).** The phase of each
particle is a free parameter; the loss only sees magnitude.

$$\mathcal{L}(\{c_i\}) = \sum_j \bigl(\, |\Psi(z_j)|^2 - y_j \,\bigr)^2, \qquad y_j \in \mathbb{R}$$

This is a non-convex objective (product $\Psi^* \Psi$ is bilinear in $c_i$),
so initialization matters. Warm-starting phases with one of Patterns 1–3 is a
good strategy.

**Option B — Phase-aware loss (when you have a phase signal).** If the task
provides a phase target (e.g., time-of-experience, context ID), regularize:

$$\mathcal{L} = \sum_j \bigl(\,|\Psi(z_j)|^2 - y_j\bigr)^2 + \lambda \sum_i \bigl(\phi_i - \phi_i^{\text{target}}\bigr)^2_\circ$$

where $(\cdot)^2_\circ$ is the circular-squared-distance (handles wrap-around).

**Option C — Complex-amplitude regression.** If you *can* produce complex
targets (e.g., in simulation where amplitudes are meaningful), fit directly
$\mathcal{L} = \sum_j |\Psi(z_j) - \psi_j^{\text{target}}|^2$. This is convex
in $\{c_i\}$ and has a closed-form solution — the complex analogue of
kernel ridge regression.

### Complex MemoryUpdate (sketch)

```python
def memory_update_complex(memory, z_new, y_new, phi_new=0.0, ell=1.0, ridge=1e-3):
    """
    Add a particle and refit complex weights.

    Args:
        memory: list of particles with keys z, w, phi.
        z_new:  new experience location.
        y_new:  observed real-valued return (or reward).
        phi_new: phase assigned to the new particle (user-chosen per Pattern 1-4).
    """
    memory.append({"z": z_new, "w": 1.0, "phi": phi_new})
    Z = np.array([p["z"] for p in memory])
    C = np.array([p["w"] * np.exp(1j * p["phi"]) for p in memory])
    K = np.array([[complex_rbf(zi, zj, ell) * np.exp(1j * (pi - pj))
                   for pj, zj in zip([p["phi"] for p in memory], Z)]
                  for pi, zi in zip([p["phi"] for p in memory], Z)])
    # Solve for magnitudes that reproduce observed |Psi|^2 at memory points.
    # This is one iteration; in practice you'd run a short local optimizer
    # since the objective is non-convex in the phase block.
    ...
```

(A full implementation belongs in `src/grl/core/` alongside the planned
reference RF-SARSA — deferred until the baseline is in place.)

---

## 7. A Concrete Walk-Through: Context-Switching Gridworld

To make this concrete, consider a **two-context gridworld**:

- Context $A$: goal at $(3, 0)$, reward +1.
- Context $B$: goal at $(-3, 0)$, reward +1.
- Agent observes only position $(x, y)$, not which context is active.
- A single particle memory is shared across both contexts.

**Real-valued GRL.** The memory ends up with positive particles near both
goals. The value landscape has two peaks, and the policy is confused when
in context $A$: it gets pulled toward $(-3, 0)$ from the left half of the
map.

**Complex-valued GRL with context phase (Pattern 2).** Each particle is
stored with $\phi_i = 0$ if observed in context $A$, $\phi_i = \pi$ in
context $B$. A single memory holds both.

- In context $A$, the policy reads the phase-projected field
  $\tilde\Psi_A(z) = \Psi(z)$ (projection onto phase 0). The $\pi$-phase
  particles contribute with a sign flip, creating destructive interference
  near $(-3, 0)$. The peak near $(+3, 0)$ survives.
- In context $B$, we read $\tilde\Psi_B(z) = -\Psi(z)$, which re-flips the
  signs and reveals the $(-3, 0)$ peak while suppressing $(+3, 0)$.

One memory, two disambiguated policies, selected by a phase choice at
inference time. This is the concrete win.

**What had to change from the real-valued version:**
- Each `MemoryUpdate` call tags the new particle with the current context's
  phase.
- Each query provides the context, which selects the readout projection.
- No new learning algorithm, no new capacity — just phase bookkeeping.

This is the kind of leverage complex kernels offer: **structural priors
that compose through phase rather than through separate models.**

---

## 8. Pitfalls

### Phase drift under online updates

If phases are free parameters and you update them by gradient descent,
they can rotate collectively without changing $|\Psi|^2$ — a gauge
symmetry. This manifests as apparent non-convergence. Fix: either pin one
particle's phase to 0 (break the gauge), or project gradients onto the
phase-difference subspace.

### Loss of monotonicity

Real $Q^+$ is linear in particle weights, so adding a positive-weight
particle monotonically increases value everywhere. $|\Psi|^2$ is
**not** monotone in magnitudes — a new particle can *reduce* value through
destructive interference. Don't rely on monotone heuristics (e.g.,
"more particles is always better") in tests.

### Over-interpretation of phase

Phase $\phi_i$ is a mathematical object. It's temporal/contextual/directional
only because you *chose* to make it so. There's no canonical meaning;
debugging benefits hugely from tracking what each particle's phase was
supposed to encode.

### Normalization

Unlike quantum mechanics, GRL does not require $\int |\Psi|^2 = 1$. The
reinforcement field is an **energy/value**, not a probability density. You
can scale $\Psi \to \alpha \Psi$ freely — it scales $V$ by $|\alpha|^2$
but preserves relative ordering. If you *choose* to normalize (e.g., for
a Boltzmann policy), do it explicitly at the readout step, not in the
field definition.

### Computational cost

Complex arithmetic is roughly 4× real arithmetic on CPU, 2× on GPUs with
native complex support. For small particle counts this is irrelevant;
for large memories, factor the complex multiplications into real/imaginary
batched matmuls.

---

## 9. When NOT to Use Complex Kernels

Complex kernels are not free lunch. Reach for them only when:

1. **Your task has a natural phase structure** — time, context, direction,
   or a cycle. If you can't articulate what phase means, don't add it.
2. **You want interference, not just separation.** If Pattern 2 (multi-task)
   is your use case, you could equally well use $C$ independent real-valued
   memories — which is simpler and has the same expressive power as
   hard-projected complex GRL.
3. **You have visualization discipline.** Debugging $|\Psi|^2$ landscapes
   without separately inspecting magnitude and phase is painful.

For most first-attempt GRL problems, start real-valued. Add complexity
(literally) only when the real-valued field has a concrete failure mode
that phase addresses.

---

## 10. Summary and Connections

| Question | Real GRL | Complex GRL |
|---|---|---|
| What's stored per particle? | $(z_i, w_i)$ | $(z_i, w_i, \phi_i)$ |
| What's the field? | $Q^+(z) \in \mathbb{R}$ | $\Psi(z) \in \mathbb{C}$ |
| What does policy read? | $Q^+(s, a)$ | $\lvert\Psi(s, a)\rvert^2$ |
| How do particles combine? | Weighted sum | Complex superposition |
| What interference exists? | Additive only | Constructive / destructive / partial |
| What's the extra knob? | — | Phase $\phi_i$ |

**Cross-references:**
- [Chapter 03](03-complex-rkhs.md) — theoretical foundations and Hermitian
  kernel math.
- [Chapter 04](04-action-and-state-fields.md) — action/state projections
  generalize to complex projections naturally; Pattern 2's phase projection
  is the action-field projection with $e^{i\phi}$ modulation.
- [Chapter 07](07-learning-the-field-beyond-gp.md) — non-GP learning
  mechanisms; §6 of this chapter sketches how they extend to complex
  amplitudes.
- [Chapter 09](09-path-integrals-and-action-principles.md) — path-integral
  formulation and genuine quantum interference; the Boltzmann readout
  (Choice 2 in §5) is the link.
- [`dev/GRL_extensions/learned_kernels/`](../../../dev/GRL_extensions/learned_kernels/)
  — learned feature maps; Pattern 4 (learned phase) is the complex-valued
  analogue.

---

## 11. Next Steps

- [ ] **Companion notebook**: `notebooks/field_series/03_complex_fields.ipynb`
  — visualize the two-particle example from §2 across the three phase cases;
  then implement the context-switching gridworld from §7.
- [ ] **Complex kernel module**: scaffold `src/grl/kernels/complex.py` once
  `src/grl/core/` exists, with the RBF + phase kernel from §1 as the
  reference implementation.
- [ ] **Spectral clustering on complex Gram matrices**: cross-promote this
  tutorial's Pattern 2 to Part II of the tutorial series (Emergent Structure
  & Spectral Abstraction) — complex eigendecomposition naturally produces
  phase-coherent concept subspaces.

---

**Last Updated:** 2026-04-22
