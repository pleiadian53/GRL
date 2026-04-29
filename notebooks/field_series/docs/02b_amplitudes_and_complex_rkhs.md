# Supplementary: Complex RKHS and Amplitude Overlaps

Companion to [`../02b_kernels_and_amplitudes.ipynb`](../02b_kernels_and_amplitudes.ipynb).
This document fills in the formal side of the quantum-inspired treatment: the
complex RKHS axioms, Schur's product theorem, Bochner's theorem in the complex
case, and the reconciliation with the GRL quantum-inspired chapters in
[`docs/GRL0/quantum_inspired/`](../../../docs/GRL0/quantum_inspired/).

---

## Contents

1. [Complex Hilbert space — conventions](#1-complex-hilbert-space)
2. [Hermitian PSD kernels](#2-hermitian-psd-kernels)
3. [Complex Moore–Aronszajn](#3-complex-moorearonszajn)
4. [Schur product theorem (our damped-oscillatory kernel is PSD)](#4-schur-product-theorem)
5. [Bochner's theorem and stationary complex kernels](#5-bochners-theorem)
6. [Born-like readout and its ambiguities](#6-born-like-readout)
7. [Connection to the quantum-inspired chapters](#7-connection-to-quantum-inspired-chapters)
8. [References](#8-references)

---

## 1. Complex Hilbert space

Convention (matches the physics-style bra-ket treatment in the quantum-inspired chapters):

- Inner product is **conjugate-linear in the first argument**, linear in the second:
  $$\langle \alpha u, v\rangle = \bar\alpha \langle u, v\rangle, \qquad \langle u, \alpha v\rangle = \alpha \langle u, v\rangle.$$
- This matches $\langle u, v\rangle = u^\dagger v$ in matrix form (row-conjugate times column).

With this convention, $\langle u, u\rangle = \|u\|^2 \ge 0$ and the Cauchy–Schwarz inequality still reads $|\langle u, v\rangle|^2 \le \|u\|^2 \|v\|^2$.

## 2. Hermitian PSD kernels

**Definition.** $k: \mathcal{X}\times\mathcal{X}\to \mathbb{C}$ is a **complex kernel** if:

1. *Hermitian symmetry:* $k(x, x') = \overline{k(x', x)}$.
2. *Positive semi-definite over $\mathbb{C}$:* for every finite $\{x_i\}$ and every $\mathbf{c} \in \mathbb{C}^n$,
   $$\mathbf{c}^\dagger K \mathbf{c} = \sum_{i, j} \bar c_i\, c_j\, k(x_i, x_j) \ge 0.$$

The Gram matrix $K \in \mathbb{C}^{n\times n}$ is Hermitian PSD. Its eigenvalues are real and non-negative.

**Consequences.**

- $k(x, x) = \overline{k(x, x)} \in \mathbb{R}_{\ge 0}$. (Take $\mathbf{c} = \mathbf{e}_i$ for $k(x_i, x_i) \ge 0$.)
- $|k(x, x')|^2 \le k(x, x)\, k(x', x')$. (Cauchy–Schwarz on the 2×2 Gram matrix with entries $k(x_i, x_j)$ for $i, j \in \{1, 2\}$ being PSD.)

## 3. Complex Moore–Aronszajn

**Claim.** $k: \mathcal{X}\times\mathcal{X}\to\mathbb{C}$ is Hermitian PSD if and only if there exists a complex Hilbert space $\mathcal{H}$ and a map $\phi: \mathcal{X}\to\mathcal{H}$ such that

$$
k(x, x') = \langle \phi(x), \phi(x')\rangle_{\mathcal{H}} = \phi(x)^\dagger \phi(x').
$$

**Proof sketch (⇒).** Exactly parallels the real case (see 02a supplementary §1). Build $\mathcal{H}_0$ as finite $\mathbb{C}$-linear combinations $f = \sum_i \alpha_i k(x_i, \cdot)$ with the sesquilinear form

$$
\left\langle \sum_i \alpha_i k(x_i, \cdot),\ \sum_j \beta_j k(y_j, \cdot) \right\rangle_{\mathcal{H}_0} = \sum_{i, j} \bar\alpha_i \beta_j\, k(x_i, y_j).
$$

- *Sesquilinear + Hermitian* from the Hermitian symmetry of $k$.
- *PSD:* $\langle f, f\rangle = \sum \bar\alpha_i \alpha_j k(x_i, x_j) \ge 0$.
- *Non-degenerate:* Cauchy–Schwarz gives $|f(x)|^2 \le \langle f, f\rangle k(x, x)$, so $\langle f, f\rangle = 0 \Rightarrow f \equiv 0$.

Complete under the induced norm and take $\phi(x) = k(x, \cdot)$. Then $\langle \phi(x), \phi(x')\rangle_{\mathcal{H}} = \overline{k(x, x)}^{\text{(of arg 1)}}\cdot k(x', \cdot)|_x$ — which, after carefully unwinding the conjugate-linearity, equals $k(x, x')$.

**Proof sketch (⇐).** If $k = \langle\phi(\cdot), \phi(\cdot)\rangle$, then:

- *Hermitian:* $k(x', x) = \langle\phi(x'), \phi(x)\rangle = \overline{\langle \phi(x), \phi(x')\rangle} = \overline{k(x, x')}$.
- *PSD:* $\sum \bar c_i c_j k(x_i, x_j) = \left\|\sum_j c_j \phi(x_j)\right\|^2 \ge 0$.

## 4. Schur product theorem

Used in the notebook to argue that $k_{\text{osc}} = (\text{RBF}) \cdot (e^{i\omega\Delta})$ is PSD.

**Theorem (Schur / Hadamard).** If $A, B \in \mathbb{C}^{n\times n}$ are both Hermitian PSD, then their **Hadamard (elementwise) product** $A \circ B$ is Hermitian PSD.

**Proof sketch.** Write $A = \sum_i \lambda_i u_i u_i^\dagger$ and $B = \sum_j \mu_j v_j v_j^\dagger$ (spectral decompositions; $\lambda_i, \mu_j \ge 0$). Then

$$
A \circ B = \sum_{i, j} \lambda_i \mu_j\, (u_i u_i^\dagger) \circ (v_j v_j^\dagger) = \sum_{i, j} \lambda_i \mu_j\, (u_i \circ v_j)(u_i \circ v_j)^\dagger,
$$

a non-negative combination of rank-1 PSD matrices, hence PSD. $\square$

**Corollary (kernels).** If $k_1, k_2$ are PSD kernels on $\mathcal{X}$, then $k_1 \cdot k_2$ is a PSD kernel.

**Applied to the notebook.** Both factors below are PSD kernels:

- $k_1(x, x') = \exp(-(x-x')^2/(2\ell^2))$ — RBF (real PSD; also Hermitian since real-valued).
- $k_2(x, x') = \exp(i\omega(x-x'))$ — the "plane-wave" kernel; it is rank-1 (feature map $\phi(x) = e^{i\omega x}$, a scalar), so its Gram matrix is $\phi \phi^\dagger$, automatically Hermitian PSD.

Their product $k_{\text{osc}} = k_1 \cdot k_2$ is therefore Hermitian PSD.

## 5. Bochner's theorem

A classical result that characterizes all **stationary** complex-valued kernels.

**Theorem (Bochner).** A continuous function $k: \mathbb{R}^d \to \mathbb{C}$ is the Fourier transform of a finite, non-negative Borel measure $\mu$ on $\mathbb{R}^d$ if and only if the function $(x, x') \mapsto k(x - x')$ is a Hermitian PSD kernel.

$$
k(\tau) = \int_{\mathbb{R}^d} e^{i\,\omega^\top \tau}\, d\mu(\omega), \qquad \mu \ge 0.
$$

**Reading for GRL.** Stationary kernels are exactly the kernels whose PSD-ness is equivalent to a non-negative spectral density. This is what makes the **plane-wave kernel** $k(x, x') = e^{i\omega(x - x')}$ immediately valid: it is the transform of a point mass at a single frequency $\omega$.

**Random Fourier features.** Bochner also gives a Monte Carlo approximation: draw $\omega_k \sim \mu / \|\mu\|$ and form

$$
\hat k(x, x') = \frac{\|\mu\|}{M} \sum_{k=1}^{M} e^{i\,\omega_k^\top(x - x')}.
$$

This is how large-scale kernel methods approximate RBF / Matérn kernels in practice; the complex version is the natural home of this construction.

## 6. Born-like readout

The notebook reads probabilities from $|Q^+(z)|^2$ after normalization. Some careful points that notebook 02b deliberately glosses over:

**(a) $Q^+$ is not a wavefunction in the QM sense.** It is a vector in an RKHS of functions on $\mathcal{X}$, not a vector in $L^2(\mathcal{X})$. For $|Q^+|^2$ to be integrable we need extra conditions (e.g. $k$ bounded and $\mathcal{X}$ compact, or $k$ in $L^2$ — the RBF case).

**(b) The readout is phase-forgetful, but superposition is not.** Adding amplitudes $Q^+ = \sum_i w_i\, k(z_i, \cdot)$ preserves phase; $|Q^+|^2$ collapses it. This matches the QM measurement pattern.

**(c) Normalization is not unique.** Different normalization schemes — $L^2$ normalization ($\int |Q^+|^2 = 1$) vs. RKHS normalization ($\|Q^+\|_{\mathcal{H}_k} = 1$) — give different probability densities. The RKHS norm is often the "natural" one; $L^2$ normalization is what looks most like QM. Choose explicitly and state which you mean.

**(d) Classical limit.** If all weights are real and phases align ($e^{i\theta_i} = \pm 1$), the complex machinery reduces to ordinary signed RKHS. The interference story is exactly the "extra" content gained by going complex.

## 7. Connection to quantum-inspired chapters

The GRL project already has an extended treatment under
[`docs/GRL0/quantum_inspired/`](../../../docs/GRL0/quantum_inspired/). Map from
notebook 02b to those chapters:

| Notebook section | Chapter | Topic |
|---|---|---|
| Part 2 (definition) | [03-complex-rkhs](../../../docs/GRL0/quantum_inspired/03-complex-rkhs.md) | Complex RKHS axioms |
| Part 3 (interference) | [02-rkhs-basis-and-amplitudes](../../../docs/GRL0/quantum_inspired/02-rkhs-basis-and-amplitudes.md) | Amplitudes, particle basis, interference |
| Part 4 (phase as context) | [04-action-and-state-fields](../../../docs/GRL0/quantum_inspired/04-action-and-state-fields.md), [05-concept-projections](../../../docs/GRL0/quantum_inspired/05-concept-projections-and-measurements.md) | Field projections, context-sensitive measurement |
| Part 5 (Born-like readout) | [01a-wavefunction-interpretation](../../../docs/GRL0/quantum_inspired/01a-wavefunction-interpretation.md) | Wavefunction vs. state vector — what $Q^+$ is and is not |
| Part 6 (when to use) | [07-learning-the-field-beyond-gp](../../../docs/GRL0/quantum_inspired/07-learning-the-field-beyond-gp.md) | Where the quantum-inspired machinery actually helps GRL |

The notebook is the **hands-on demo** for the machinery those chapters develop
formally. Use the notebook to build intuition, then return to the chapters for
the rigorous claims.

## 8. References

- Aronszajn, N. (1950). *Theory of Reproducing Kernels.* — Section 3 covers the complex case.
- Paulsen, V., & Raghupathi, M. (2016). *An Introduction to the Theory of Reproducing Kernel Hilbert Spaces.* Cambridge. — Chapter 2 states Moore–Aronszajn in both real and complex forms.
- Bochner, S. (1932/1955). *Vorlesungen über Fouriersche Integrale* / *Harmonic Analysis and the Theory of Probability.*
- Rudin, W. (1990). *Fourier Analysis on Groups.* — classical reference for Bochner.
- Rahimi, A., & Recht, B. (2007). *Random features for large-scale kernel machines.* NIPS. — complex random Fourier features in practice.
- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers.* Springer. — readable bridge between kernel methods and complex Hilbert-space learning.

---

**Status.** 📝 Draft — the claims follow standard references; Schur and Bochner are stated and used but not proved in full. The reconciliation with the GRL quantum-inspired chapters (§7) is provisional; refine as those chapters stabilize.
