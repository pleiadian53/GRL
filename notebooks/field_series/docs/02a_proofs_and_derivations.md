# Supplementary: Proofs and Derivations for Notebook 02a

Companion to [`../02a_kernels_and_rkhs.ipynb`](../02a_kernels_and_rkhs.ipynb).
This document supplies the proofs and the more formal derivations that the
notebook only states. When the notebook says *"sketch of proof in the
companion doc,"* it means here.

---

## Contents

1. [Symmetry + PSD ⇒ feature map (Moore–Aronszajn, half 1)](#1-symmetry--psd--feature-map)
2. [Feature map ⇒ RKHS (Moore–Aronszajn, half 2)](#2-feature-map--rkhs)
3. [Uniqueness of the RKHS](#3-uniqueness-of-the-rkhs)
4. [The reproducing property — where it comes from](#4-the-reproducing-property)
5. [Mercer's theorem — statement and idea of proof](#5-mercers-theorem)
6. [RKHS ⊂ $C^s$ — smoothness class of functions in $\mathcal{H}_k$](#6-smoothness-class)
7. [Matérn family — derivation sketch](#7-matern-family)
8. [References](#8-references)

---

## 1. Symmetry + PSD ⇒ feature map

**Claim.** If $k: \mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is symmetric and positive semi-definite, there is a Hilbert space $\mathcal{H}$ and a map $\phi: \mathcal{X}\to \mathcal{H}$ such that

$$
k(x, x') = \langle \phi(x), \phi(x')\rangle_{\mathcal{H}}.
$$

**Construction (one of several).** Define the **pre-RKHS** $\mathcal{H}_0$ as the set of finite linear combinations

$$
f = \sum_{i=1}^{n} \alpha_i\, k(x_i, \cdot), \qquad n\in\mathbb{N},\ \alpha_i \in \mathbb{R},\ x_i \in \mathcal{X}.
$$

Define a bilinear form on $\mathcal{H}_0$ by

$$
\left\langle \sum_i \alpha_i k(x_i,\cdot),\ \sum_j \beta_j k(y_j,\cdot) \right\rangle_{\mathcal{H}_0} \;=\; \sum_{i,j} \alpha_i \beta_j\, k(x_i, y_j).
$$

*Well-defined.* The same $f \in \mathcal{H}_0$ can have many representations; one checks that any two representations give the same value for the inner product with any $g$. (This uses PSD — a non-PSD $k$ would give contradictory values.)

*Bilinear, symmetric.* Immediate from the definition and $k$'s symmetry.

*Positive semi-definite.* $\langle f, f\rangle = \sum_{i,j} \alpha_i \alpha_j k(x_i, x_j) \ge 0$ by the PSD property of $k$.

*Positive definite.* If $\langle f, f\rangle = 0$ we need $f \equiv 0$. Using Cauchy–Schwarz, for any $x$:

$$
|f(x)|^2 = \left|\sum_i \alpha_i k(x_i, x)\right|^2 \le \langle f, f\rangle \cdot k(x, x) = 0,
$$

so $f \equiv 0$. (Cauchy–Schwarz holds on any PSD bilinear form.)

Take $\mathcal{H}$ to be the completion of $\mathcal{H}_0$ under the norm induced by this inner product. Take $\phi(x) = k(x, \cdot) \in \mathcal{H}$. Then

$$
\langle \phi(x), \phi(x')\rangle_{\mathcal{H}} = \langle k(x, \cdot), k(x', \cdot)\rangle_{\mathcal{H}_0} = k(x, x').
$$

$\square$

## 2. Feature map ⇒ RKHS

Conversely, if $k(x, x') = \langle \phi(x), \phi(x')\rangle_{\mathcal{H}}$ for some Hilbert space $\mathcal{H}$ and feature map $\phi$, then $k$ is symmetric and PSD:

- *Symmetry:* $\langle \phi(x), \phi(x')\rangle = \langle \phi(x'), \phi(x)\rangle$ in any real Hilbert space.
- *PSD:* $\sum_{i,j} c_i c_j k(x_i, x_j) = \sum_{i,j} c_i c_j \langle \phi(x_i), \phi(x_j)\rangle = \left\| \sum_i c_i \phi(x_i)\right\|^2 \ge 0$.

And the space $\mathcal{H}_0$ from §1 is an RKHS:

- It is a Hilbert space of functions on $\mathcal{X}$ (after completion).
- For every $x$, $k(x, \cdot) \in \mathcal{H}_0$.
- **Reproducing property:** for any $f = \sum_i \alpha_i k(x_i, \cdot) \in \mathcal{H}_0$,
  $$\langle f, k(x, \cdot)\rangle = \sum_i \alpha_i k(x_i, x) = f(x).$$

## 3. Uniqueness of the RKHS

**Claim.** For a given PSD kernel $k$, the RKHS $\mathcal{H}_k$ is unique.

*Proof sketch.* Suppose $\mathcal{H}_1$ and $\mathcal{H}_2$ are two Hilbert spaces of functions on $\mathcal{X}$, both containing all $k(x, \cdot)$ and both having $k$ as their reproducing kernel. Their inner products agree on the pre-RKHS $\mathcal{H}_0 = \mathrm{span}\{k(x,\cdot) : x \in \mathcal{X}\}$ (by the reproducing property and bilinearity). Both are completions of $\mathcal{H}_0$ under the *same* norm, so they coincide as sets, with the same inner product.

## 4. The reproducing property

The notebook's demo verifies $f(x) = \langle f, k(x,\cdot)\rangle$ numerically for $f \in \mathcal{H}_0$. Where does this come from?

Define the **evaluation functional** at $x$: $L_x: f \mapsto f(x)$. On the pre-RKHS, $L_x$ is linear, and a Cauchy–Schwarz bound (as in §1) shows it is *continuous*:

$$
|L_x f|^2 = |f(x)|^2 \le \langle f, f\rangle \cdot k(x, x),
$$

so $\|L_x\| \le \sqrt{k(x, x)}$. By the **Riesz representation theorem**, a continuous linear functional on a Hilbert space is an inner product with some fixed element — so there exists $r_x \in \mathcal{H}_k$ with $L_x f = \langle f, r_x\rangle$. Applying both sides to the specific $f = k(x', \cdot)$:

$$
k(x, x') = L_x k(x', \cdot) = \langle k(x', \cdot), r_x\rangle.
$$

Taking $r_x = k(x, \cdot)$ works. So the kernel *is* the Riesz representer of evaluation — this is why RKHSs are the natural setting for pointwise evaluation, and why point evaluation is continuous in these spaces (it is not, in general, in e.g. $L^2$).

Connection: notebook 02a frames (iii) as "the kernel is a basis of evaluation representers" — this is the precise statement of that framing.

## 5. Mercer's theorem

**Setup.** Let $\mathcal{X}$ be compact, $\mu$ a finite measure on $\mathcal{X}$, $k: \mathcal{X}\times\mathcal{X}\to \mathbb{R}$ continuous, symmetric, PSD. Define the integral operator

$$
(T_k f)(x) = \int_{\mathcal{X}} k(x, x') f(x')\, d\mu(x').
$$

**Theorem.** $T_k: L^2(\mu) \to L^2(\mu)$ is compact, self-adjoint, and positive. Its spectrum is a sequence $\lambda_1 \ge \lambda_2 \ge \cdots \ge 0$ with $\lambda_j \to 0$, and there exist orthonormal eigenfunctions $\psi_j \in L^2(\mu)$ with $T_k \psi_j = \lambda_j \psi_j$. Moreover, the kernel expands uniformly and absolutely on $\mathcal{X}\times\mathcal{X}$ as

$$
k(x, x') = \sum_{j=1}^{\infty} \lambda_j\, \psi_j(x)\, \psi_j(x').
$$

**Idea of proof.** Compactness of $T_k$ (via continuity of $k$ on compact $\mathcal{X}$) + self-adjointness ⇒ spectral theorem for compact self-adjoint operators yields the $(\lambda_j, \psi_j)$. Positivity of $T_k$ (inherited from PSD of $k$) gives $\lambda_j \ge 0$. The uniform convergence of the kernel series is **Mercer's harder content** and uses Dini's theorem plus the monotonicity of partial sums.

**Finite-sample version.** For a sample $\{x_1,\dots,x_n\}$, the $n\times n$ Gram matrix $K$ is the discretized version of $T_k$ and its eigendecomposition

$$
K = \sum_{j=1}^{n} \lambda_j v_j v_j^\top
$$

is a finite-dimensional Mercer expansion. This is what the notebook's Demo 3.1 computes.

**Feature map via Mercer.** Setting

$$
\phi(x) = \big(\sqrt{\lambda_1}\,\psi_1(x),\ \sqrt{\lambda_2}\,\psi_2(x),\ \dots\big) \in \ell^2,
$$

one has $\langle\phi(x), \phi(x')\rangle = \sum_j \lambda_j \psi_j(x)\psi_j(x') = k(x, x')$. So Mercer gives a *concrete* feature map for any kernel on a compact domain.

## 6. Smoothness class

A precise statement linking kernel smoothness to RKHS functions:

**Proposition.** If $k(x, x')$ is $C^{2s}$ jointly (i.e., $\partial^\alpha \partial^{\alpha'} k$ exists and is continuous for $|\alpha|, |\alpha'| \le s$), then every $f \in \mathcal{H}_k$ is $C^s$, and differentiation commutes with the RKHS inner product:

$$
(\partial^{\alpha} f)(x) = \langle f, \partial_{1}^{\alpha} k(x, \cdot)\rangle_{\mathcal{H}_k}\qquad \text{for } |\alpha| \le s.
$$

**Idea.** The reproducing property $f(x) = \langle f, k(x,\cdot)\rangle$ can be differentiated under the inner product when $k$ is smooth enough; the derivative of the Riesz representer $k(x, \cdot)$ remains in $\mathcal{H}_k$ and represents the evaluation of $\partial^\alpha f$.

**Consequences for kernel choice.**

- RBF: $C^\infty$ in both arguments ⇒ $\mathcal{H}_{\text{RBF}}$ contains only $C^\infty$ functions.
- Matérn-$\nu$: $C^{\lceil \nu \rceil - 1}$ smooth ⇒ $\mathcal{H}_{\text{Matern-}\nu}$ contains functions of exactly that regularity.
- A $Q^+$ expected to have sharp transitions cannot be faithfully represented in $\mathcal{H}_{\text{RBF}}$; Matérn-$\tfrac{1}{2}$ allows $C^0$-only functions.

## 7. Matérn family

The Matérn kernel of order $\nu > 0$ on $\mathbb{R}^d$ with lengthscale $\ell$:

$$
k_\nu(r) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\,r}{\ell}\right)^{\nu} K_\nu\!\left(\frac{\sqrt{2\nu}\,r}{\ell}\right),
\qquad r = \|x - x'\|,
$$

where $K_\nu$ is the modified Bessel function of the second kind.

For half-integer $\nu = p + \tfrac{1}{2}$, $p \in \mathbb{N}_0$, the Bessel function collapses to an elementary product of a polynomial and an exponential:

- $\nu = 1/2$: $k(r) = \exp(-r/\ell)$  (Ornstein–Uhlenbeck / exponential kernel)
- $\nu = 3/2$: $k(r) = (1 + \sqrt{3}\,r/\ell)\exp(-\sqrt{3}\,r/\ell)$
- $\nu = 5/2$: $k(r) = (1 + \sqrt{5}\,r/\ell + \tfrac{5r^2}{3\ell^2})\exp(-\sqrt{5}\,r/\ell)$

**Sample path regularity.** A Gaussian process $f \sim \mathcal{GP}(0, k_\nu)$ has sample paths that are $\lfloor \nu \rfloor$-times mean-square differentiable and $(\lfloor \nu \rfloor - \varepsilon)$-times Hölder-continuous almost surely.

**Limit.** As $\nu \to \infty$, $k_\nu \to \exp(-r^2/(2\ell^2))$ (the RBF), and sample paths become $C^\infty$.

**Spectral density** (Bochner's theorem, $\mathbb{R}^d$):

$$
S_\nu(\omega) \propto \left(\frac{2\nu}{\ell^2} + \|\omega\|^2\right)^{-(\nu + d/2)}.
$$

The polynomial decay rate of $S_\nu$ at high frequency matches the smoothness of the sample paths — faster decay ↔ smoother paths ↔ larger $\nu$.

## 8. References

- Aronszajn, N. (1950). *Theory of Reproducing Kernels.* Transactions of the AMS, 68(3), 337–404.
- Schölkopf, B., & Smola, A. (2002). *Learning with Kernels.* MIT Press. — Chapter 2 for Moore–Aronszajn, Chapter 3 for Mercer.
- Berlinet, A., & Thomas-Agnan, C. (2004). *Reproducing Kernel Hilbert Spaces in Probability and Statistics.* Kluwer.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. — §4 for the Matérn family and spectral view.
- Steinwart, I., & Christmann, A. (2008). *Support Vector Machines.* Springer. — rigorous functional-analytic treatment, especially §4 (RKHS as function spaces) and §4.6 (smoothness).

---

**Status.** 🔬 Validated — derivations follow standard references; the finite-sample Mercer demo in the notebook is numerically tight. Proofs are sketches; substitute a reference for a full treatment.
