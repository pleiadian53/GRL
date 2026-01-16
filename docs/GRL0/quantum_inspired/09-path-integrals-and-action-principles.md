# Chapter 09: Path Integrals and Action Principles

**Purpose**: This chapter makes explicit the deep connection between GRL's energy-based formulation and quantum mechanics via **Feynman's path integral formulation**. While Chapter 03a introduced the least action principle from a classical control perspective, here we explore the quantum mechanical version and its implications for amplitude-based learning.

**Key insight**: Just as quantum mechanics can be formulated via path integrals over complex-valued amplitudes, GRL can be extended to a path integral formulation over **complex-valued reinforcement fields**, enabling interference effects and richer policy representations.

---

## 1. Feynman's Path Integral Formulation

### 1.1 From Schrödinger to Path Integrals

In standard quantum mechanics, the state evolves via the **Schrödinger equation**:

$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

Feynman showed this is equivalent to **summing over all possible paths**:

$$\langle x_f | e^{-i\hat{H}T/\hbar} | x_i \rangle = \int \mathcal{D}[x(t)] \, e^{iS[x(t)]/\hbar}$$

where:

- $S[x(t)] = \int_0^T L(x, \dot{x}, t) dt$ = **action functional**

- $L = T - V$ = Lagrangian (kinetic - potential energy)
- $\hbar$ = Planck's constant (sets the scale)

**Interpretation**: 
The probability amplitude to go from $x_i$ to $x_f$ is the **sum over all paths**, each weighted by $e^{iS/\hbar}$.

---

### 1.2 Key Differences from Classical Action

**Classical mechanics**: One path minimizes action
- $\delta S = 0$ → Euler-Lagrange equations
- Deterministic trajectory

**Quantum mechanics**: All paths contribute
- $\text{Amplitude} = \sum_{\text{paths}} e^{iS/\hbar}$
- **Interference** between paths (constructive/destructive)
- Paths near the classical trajectory dominate (stationary phase)

**Why complex amplitudes?**

- Phase $e^{iS/\hbar}$ encodes action along path
- Allows **interference**: paths can cancel or reinforce
- Recovers classical limit as $\hbar \to 0$ (only minimal action path survives)

---

### 1.3 The Imaginary Time (Wick Rotation)

**Key trick**: Replace $t \to -i\tau$ (imaginary time):

$$e^{-i\hat{H}t/\hbar} \to e^{-\hat{H}\tau/\hbar}$$

**Path integral becomes**:

$$\langle x_f | e^{-\hat{H}\tau/\hbar} | x_i \rangle = \int \mathcal{D}[x(\tau)] \, e^{-S_E[x(\tau)]/\hbar}$$

where $S_E$ is the **Euclidean action** (with sign flip).

**Why this matters for RL**:

- Real exponentials $e^{-S_E/\hbar}$ (no oscillations!)
- Becomes a **probability distribution** (after normalization)
- This is the statistical mechanics / thermodynamics connection
- **Temperature** $\beta = 1/(k_B T)$ plays the role of $1/\hbar$

---

## 2. Stochastic Control as Imaginary Time QM

### 2.1 The Feynman-Kac Formula

The **Feynman-Kac formula** connects:

- Quantum mechanics (path integrals)
- Stochastic processes (Brownian motion)
- Parabolic PDEs (diffusion equations)

**For RL**: The optimal value function satisfies:

$$V(s) = -\nu \log \mathbb{E}_{\text{paths}}\left[e^{-\frac{1}{\nu}\int_0^\infty C(s_t, u_t) dt}\right]$$

This is a path integral over trajectories with cost functional $C$!

**Explicitly**:

$$e^{-V(s)/\nu} = \int \mathcal{D}[\tau] \, e^{-\frac{1}{\nu}\int_0^\infty C(s_t, u_t) dt}$$

**Comparison to QM**:

| Quantum Mechanics | Stochastic Control |
|---|---|
| $\hbar$ (Planck's constant) | $\nu$ (temperature) |
| $\psi(x)$ (wavefunction) | $e^{-V(s)/\nu}$ (value function) |
| $\hat{H}$ (Hamiltonian) | $\hat{H}_{\text{HJB}}$ (Hamilton-Jacobi-Bellman) |
| Schrödinger equation | Diffusion equation |

---

### 2.2 The HJB Equation as Schrödinger Equation

The **Hamilton-Jacobi-Bellman (HJB)** equation for optimal control:

$$-\frac{\partial V}{\partial t} = \min_u \left[C(s, u) + \nabla V \cdot f(s, u) + \frac{\nu}{2}\nabla^2 V\right]$$

**In imaginary time QM**, the Schrödinger equation becomes:

$$-\frac{\partial \psi}{\partial \tau} = -\frac{\hbar}{2m}\nabla^2 \psi + V(x)\psi$$

**With** $\psi = e^{-V/\nu}$, these are **isomorphic**!

| HJB | Schrödinger (imaginary time) |
|---|---|
| $V(s, t)$ | $-\nu \log \psi(x, \tau)$ |
| $C(s, u)$ | Potential $V(x)$ |
| $\nu$ | $\hbar$ |
| Diffusion term | Kinetic energy $-\frac{\hbar}{2m}\nabla^2$ |

**This is not just an analogy—it's a mathematical equivalence!**

---

## 3. GRL's Path Integral Formulation

### 3.1 The GRL Action Functional (Revisited)

In augmented space $z = (s, \theta)$, the action along trajectory $\tau = \{z_t\}$ is:

$$S_{\text{GRL}}[\tau] = \int_0^T \left[E(z_t) + \frac{1}{2\lambda}\|\dot{z}_t\|^2\right] dt$$

where:

- $E(z) = -Q^+(z)$ = energy
- $\lambda$ = temperature (exploration parameter)

**Path integral formulation**:

$$\mathcal{Z}(z_i, z_f; T) = \int \mathcal{D}[\tau] \, e^{-S_{\text{GRL}}[\tau]/\lambda}$$

This is the **partition function** summing over all trajectories from $z_i$ to $z_f$ in time $T$.

---

### 3.2 The Boltzmann Policy as Path Integral

**Single-step decision** (instantaneous action selection):

$$\pi(\theta|s) = \frac{e^{-E(s,\theta)/\lambda}}{\int e^{-E(s,\theta')/\lambda} d\theta'} = \frac{e^{Q^+(s,\theta)/\lambda}}{\int e^{Q^+(s,\theta')/\lambda} d\theta'}$$

**Multi-step trajectory**:

$$\pi[\tau | s_0] \propto e^{-S_{\text{GRL}}[\tau]/\lambda}$$

This says: **The probability of a trajectory is proportional to $\exp(-\text{action}/\lambda)$**.

**In QM language**: This is the **Boltzmann weight** of the trajectory in imaginary time.

---

### 3.3 Soft State Transitions Revisited

In Chapter 01a (Wavefunction Interpretation), we noted that GRL induces **soft state transitions** due to kernel overlap.

**Path integral perspective**: 

- Each trajectory contributes to the transition amplitude
- Kernel $k(z, z')$ encodes **transition amplitude** between configurations
- Overlap creates **interference-like effects** (constructive/destructive reinforcement)

**Formally**, the transition kernel is:

$$P(s_{t+1}, \theta_{t+1} | s_t, \theta_t) = \frac{1}{\mathcal{Z}}\int \mathcal{D}[\text{paths}] \, e^{-S[\text{path}]/\lambda}$$

where paths connect $(s_t, \theta_t)$ to $(s_{t+1}, \theta_{t+1})$.

**This is a path integral!** The "softness" comes from summing over all possible intermediate trajectories.

---

## 4. Complex-Valued GRL: Enabling True Interference

### 4.1 Motivation: Real vs. Complex Amplitudes

**Current GRL** (Chapters 01-08):

- Real-valued $Q^+(z)$
- Boltzmann weights $e^{Q^+/\lambda}$ are always positive
- No true interference (only additive contributions)

**Complex-valued extension**:

- $Q^+(z) \in \mathbb{C}$ (complex reinforcement field)
- Amplitudes $\psi(z) = e^{i\phi(z)}$ where $\phi(z) = Q^+(z)/\hbar_{\text{eff}}$
- **Phase** $\phi(z)$ enables interference

**Why complex?**

- **Interference**: Paths can destructively interfere (cancel out)
- **Richer representations**: Phase encodes additional structure
- **Quantum-inspired exploration**: "Tunneling" through barriers

---

### 4.2 Complex RKHS (Chapter 03 Revisited)

From **Chapter 03: Complex-Valued RKHS**, we can extend GRL to complex functions:

$$Q^+: \mathcal{Z} \to \mathbb{C}$$

with complex kernel $k: \mathcal{Z} \times \mathcal{Z} \to \mathbb{C}$ (sesquilinear).

**Path integral with complex action**:

$$\mathcal{Z}(z_i, z_f) = \int \mathcal{D}[\tau] \, e^{iS_{\text{complex}}[\tau]/\hbar_{\text{eff}}}$$

where $S_{\text{complex}}[\tau] = \int [Q^+(z_t) + \frac{i}{2\hbar_{\text{eff}}}\|\dot{z}_t\|^2] dt$.

**Probability** (Born rule):

$$P(\tau) = \frac{|\mathcal{Z}_\tau|^2}{\sum_{\tau'} |\mathcal{Z}_{\tau'}|^2}$$

**This allows paths to interfere!**

---

### 4.3 Example: Double-Slit in Action Space

**Setup**: Two discrete actions $\theta_A$ and $\theta_B$ both lead to the same next state $s'$.

**Classical GRL** (real $Q^+$):

$$P(s') = P(s' | \theta_A)P(\theta_A) + P(s' | \theta_B)P(\theta_B)$$

(Incoherent sum, no interference)

**Complex GRL**:

$$\text{Amplitude}(s') = \psi(s' | \theta_A) + \psi(s' | \theta_B)$$

$$P(s') = |\text{Amplitude}(s')|^2 = |\psi(s'|\theta_A) + \psi(s'|\theta_B)|^2$$

**Expanding**:

$$P(s') = |\psi(s'|\theta_A)|^2 + |\psi(s'|\theta_B)|^2 + 2\text{Re}[\psi^*(s'|\theta_A)\psi(s'|\theta_B)]$$

The last term is the **interference term**! It can be positive (constructive) or negative (destructive).

**Interpretation**:

- If phases align: $\psi(s'|\theta_A)$ and $\psi(s'|\theta_B)$ reinforce → higher $P(s')$
- If phases oppose: They cancel → lower $P(s')$
- This allows the agent to **suppress undesirable state transitions**

---

## 5. Practical Path Integral Algorithms

### 5.1 Path Integral Policy Improvement (PI²)

**Algorithm** (Theodorou et al., 2010):

1. **Rollout**: Sample $K$ noisy trajectories around current policy
   $$\tau_k \sim \pi(\cdot | s_t) + \epsilon_k, \quad k = 1, \ldots, K$$

2. **Compute costs**: $S_k = \int_0^T C(s_t^{(k)}, u_t^{(k)}) dt$

3. **Weight trajectories**: $w_k = \frac{e^{-S_k/\lambda}}{\sum_{j=1}^K e^{-S_j/\lambda}}$

4. **Update policy**: $\pi_{\text{new}}(u|s) = \sum_{k=1}^K w_k \, \delta(u - u_k)$

**This is empirical path integration!**

---

### 5.2 GRL Adaptation: Particle-Weighted Path Integration

**For GRL**:

1. **Particle memory**: $\{(z_i, w_i)\}$ defines $Q^+(z)$

2. **Trajectory rollout**: From state $s_t$, sample $K$ action sequences:
   $$\{\theta_t^{(k)}, \theta_{t+1}^{(k)}, \ldots\}_{k=1}^K$$

3. **Compute cumulative return**: 
   $$R_k = \sum_{\tau=t}^T r_\tau^{(k)}$$

4. **Path action**:
   $$S_k = -R_k + \frac{1}{2\lambda}\sum_{\tau} \|\theta_{\tau+1}^{(k)} - \theta_\tau^{(k)}\|^2$$

5. **Weight paths**: $w_k \propto e^{-S_k/\lambda}$

6. **Update particles**: Add/modify particles along high-weight paths

**Key difference from PI²**: GRL uses particle memory to encode $Q^+$, not parametric policy.

---

### 5.3 Langevin Sampling on the Field

**For continuous action spaces**, sample actions via Langevin dynamics:

$$\theta_{k+1} = \theta_k + \epsilon \nabla_\theta Q^+(s, \theta_k) + \sqrt{2\epsilon\lambda} \, \xi_k$$

where $\xi_k \sim \mathcal{N}(0, I)$.

**This is gradient flow on the reinforcement field!**

**Path integral view**: 

- Each Langevin trajectory is a sample from the path integral
- Multiple samples approximate $\mathcal{Z}(z_i, z_f)$
- Action selection = finding high-probability paths

---

## 6. Connection to Quantum Measurement (Chapter 05)

### 6.1 Concept Subspaces as Measurement Operators

From **Chapter 05: Concept Projections and Measurements**, concepts are subspaces $\mathcal{H}_c \subset \mathcal{H}$ with projection operator $\hat{P}_c$.

**Path integral interpretation**:

- Measuring concept $c$ = projecting trajectory onto $\mathcal{H}_c$
- Post-measurement state: $|\psi'\rangle = \hat{P}_c |\psi\rangle / \|\hat{P}_c |\psi\rangle\|$
- Measurement collapses the path integral to concept-compatible paths

**Formally**, after measuring $c$:

$$\mathcal{Z}_c(z_i, z_f) = \int_{\tau \in \mathcal{H}_c} \mathcal{D}[\tau] \, e^{-S[\tau]/\lambda}$$

This is a **conditional path integral** over concept-consistent trajectories.

---

### 6.2 Hierarchical Composition via Path Integrals

**Nested concepts** $\mathcal{H}_{c_1} \subset \mathcal{H}_{c_2}$ define a hierarchy.

**Path integral view**:

- Low-level concept $c_1$: Paths stay in local subspace
- High-level concept $c_2$: Paths stay in larger subspace
- Transition $c_1 \to c_1'$: Path integral between subspaces

**This is exactly how quantum mechanics handles multi-scale dynamics!**

Example: Molecular dynamics
- Electronic states (fast, local paths)
- Nuclear motion (slow, global paths)
- Born-Oppenheimer approximation: Separate path integrals

**In GRL**: 

- Low-level: Action parameter trajectories
- High-level: Concept activation trajectories
- Spectral methods: Identify slow vs. fast modes

---

## 7. Advanced Topics: Feynman Diagrams for GRL?

### 7.1 Perturbation Theory

In QM, **Feynman diagrams** represent terms in the perturbative expansion of path integrals.

**For GRL**, could we develop a diagrammatic expansion?

**Idea**:

- Vertices: MemoryUpdate operations (particle creation/modification)
- Edges: Kernel interactions (particle propagation)
- Loops: Circular dependencies in belief update

**This is speculative**, but could formalize:

- How perturbations propagate through memory
- Which particles are "entangled" (strongly coupled)
- Computational complexity of belief updates

---

### 7.2 Instantons and Rare Events

In QM, **instantons** are classical solutions to the Euclidean equations of motion that dominate tunneling.

**For GRL**:

- Instanton = rare, high-cost trajectory that dramatically changes $Q^+$
- Example: Discovering a shortcut in navigation (low probability, high reward)
- **Instanton calculus**: Approximate path integral by saddle points

**Practical use**:

- Identify critical experiences (instantons) for learning
- Prioritize replay of high-impact trajectories
- Understand exploration-exploitation via tunneling rates

---

## 8. Implementation Sketch

### 8.1 Complex-Valued Particle Weights

**Extend particles** to complex weights:

```python
class ComplexParticle:
    def __init__(self, z, w_real, w_imag):
        self.z = z  # Augmented state (s, θ)
        self.w = complex(w_real, w_imag)  # Complex weight
    
    def phase(self):
        return np.angle(self.w)
    
    def magnitude(self):
        return np.abs(self.w)
```

**Complex kernel**:

```python
def complex_kernel(z1, z2, length_scale=1.0, phase_scale=1.0):
    """Complex-valued RBF kernel with phase."""
    dist = np.linalg.norm(z1 - z2)
    magnitude = np.exp(-dist**2 / (2 * length_scale**2))
    phase = phase_scale * np.dot(z1 - z2, np.ones_like(z1))  # Example phase
    return magnitude * np.exp(1j * phase)
```

---

### 8.2 Path Integral Sampling

**Sample action via path integral**:

```python
def sample_action_path_integral(particles, s, lambda_temp, n_samples=100):
    """Sample action using path integral over trajectories."""
    theta_samples = []
    weights = []
    
    for _ in range(n_samples):
        # Sample candidate action
        theta = sample_from_prior(s)
        
        # Compute path action
        S = compute_action(particles, s, theta, lambda_temp)
        
        # Boltzmann weight
        w = np.exp(-S / lambda_temp)
        
        theta_samples.append(theta)
        weights.append(w)
    
    # Normalize
    weights = np.array(weights)
    weights /= weights.sum()
    
    # Sample from weighted distribution
    idx = np.random.choice(len(theta_samples), p=weights)
    return theta_samples[idx]

def compute_action(particles, s, theta, lambda_temp):
    """Compute action for single-step decision."""
    z = (s, theta)
    
    # Energy term (from particles)
    Q_plus = sum(p.w.real * kernel(z, p.z) for p in particles)
    E = -Q_plus
    
    # Kinetic term (assume small step)
    kinetic = 0.0  # Simplified for single step
    
    return E + kinetic / (2 * lambda_temp)
```

---

### 8.3 Interference Visualization

**Plot interference pattern** in action space:

```python
import matplotlib.pyplot as plt

def plot_interference(particles, s, theta_min, theta_max, n_points=200):
    """Plot real and imaginary parts of complex Q+."""
    theta_range = np.linspace(theta_min, theta_max, n_points)
    Q_real = np.zeros(n_points)
    Q_imag = np.zeros(n_points)
    
    for i, theta in enumerate(theta_range):
        z = (s, theta)
        Q_complex = sum(p.w * complex_kernel(z, p.z) for p in particles)
        Q_real[i] = Q_complex.real
        Q_imag[i] = Q_complex.imag
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(theta_range, Q_real, label='Re(Q+)')
    plt.xlabel('θ')
    plt.ylabel('Real part')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(theta_range, Q_imag, label='Im(Q+)')
    plt.xlabel('θ')
    plt.ylabel('Imaginary part')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    prob = Q_real**2 + Q_imag**2  # |ψ|^2
    plt.plot(theta_range, prob, label='|Q+|^2 (probability)')
    plt.xlabel('θ')
    plt.ylabel('Probability density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

---

## 9. Summary: Why Path Integrals Matter for GRL

**Quantum mechanical foundation**:

- GRL's Boltzmann policy is the imaginary-time path integral solution
- Not an analogy—a mathematical equivalence
- Connects RL to 70+ years of QM path integral techniques

**Complex-valued extension**:

- Enables true interference (constructive/destructive)
- Richer policy representations via phase
- Tunneling-like exploration through barriers

**Practical algorithms**:

- Path integral policy improvement (PI²)
- Langevin sampling as path integration
- Complex particle weights for interference

**Future directions**:

- Feynman diagrams for belief propagation
- Instanton calculus for rare events
- Hierarchical path integrals for concept composition

**This completes the quantum-inspired trilogy**:

1. **Chapters 01-02**: RKHS ↔ Hilbert space parallel
2. **Chapters 03-08**: Amplitude formulation and learning
3. **Chapter 09**: Path integrals as foundational formalism

---

## Further Reading

**Feynman Path Integrals**:

- Feynman, R. P., & Hibbs, A. R. (1965). *Quantum Mechanics and Path Integrals*. McGraw-Hill.
- Kleinert, H. (2009). *Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets* (5th ed.). World Scientific.

**Path Integral Control**:

- Kappen, H. J. (2005). "Path integrals and symmetry breaking for optimal control theory." *Journal of Statistical Mechanics*.
- Theodorou, E., Buchli, J., & Schaal, S. (2010). "A generalized path integral control approach to reinforcement learning." *JMLR*.
- Levine, S. (2018). "Reinforcement learning and control as probabilistic inference: Tutorial and review." *arXiv:1805.00909*.

**Quantum Computation & ML**:

- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.
- Wittek, P. (2014). *Quantum Machine Learning*. Academic Press.
- Biamonte, J., et al. (2017). "Quantum machine learning." *Nature*, 549, 195-202.

**Complex-Valued Neural Networks**:

- Trabelsi, C., et al. (2018). "Deep complex networks." *ICLR*.
- Virtue, P., Yu, S. X., & Lustig, M. (2017). "Better than real: Complex-valued neural nets for MRI fingerprinting." *ICIP*.

---

**[← Back to Chapter 08: Memory Dynamics](08-memory-dynamics-formation-consolidation-retrieval.md)** | **[Quantum-Inspired README](README.md)**

**[Related: Tutorial Chapter 03a - Least Action Principle](../tutorials/03a-least-action-principle.md)**
