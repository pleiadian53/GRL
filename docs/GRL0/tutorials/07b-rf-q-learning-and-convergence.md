# Chapter 07b: RF-Q-Learning and the Deadly Triad

**Purpose**: Analyze whether GRL's reinforcement field is susceptible to the divergence problems of off-policy learning, and whether an RF-Q-learning variant is viable  
**Prerequisites**: Chapter 07 (RF-SARSA), familiarity with the deadly triad (Sutton & Barto, Ch. 11)  
**Key Concepts**: Deadly triad, off-policy divergence, RKHS regularization, RF-Q-learning, structural stability

---

## Introduction

Chapter 7 introduced RF-SARSA as GRL's core learning algorithm. A natural question arises: **why SARSA and not Q-learning?**

In classical RL, Q-learning is often preferred over SARSA because it is off-policy — it can learn the optimal policy while following an exploratory behavior policy, and it can reuse experience from replay buffers. However, Q-learning with function approximation is notoriously unstable, a problem known as the **deadly triad**.

This chapter asks two questions:

1. **Does GRL's reinforcement field have any special protection against the deadly triad?**
2. **Is RF-Q-learning viable — and if so, under what conditions?**

The answers turn out to be nuanced: GRL's RKHS structure provides several structural safeguards that standard neural network approximators lack, but these safeguards mitigate rather than eliminate the fundamental instability. Understanding *why* reveals deep connections between kernel geometry, belief dynamics, and the nature of off-policy learning.

---

## 1. The Deadly Triad: A Recap

### 1.1 The three conditions

Sutton & Barto (2018, Ch. 11) identified three conditions that, when combined, can cause value estimates to diverge:

1. **Function approximation** — the value function is represented by a parameterized model (neural network, linear features, etc.) rather than a lookup table
2. **Bootstrapping** — the update target includes a value estimate (as in TD learning), rather than waiting for the full Monte Carlo return
3. **Off-policy learning** — the agent learns about a policy (the *target* policy) different from the one generating the data (the *behavior* policy)

Any two of these three are safe. All three together can cause unbounded growth of value estimates — even in simple MDPs.

### 1.2 Why the combination is dangerous

The intuition is:

- **Function approximation** means updating one state-action pair affects nearby ones (generalization)
- **Bootstrapping** means the update target depends on the current value estimates (self-referential)
- **Off-policy learning** means the distribution of updates doesn't match the distribution the target policy would visit

When all three combine, a positive error at one state can propagate to neighbors (via function approximation), inflate the bootstrap target (via bootstrapping), and never get corrected (because the behavior policy doesn't visit the states where the error is worst — off-policy). The result is a self-reinforcing feedback loop that drives values to infinity.

### 1.3 Where RF-SARSA sits

Let's check RF-SARSA against the triad:

| Condition | RF-SARSA | Present? |
|-----------|----------|----------|
| **Function approximation** | GP regression over particles in RKHS | ✓ Yes |
| **Bootstrapping** | TD update: $\delta = r + \gamma Q(s', a') - Q(s, a)$ | ✓ Yes |
| **Off-policy learning** | SARSA uses the *actual* next action $a'$ | ✗ No |

**RF-SARSA avoids the triad** by being on-policy. The primitive SARSA layer learns about the policy being executed, and the field layer generalizes these on-policy estimates. This is a deliberate design choice, not an accident.

---

## 2. What Would RF-Q-Learning Look Like?

### 2.1 The modification

RF-Q-learning would replace the SARSA update in the primitive layer with a Q-learning update:

**RF-SARSA (current)**:
$$\delta = r + \gamma Q(s', a') - Q(s, a) \quad \text{where } a' \text{ is the action actually taken}$$

**RF-Q-learning (proposed)**:
$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a) \quad \text{where } \max \text{ is over all actions}$$

For continuous actions, the $\max$ becomes an optimization:
$$\delta = r + \gamma \max_\theta Q^+(s', \theta) - Q^+(s, \theta)$$

which can be approximated via gradient ascent or Langevin sampling on the field.

### 2.2 What changes in the pipeline

The rest of the RF-SARSA pipeline would remain identical:

1. **Field queries** for action selection — unchanged
2. **Environment interaction** — unchanged
3. **Primitive update** — changed from SARSA to Q-learning (the $\max$)
4. **Particle reinforcement** via MemoryUpdate — unchanged (but now receives off-policy TD signals)
5. **ARD** — unchanged

The critical difference: the TD error $\delta$ now reflects the *optimal* policy's value, not the behavior policy's value. This $\delta$ is then propagated through MemoryUpdate into the particle memory, reshaping the field $Q^+$ toward the optimal value landscape rather than the behavior policy's landscape.

### 2.3 Why this is appealing

- **Sample efficiency**: Can reuse old experience (replay buffers)
- **Optimal policy learning**: Learns $Q^*$ directly, not $Q^\pi$
- **Exploration freedom**: Behavior policy can explore aggressively without corrupting value estimates

---

## 3. Does GRL's RKHS Structure Provide Protection?

This is the central question. GRL's reinforcement field has several structural properties that standard neural network function approximators lack. Let's examine each and assess whether it helps against the deadly triad.

### 3.1 RKHS norm regularization

**Property**: The GP posterior mean always lives in RKHS, and GP regression implicitly minimizes the RKHS norm:

$$Q^+(z) = \sum_i \alpha_i k(z, z_i) \quad \text{with} \quad \|Q^+\|_{\mathcal{H}_k}^2 = \alpha^\top K \alpha$$

The regularized GP objective is:

$$\min_\alpha \|q - K\alpha\|^2 + \sigma_n^2 \|\alpha\|_{K}^2$$

This penalizes functions with large RKHS norm — effectively bounding how "wild" the value function can become.

**Does this help?** **Partially.** RKHS regularization prevents the value function from developing arbitrarily sharp peaks or oscillations. In the deadly triad, divergence often manifests as unbounded growth of value estimates. The RKHS norm penalty resists this by penalizing large $\alpha$ coefficients. However, if the particle values $q_i$ themselves grow unboundedly (driven by bootstrapping), the GP will faithfully interpolate those growing values — the regularization controls the *shape* of $Q^+$, not the *scale* of the particle values feeding into it.

**Verdict**: Mitigates but does not eliminate divergence risk.

---

### 3.2 Kernel smoothness as implicit regularization

**Property**: Smooth kernels (RBF, Matérn) enforce smooth value functions. The field $Q^+(z)$ inherits the smoothness of the kernel $k(z, z')$.

**Does this help?** **Yes, significantly.** One mechanism of divergence in neural network Q-learning is that a large update at one state-action pair can cause *catastrophic* changes at distant states (due to the global nature of neural network parameter updates). In GRL, kernel smoothness ensures that:

- Updates propagate *locally* (weighted by kernel similarity)
- The influence of any single particle decays smoothly with distance
- No single update can cause a discontinuous jump in the field

This is analogous to the stability advantage of **local** function approximators (tile coding, RBFs) over **global** ones (neural networks) — and it is well-known that the deadly triad is less severe with local approximators.

**Verdict**: Provides meaningful protection against the *propagation* mechanism of divergence.

---

### 3.3 MemoryUpdate as a controlled belief transition

**Property**: MemoryUpdate (Algorithm 1) doesn't simply overwrite values — it performs a kernel-weighted association and update:

- New particles are associated with existing particles via kernel similarity
- Weight updates are modulated by the association strength
- The association threshold $\tau$ limits how far updates propagate

**Does this help?** **Yes.** MemoryUpdate acts as a **damping mechanism**. In standard Q-learning, the update $Q(s,a) \leftarrow Q(s,a) + \alpha \delta$ directly modifies the value at $(s,a)$. In RF-Q-learning, the TD error $\delta$ would be mediated by MemoryUpdate, which:

1. Checks whether the new particle associates with existing particles (kernel similarity $> \tau$)
2. If associated: blends the new value with existing particle values (weighted average)
3. If not associated: creates a new particle (no interference with existing field)

This means a single large TD error cannot arbitrarily distort the entire field — it is absorbed and smoothed by the particle ensemble. This is structurally similar to **experience averaging**, which is known to stabilize off-policy learning.

**Verdict**: Provides significant damping against error amplification.

---

### 3.4 Non-parametric representation avoids catastrophic interference

**Property**: GRL's particle memory is non-parametric — adding a new particle doesn't modify existing particles (unlike updating weights in a neural network, which changes predictions everywhere).

**Does this help?** **Yes, strongly.** One of the most pernicious aspects of the deadly triad with neural networks is **catastrophic interference**: updating the network to correct one state-action pair can silently corrupt estimates at other state-action pairs. This creates a "whack-a-mole" dynamic where fixing one error creates others.

In GRL, particles are independent data points. Adding particle $(z_{\text{new}}, q_{\text{new}})$ to $\Omega$ changes the GP posterior everywhere (because GP prediction depends on all data), but:

- The change is smooth and local (kernel-weighted)
- Existing particles retain their values
- The GP posterior is the *optimal* interpolant given all particles (no gradient descent instability)

This is a fundamental structural advantage of non-parametric methods over parametric ones for off-policy learning.

**Verdict**: Strong protection against the interference mechanism of divergence.

---

### 3.5 The remaining vulnerability: optimistic bias amplification

Despite all these safeguards, one fundamental problem remains.

**The $\max$ operator introduces systematic optimistic bias.** When we compute:

$$\max_\theta Q^+(s', \theta)$$

we are selecting the action parameter that maximizes the *estimated* field value. If the field has estimation errors (which it always does, especially in under-explored regions), the $\max$ preferentially selects the action where the error is most positive. This is the **maximization bias** (Sutton & Barto, Ch. 6.7).

In RF-Q-learning, this bias enters the particle values:

1. Compute $\delta = r + \gamma \max_\theta Q^+(s', \theta) - Q^+(s, \theta)$ — biased high
2. Update particle value $q \leftarrow Q(s,a) + \alpha \delta$ — biased high
3. GP interpolates biased particle values → field $Q^+$ is biased high everywhere
4. Next $\max$ query on biased field → even more biased
5. Feedback loop

The RKHS regularization controls the *shape* but not the *level* of $Q^+$. If all particle values drift upward uniformly, the GP will faithfully represent this uniform inflation — the RKHS norm doesn't penalize constant offsets.

**Verdict**: This is the primary remaining vulnerability. GRL's structure slows the feedback loop (via damping and smoothness) but does not break it.

---

## 4. Assessment: Is RF-Q-Learning Viable?

### 4.1 Summary of structural analysis

| Mechanism | Protection Level | What It Addresses |
|-----------|-----------------|-------------------|
| RKHS norm regularization | Moderate | Bounds function complexity, resists sharp peaks |
| Kernel smoothness | Significant | Prevents catastrophic propagation of errors |
| MemoryUpdate damping | Significant | Absorbs and smooths large TD errors |
| Non-parametric representation | Strong | Eliminates catastrophic interference |
| **Against maximization bias** | **Weak** | **Optimistic bias can still accumulate** |

### 4.2 The honest answer

**RF-Q-learning is more viable than standard neural-network-based Q-learning, but less reliable than RF-SARSA.**

GRL's RKHS structure provides genuine structural protection that neural networks lack. The kernel smoothness, MemoryUpdate damping, and non-parametric representation collectively address three of the four mechanisms by which the deadly triad causes divergence. The remaining vulnerability — maximization bias amplification — is the same fundamental issue that affects all forms of Q-learning with function approximation, and GRL's structure slows it but doesn't eliminate it.

**Practical expectation**: RF-Q-learning would likely work in many problems where neural-network Q-learning diverges, but would still be less stable than RF-SARSA in problems with:

- Sparse rewards (large regions of uncertain $Q^+$ → large maximization bias)
- High-dimensional action spaces (more opportunities for the $\max$ to find spurious peaks)
- Long horizons (more bootstrapping steps for bias to compound)

---

## 5. Making RF-Q-Learning More Reliable

If we want the sample efficiency benefits of off-policy learning while preserving GRL's stability, several strategies can help:

### 5.1 Double Q-learning in RKHS

**Idea**: Maintain two independent particle memories $\Omega_A$ and $\Omega_B$, and use one to select the maximizing action and the other to evaluate it:

$$\delta = r + \gamma Q_B^+(s', \arg\max_\theta Q_A^+(s', \theta)) - Q_A^+(s, \theta)$$

This breaks the maximization bias because the action selected by $Q_A^+$ is evaluated by $Q_B^+$, which has independent estimation errors.

**In GRL**: This requires maintaining two separate particle memories and alternating which one is updated. The computational cost roughly doubles, but the stability improvement is substantial.

**GRL advantage**: Unlike neural network Double DQN (which requires two separate networks), GRL's two particle memories are naturally independent — they share the kernel but have separate particles. This makes the independence assumption more credible.

### 5.2 Pessimistic value estimation (conservative Q-learning)

**Idea**: Instead of using the GP posterior mean for the $\max$, use a **lower confidence bound**:

$$\max_\theta \left[ Q^+(s', \theta) - \kappa \, \sigma(s', \theta) \right]$$

where $\sigma(s', \theta)$ is the GP posterior standard deviation and $\kappa > 0$ controls pessimism.

**Why this helps**: In under-explored regions, $\sigma$ is large, so the lower confidence bound is low. The $\max$ will avoid selecting actions in uncertain regions, reducing the optimistic bias.

**GRL advantage**: GP regression provides uncertainty estimates $\sigma(z)$ for free — this is a natural capability of the reinforcement field that neural networks lack (or require expensive ensembles to approximate). This makes pessimistic RF-Q-learning particularly natural in GRL.

### 5.3 Soft maximization (mellowmax or Boltzmann)

**Idea**: Replace the hard $\max$ with a soft version:

$$\text{softmax}_\beta Q^+(s', \theta) = \frac{1}{\beta} \log \mathbb{E}_{\theta \sim \text{Uniform}} \left[ \exp(\beta \, Q^+(s', \theta)) \right]$$

For finite $\beta$, this averages over actions rather than selecting the single best one, reducing the maximization bias.

**Connection to GRL**: The Boltzmann policy $\pi(\theta | s) \propto \exp(\beta Q^+(s, \theta))$ already uses soft maximization for action *selection*. Extending this to the bootstrap *target* creates a consistent framework:

$$\delta = r + \gamma \mathbb{E}_{\theta' \sim \pi(\cdot | s')} [Q^+(s', \theta')] - Q^+(s, \theta)$$

This is essentially **Expected SARSA** — an algorithm that interpolates between SARSA ($\beta \to 0$) and Q-learning ($\beta \to \infty$). In GRL, this is particularly natural because the Boltzmann policy is already the default action selection mechanism.

### 5.4 Experience replay with kernel-weighted prioritization

**Idea**: Since RF-Q-learning is off-policy, we can maintain a replay buffer and resample old transitions. Prioritize transitions where the TD error is large *and* the kernel similarity to current particles is high:

$$p(\text{replay } i) \propto |\delta_i| \cdot \max_j k(z_i, z_j)$$

This focuses replay on transitions that are both surprising (large $|\delta_i|$) and relevant to the current field (high kernel similarity).

**GRL advantage**: The kernel provides a natural relevance metric that neural network replay methods lack.

---

## 6. RF-Q-Learning vs. RF-SARSA: When to Use Which

| Criterion | RF-SARSA | RF-Q-Learning |
|-----------|----------|---------------|
| **Stability** | High (on-policy, no maximization bias) | Moderate (off-policy, requires safeguards) |
| **Sample efficiency** | Lower (discards off-policy data) | Higher (can reuse experience) |
| **Learned policy** | Behavior policy $Q^\pi$ | Optimal policy $Q^*$ |
| **Exploration impact** | Reflected in value estimates | Not reflected (learns greedy values) |
| **Risk sensitivity** | Risk-averse (accounts for exploration mistakes) | Risk-neutral (assumes optimal execution) |
| **Implementation complexity** | Simpler | More complex (needs Double Q or pessimism) |
| **Best for** | Safety-critical, stable learning | Sample-limited, known-safe environments |

### Practical recommendation

**Start with RF-SARSA** (Chapter 7) as the default. Switch to RF-Q-learning only when:

1. Sample efficiency is critical (expensive environment interactions)
2. The action space is well-explored (reducing maximization bias risk)
3. You implement at least one safeguard (Double Q, pessimism, or soft max)

The **Expected SARSA** variant (Section 5.3) is a particularly attractive middle ground — it provides some off-policy benefit while retaining most of SARSA's stability.

---

## 7. A Deeper Perspective: Why GRL Is Structurally Different

### 7.1 The deadly triad through GRL's lens

The deadly triad is fundamentally about **uncontrolled error propagation** through three channels:

1. **Spatial propagation** (function approximation): errors at one state affect others
2. **Temporal propagation** (bootstrapping): errors compound across time steps
3. **Distributional mismatch** (off-policy): errors accumulate in unvisited regions

GRL's architecture addresses each channel differently than standard RL:

**Spatial propagation** is controlled by the kernel. In neural networks, a weight update affects predictions globally and unpredictably. In GRL, the kernel defines a precise, smooth, and local influence function. The "blast radius" of any single update is bounded by the kernel lengthscale.

**Temporal propagation** is mediated by MemoryUpdate. Rather than directly modifying a parameter vector, TD errors are absorbed into the particle ensemble through a controlled association process. This acts as a low-pass filter on the temporal error signal.

**Distributional mismatch** remains the primary vulnerability. GRL's non-parametric representation helps (no catastrophic forgetting), but the maximization bias is a statistical phenomenon that no representation can fully eliminate — it requires algorithmic solutions (Double Q, pessimism, etc.).

### 7.2 The particle memory as a stabilizing mechanism

There is a deeper reason why GRL's particle-based representation is more stable than parametric approximators for off-policy learning.

In parametric Q-learning (e.g., DQN), the value function is $Q_w(s, a)$ where $w$ is a weight vector. An update at $(s, a)$ changes $w$, which changes $Q_w$ *everywhere*. The mapping from update to global effect is mediated by the network architecture — complex, nonlinear, and hard to predict.

In GRL, the value function is $Q^+(z) = \sum_i \alpha_i k(z, z_i)$. An update at $z$ either:

- **Modifies an existing particle's weight** $\alpha_i$ (local effect, kernel-bounded)
- **Adds a new particle** $(z_{\text{new}}, q_{\text{new}})$ (enriches the field without disturbing existing particles)

This is a fundamentally more controlled update mechanism. The particle memory acts as a **buffer** between raw TD signals and the value function, absorbing noise and preventing the kind of rapid, global changes that trigger divergence in neural networks.

### 7.3 Connection to kernel TD convergence theory

Engel et al. (2005) showed that kernel-based TD learning converges under milder conditions than linear TD. The key insight is that the RKHS provides a **fixed feature space** (the kernel sections $k(z_i, \cdot)$) that doesn't change during learning — unlike neural network features, which co-adapt with the value function.

This fixed-feature property means that kernel TD is equivalent to **linear TD in a (potentially infinite-dimensional) feature space**, and linear TD has well-understood convergence guarantees (Tsitsiklis & Van Roy, 1997). Off-policy linear TD can still diverge, but the conditions for divergence are more restrictive and better characterized.

For RF-Q-learning, this suggests that convergence is more likely when:

- The kernel is well-chosen (captures the relevant structure of the value function)
- The particle set provides good coverage of the state-action space
- ARD keeps the kernel adapted to the current value landscape

---

## 8. Summary

### 8.1 Does the reinforcement field suffer from the deadly triad?

**Not in its current form (RF-SARSA)**, because it is on-policy. The triad requires all three conditions, and RF-SARSA only has two (function approximation + bootstrapping).

**An RF-Q-learning variant would introduce the third condition** (off-policy learning), but GRL's RKHS structure provides four layers of protection that standard neural network approximators lack:

1. **RKHS norm regularization** — bounds function complexity
2. **Kernel smoothness** — prevents catastrophic error propagation
3. **MemoryUpdate damping** — absorbs and smooths TD errors
4. **Non-parametric representation** — eliminates catastrophic interference

These safeguards make RF-Q-learning **more viable than neural-network Q-learning**, but they do not eliminate the fundamental **maximization bias** that is intrinsic to the $\max$ operator.

### 8.2 Is RF-Q-learning possible and reliable?

**Possible**: Yes, the modification is straightforward (replace SARSA update with Q-learning update in the primitive layer).

**Reliable**: Conditionally. With appropriate safeguards (Double Q, pessimistic estimation, or soft maximization), RF-Q-learning can be made practical. The **Expected SARSA** variant is particularly natural in GRL, since the Boltzmann policy already provides a soft maximization mechanism.

### 8.3 The key insight

> **The deadly triad is about uncontrolled error propagation. GRL's kernel-based architecture controls three of the four propagation channels. The remaining channel (maximization bias) requires algorithmic, not architectural, solutions.**

This is why RF-SARSA was the right default choice for GRL: it eliminates the one vulnerability that the architecture cannot address. But for practitioners who need sample efficiency, RF-Q-learning with safeguards is a viable and theoretically grounded alternative.

---

## 9. Open Questions

1. **Convergence guarantees**: Can we prove convergence of RF-Q-learning with Double Q or pessimistic estimation in RKHS? The kernel TD convergence theory (Engel et al., 2005) provides a starting point.

2. **Optimal pessimism level**: How should $\kappa$ (the pessimism coefficient in Section 5.2) be set? Can it be adapted based on the GP uncertainty structure?

3. **Replay buffer design**: What is the optimal replay strategy for RF-Q-learning? Should old particles be replayed, or should we maintain a separate replay buffer alongside the particle memory?

4. **Empirical comparison**: How does RF-Q-learning (with safeguards) compare to RF-SARSA in practice? In which domains does the sample efficiency gain outweigh the stability cost?

5. **Connection to SAC**: Soft Actor-Critic (Haarnoja et al., 2018) uses entropy-regularized Q-learning, which is closely related to the Boltzmann/Expected SARSA approach in Section 5.3. Can GRL's RKHS critic be combined with SAC's maximum entropy framework?

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapters 6.7 and 11.
2. Engel, Y., Mannor, S., & Meir, R. (2005). "Reinforcement learning with Gaussian processes." *ICML*.
3. Tsitsiklis, J. N., & Van Roy, B. (1997). "An analysis of temporal-difference learning with function approximation." *IEEE TAC*.
4. van Hasselt, H. (2010). "Double Q-learning." *NeurIPS*.
5. van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." *AAAI*.
6. Kumar, A., et al. (2020). "Conservative Q-learning for offline reinforcement learning." *NeurIPS* (CQL).
7. Haarnoja, T., et al. (2018). "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." *ICML*.
8. Chiu, C.-C. & Huber, M. (2022). "Generalized Reinforcement Learning." [arXiv:2208.04822](https://arxiv.org/abs/2208.04822).

---

**[← Back to Chapter 07: RF-SARSA](07-rf-sarsa.md)** | **[Related: Chapter 07a - Continuous Policy Inference](07a-continuous-policy-inference.md)**

---

*Last Updated*: February 2026
