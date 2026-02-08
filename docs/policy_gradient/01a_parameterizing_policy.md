# Parameterizing the Policy: From Gaussians to Real-World Control

**How $\pi_\theta(a \mid s)$ is actually implemented — and why it matters**

> *Prerequisites: [Policy Gradient Fundamentals](01_PG.md)*

---

## 1. What Does "Parameterizing a Policy" Mean?

Every policy gradient method — REINFORCE, TRPO, PPO — optimizes a parameterized policy $\pi_\theta(a \mid s)$. But the theory is silent on *what form* this distribution takes. That choice is left to the practitioner, and it determines whether the algorithm stays on a whiteboard or controls a real robot.

Parameterizing a policy means answering three questions:

1. **Action space**: Is $a$ discrete, continuous, or hybrid?
2. **Distribution family**: What class of distributions do we use (categorical, Gaussian, mixture, flow)?
3. **State-dependence**: Which parameters of that distribution are learned functions of $s$?

This chapter walks through the major parameterizations, from textbook examples to production-grade systems used in robotics and autonomous driving.

---

## 2. Discrete Actions: Softmax Categorical Policy

The simplest case. The action space is finite: $a \in \{1, 2, \ldots, K\}$.

A neural network $f_\theta(s) \in \mathbb{R}^K$ produces **logits**, and softmax converts them to probabilities:

$$\pi_\theta(a = i \mid s) = \frac{\exp(f_\theta(s)_i)}{\sum_{j=1}^K \exp(f_\theta(s)_j)}$$

**Log-probability** (needed for policy gradient):

$$\log \pi_\theta(a = i \mid s) = f_\theta(s)_i - \log \sum_{j=1}^K \exp(f_\theta(s)_j)$$

**Concrete example — lane selection in highway driving:**

| Action $a$ | Meaning |
|------------|---------|
| 0 | Stay in current lane |
| 1 | Change to left lane |
| 2 | Change to right lane |

The network takes as input the ego vehicle state (position, velocity, surrounding vehicles) and outputs 3 logits. Softmax gives the probability of each lane decision.

**Where this is used:**

- Atari game-playing agents (DQN, A3C, PPO)
- High-level decision layers in autonomous driving (lane change, intersection behavior)
- Discrete planning and scheduling

Discrete policies remain important even in "continuous" domains — many real systems have a discrete strategic layer on top of continuous low-level control.

---

## 3. Diagonal Gaussian Policy: Continuous Control Basics

For continuous actions $a \in \mathbb{R}^d$, the most common choice is a **diagonal Gaussian**:

$$\pi_\theta(a \mid s) = \mathcal{N}\big(a \;\big|\; \mu_\theta(s), \; \text{diag}(\sigma_\theta^2(s))\big)$$

where:

- $\mu_\theta(s) \in \mathbb{R}^d$: mean, output of a neural network
- $\sigma_\theta(s) \in \mathbb{R}^d_{>0}$: per-dimension standard deviation (often parameterized as $\log \sigma$ for numerical stability)

**Sampling** uses the reparameterization trick:

$$a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Log-probability** (closed-form):

$$\log \pi_\theta(a \mid s) = -\frac{1}{2} \sum_{i=1}^d \left[ \frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i + \log 2\pi \right]$$

**KL divergence** between two diagonal Gaussians (closed-form — important for TRPO):

$$D_{\mathrm{KL}}(\pi_{\theta_k} \| \pi_\theta) = \frac{1}{2} \sum_{i=1}^d \left[ \frac{\sigma_{k,i}^2}{\sigma_i^2} + \frac{(\mu_i - \mu_{k,i})^2}{\sigma_i^2} - 1 + 2\log\frac{\sigma_i}{\sigma_{k,i}} \right]$$

**Concrete example — 7-DOF robotic arm (Franka Panda):**

The action is a vector of 7 joint torques: $a = (\tau_1, \tau_2, \ldots, \tau_7) \in \mathbb{R}^7$. The state includes joint angles, joint velocities, and end-effector pose. The network outputs 7 means and 7 log-standard-deviations.

**Strengths:**

- Closed-form log-prob, KL, and entropy — fast and numerically stable
- Well-understood gradient behavior
- Default choice in MuJoCo benchmarks, locomotion (quadrupeds, bipeds), and manipulation

**Limitations:**

- **Unimodal**: can only represent one "strategy" at a time
- **Independent dimensions**: assumes action dimensions are uncorrelated (diagonal covariance)
- **Unbounded**: samples can exceed physical actuator limits

---

## 4. Squashed Gaussian: Bounded Actuators

Real actuators have physical limits. A joint torque might be bounded to $[-10, 10]$ Nm. A steering angle to $[-0.5, 0.5]$ rad. Raw Gaussian samples can violate these bounds.

The **squashed Gaussian** (used in SAC and many PPO implementations) applies a $\tanh$ transform:

$$u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s)), \quad a = \tanh(u)$$

Now $a \in (-1, 1)^d$, which can be rescaled to any bounded interval.

**Log-probability** requires a change-of-variables correction:

$$\log \pi_\theta(a \mid s) = \log \mathcal{N}(u \mid \mu, \sigma^2) - \sum_{i=1}^d \log(1 - a_i^2)$$

The second term accounts for the Jacobian of the $\tanh$ transformation. Omitting it is a common implementation bug that causes silent training instability.

**Concrete example — differential-drive mobile robot:**

- Action: $(v, \omega) \in [-1, 1]^2$ (linear velocity, angular velocity)
- Rescaled to physical limits: $v \in [-0.5, 0.5]$ m/s, $\omega \in [-1.0, 1.0]$ rad/s
- The squashing ensures the policy never commands physically impossible velocities

This is the standard parameterization for **production robotics** and **sim-to-real transfer**.

---

## 5. Factorized Policies: Hybrid and Structured Action Spaces

Real-world systems often have **mixed** action spaces — some dimensions continuous, some discrete.

**Concrete example — autonomous vehicle:**

| Component | Type | Distribution |
|-----------|------|-------------|
| Steering angle $\alpha$ | Continuous, $[-0.5, 0.5]$ rad | Squashed Gaussian |
| Acceleration $\tau$ | Continuous, $[-3, 3]$ m/s$^2$ | Squashed Gaussian |
| Turn signal $d$ | Discrete, $\{$off, left, right$\}$ | Categorical |

The policy factorizes as a product of independent sub-policies:

$$\pi_\theta(a \mid s) = \pi_{\text{steer}}(\alpha \mid s) \cdot \pi_{\text{accel}}(\tau \mid s) \cdot \pi_{\text{signal}}(d \mid s)$$

**Why this works with PPO:**

- Log-probabilities **add**: $\log \pi = \log \pi_{\text{steer}} + \log \pi_{\text{accel}} + \log \pi_{\text{signal}}$
- Gradients flow independently through each head
- The probability ratio $r_t(\theta)$ decomposes cleanly in log-space

**Implementation**: a single neural network backbone with multiple output heads — one per action component. Each head has its own distribution type.

This factorized design is standard in **production autonomy stacks** and **multi-actuator robotics**.

---

## 6. Mixture of Gaussians: Multimodal Decisions

Many control problems are inherently **multimodal** — there are multiple qualitatively different good actions:

- Overtake the car ahead **or** slow down and follow
- Grasp the mug from the left **or** from the right
- Step over the obstacle **or** walk around it

A single Gaussian cannot represent this. A **mixture of Gaussians** (MoG) can:

$$\pi_\theta(a \mid s) = \sum_{k=1}^K w_k(s) \; \mathcal{N}\big(a \;\big|\; \mu_k(s), \Sigma_k(s)\big)$$

where:

- $w_k(s) = \text{softmax}(h_\theta(s))_k$: state-dependent mixture weights
- $\mu_k(s), \Sigma_k(s)$: mean and covariance of the $k$-th component

**Log-probability** (log-sum-exp):

$$\log \pi_\theta(a \mid s) = \log \sum_{k=1}^K w_k(s) \; \mathcal{N}(a \mid \mu_k(s), \Sigma_k(s))$$

**Concrete example — manipulation with contact:**

A robot hand reaching for an object on a cluttered table. With $K = 3$ mixture components:

- Component 1: approach from above (top grasp)
- Component 2: approach from the side (pinch grasp)
- Component 3: push object to a clearer location first

Each component represents a distinct strategy. The mixture weights $w_k(s)$ learn which strategy is best given the current scene.

**Challenges for PPO:**

- KL divergence between mixtures has **no closed form** — must be estimated
- Gradient variance is higher (the mixture assignment adds stochasticity)
- PPO still works, but requires more careful tuning of the clip range and learning rate

---

## 7. Normalizing Flow Policies: Expressive Modern Control

For the most demanding control tasks — dexterous manipulation, contact-rich locomotion, agile flight — even mixtures may not be expressive enough. **Normalizing flows** define arbitrarily complex distributions through a sequence of invertible transformations:

$$z \sim \mathcal{N}(0, I), \quad a = f_\theta(z; s)$$

where $f_\theta$ is an invertible, differentiable function (e.g., a sequence of affine coupling layers).

**Log-probability** via the change-of-variables formula:

$$\log \pi_\theta(a \mid s) = \log \mathcal{N}(f_\theta^{-1}(a; s)) - \log \left| \det \frac{\partial f_\theta}{\partial z} \right|$$

**What this buys:**

- **Multimodal**: can represent any number of modes
- **Correlated**: captures dependencies between action dimensions (e.g., coordinated finger movements)
- **Arbitrarily expressive**: universal approximator for continuous distributions

**Where this is used:**

- Dexterous manipulation (Allegro hand, Shadow hand)
- Contact-rich locomotion on uneven terrain
- Agile quadrotor flight

PPO still applies — you compute $\log \pi_\theta(a \mid s)$, form the ratio, clip, and optimize. The flow is just a more powerful function inside the same algorithmic shell.

---

## 8. Trajectory-Level Policies: Actions as Plans

In autonomous driving and long-horizon robotics, the "action" is not an instantaneous torque — it is an **entire trajectory** or motion plan.

$$\pi_\theta(\tau \mid s)$$

where $\tau$ might be:

- A sequence of future waypoints: $\tau = (p_1, p_2, \ldots, p_H) \in \mathbb{R}^{H \times 2}$
- Coefficients of a spline or polynomial path
- Latent codes decoded into a motion plan

**Concrete example — trajectory prediction in autonomous driving:**

The policy outputs a distribution over 5-second future trajectories (50 waypoints at 10 Hz). Each waypoint is $(x, y)$ in the vehicle frame. The distribution might be:

- A Gaussian over spline coefficients (compact, smooth)
- A mixture of trajectories (multimodal: turn left vs. go straight)
- A diffusion model (emerging approach — iteratively denoises a trajectory)

PPO operates at the **trajectory level**: the reward evaluates the full plan (safety, comfort, progress), and policy gradients push the distribution toward better plans.

---

## 9. The Unifying Requirement

Across all these parameterizations — categorical, Gaussian, squashed, factorized, mixture, flow, trajectory — policy gradient methods need exactly **three things**:

1. **Evaluable log-probability**: $\log \pi_\theta(a \mid s)$ must be computable
2. **Differentiable parameters**: $\nabla_\theta \log \pi_\theta(a \mid s)$ must exist (for autograd)
3. **Smooth KL behavior**: small changes in $\theta$ should produce small changes in the policy distribution

If these three conditions hold, PPO (and TRPO, REINFORCE, etc.) work unchanged. The algorithm doesn't care what's inside the policy — it only interacts with it through log-probabilities and their gradients.

| Parameterization | Log-prob | KL (closed-form?) | Expressiveness |
|-----------------|----------|-------------------|----------------|
| Categorical | Exact | Yes | Low (discrete) |
| Diagonal Gaussian | Exact | Yes | Low (unimodal) |
| Squashed Gaussian | Exact (with Jacobian) | Approximate | Low (unimodal, bounded) |
| Factorized hybrid | Sum of components | Per-component | Medium |
| Mixture of Gaussians | Log-sum-exp | No (estimate) | Medium (multimodal) |
| Normalizing flow | Exact (with Jacobian) | No (estimate) | High |
| Trajectory-level | Depends on model | Depends on model | High |

---

## 10. Where Simple Parameterizations Break Down

As policies become more expressive, new problems emerge that stress-test the policy gradient machinery:

- **High-dimensional actions**: a humanoid robot has 20+ joint torques. The ratio $r_t(\theta)$ becomes a product of many per-dimension terms, and its variance grows exponentially.
- **Multimodal policies**: the KL divergence between mixtures is hard to estimate, making trust regions unreliable.
- **Sharp, brittle distributions**: a near-deterministic policy (small $\sigma$) makes the ratio $r_t(\theta)$ extremely sensitive to small parameter changes.

These are not hypothetical — they are the practical reasons why PPO struggles on the hardest control tasks and why newer methods like GRPO exist. We return to this in [Chapter 04](04_PPO_variants.md).

---

## 11. Connection to Parametric Actions in GRL

The policy parameterization problem has a deep connection to GRL's concept of **parametric actions** (see [GRL Chapter 01: Core Concepts](../GRL0/tutorials/01-core-concepts.md)).

In standard policy gradient, we parameterize the *distribution over actions*: $\pi_\theta(a \mid s)$.

In GRL, we go further — we parameterize the *actions themselves* as operators:

$$\theta_a \to \hat{O}(\theta_a)$$

where $\theta_a$ is an action parameter vector and $\hat{O}$ is the operator it specifies (a force, a torque, a trajectory). The value function $Q^+(s, \theta_a)$ is defined over the joint augmented space, and the policy emerges from navigating this value landscape (see [GRL Chapter 04: Reinforcement Field](../GRL0/tutorials/04-reinforcement-field.md)).

This raises a natural question: **can we combine the expressiveness of GRL's parametric action framework with the optimization machinery of policy gradient methods?** We explore this connection in [Chapter 05: Actions as Operators](05_actions_as_operators.md).

---

*Previous: [Policy Gradient Fundamentals](01_PG.md)* | *Next: [Trust Region Policy Optimization (TRPO)](02_TRPO.md)*

*See also: [Actions as Operators](05_actions_as_operators.md) — from flat vectors to structured control*
