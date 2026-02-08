# Trust Region Policy Optimization (TRPO)

**A tutorial on stable policy updates via constrained optimization**

> *Prerequisites: Familiarity with basic reinforcement learning concepts (states, actions, rewards, policies). A companion tutorial on vanilla policy gradient and actor-critic foundations is forthcoming.*

---

## 1. Introduction

Trust Region Policy Optimization (TRPO) was introduced by [Schulman et al. (2015)][trpo-paper] as a principled approach to a long-standing problem in policy gradient methods: **how do you take the largest possible improvement step without accidentally destroying a good policy?**

TRPO is an **on-policy** algorithm that works with both discrete and continuous action spaces. It serves as the direct predecessor to [Proximal Policy Optimization (PPO)](03_PPO.md), which simplifies TRPO's constrained optimization into a clipped objective — but understanding TRPO first makes PPO's design choices much clearer.

---

## 2. Motivation: Why Not Just Use Vanilla Policy Gradient?

### The vanilla policy gradient update

The simplest policy gradient method (REINFORCE) updates parameters by ascending the gradient of expected return:

$$\theta_{k+1} = \theta_k + \alpha \, \hat{g}_k$$

where $\hat{g}_k$ is an estimate of $\nabla_\theta J(\theta)$, the gradient of the expected cumulative reward with respect to the policy parameters, and $\alpha$ is a fixed learning rate.

### The problem: sensitivity to step size

This looks straightforward, but in practice it is **extremely fragile**:

- **Too large a step** — the policy changes drastically, performance collapses, and the agent may never recover. In deep RL this is especially dangerous because the policy is a neural network: a small change in $\theta$ can produce a very different distribution over actions.
- **Too small a step** — learning is painfully slow, and the agent wastes millions of environment interactions making negligible progress.

Unlike supervised learning, where a bad gradient step just increases the loss temporarily, a bad policy gradient step **changes the data distribution itself** (because the policy determines which states the agent visits). This creates a vicious cycle: a bad update leads to bad data, which leads to worse updates.

### The core insight

What we really want is not to limit the step size in **parameter space** ($\|\Delta\theta\|$), but in **policy space** — i.e., we want to ensure the new policy $\pi_{\theta_{k+1}}$ behaves similarly to the old policy $\pi_{\theta_k}$ in terms of the actions it actually produces. Two parameter vectors that are close in Euclidean distance can produce wildly different policies, and vice versa.

> **Key idea:** TRPO replaces the fixed learning rate with a *constraint on how much the policy is allowed to change*, measured by KL divergence in the space of action distributions.

---

## 3. Notation Reference

Before diving into the algorithm, here is a summary of the notation used throughout:

| Symbol | Meaning |
|--------|---------|
| $s_t$ | State at time step $t$ |
| $a_t$ | Action taken at time step $t$ |
| $\pi_\theta(a \mid s)$ | Policy: probability of taking action $a$ in state $s$, parameterized by $\theta$ |
| $\theta_k$ | Policy parameters at iteration $k$ |
| $V_\phi(s)$ | Value function (expected return from state $s$), parameterized by $\phi$ |
| $\hat{R}_t$ | **Reward-to-go**: the sum of (discounted) rewards from time $t$ onward, $\hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ |
| $\hat{A}_t$ | **Advantage estimate**: how much better action $a_t$ is compared to the average action under the current policy, $\hat{A}_t \approx Q(s_t, a_t) - V(s_t)$ |
| $\tau$ | A trajectory (sequence of states, actions, rewards): $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ |
| $\mathcal{D}_k$ | Batch of trajectories collected at iteration $k$ |
| $D_{\mathrm{KL}}$ | Kullback-Leibler divergence between two probability distributions |
| $\delta$ | Maximum allowed KL divergence (trust region radius) |
| $\hat{H}_k$ | **Fisher information matrix** (equivalently, the Hessian of the KL divergence w.r.t. $\theta$) |
| $\hat{g}_k$ | Estimated policy gradient at iteration $k$ |
| $\alpha$ | Backtracking coefficient (used in line search, typically $\alpha \in (0, 1)$) |
| $K$ | Maximum number of backtracking steps |

---

## 4. Key Ideas Behind TRPO

TRPO rests on three interlocking ideas:

### 4.1 The surrogate objective

Rather than directly optimizing expected return (which requires new on-policy samples for every candidate $\theta$), TRPO optimizes a **surrogate objective** that can be evaluated using data collected under the *current* policy $\pi_{\theta_k}$:

$$L(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} \hat{A}_t \right]$$

The ratio $\frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}$ is called the **importance sampling ratio**. It reweights the old data to account for the fact that we are evaluating a *different* policy $\pi_\theta$ than the one that collected the data $\pi_{\theta_k}$.

**Intuition:** If the new policy assigns *higher* probability to actions that had *positive* advantage (i.e., were better than average), the surrogate objective increases — which is exactly what we want.

### 4.2 The trust region constraint

The surrogate objective is only a good approximation to the true objective when $\pi_\theta$ is close to $\pi_{\theta_k}$. TRPO enforces this by adding a **constraint**:

$$\max_\theta \; L(\theta) \quad \text{subject to} \quad \bar{D}_{\mathrm{KL}}(\pi_{\theta_k} \| \pi_\theta) \leq \delta$$

where $\bar{D}_{\mathrm{KL}}$ is the **average KL divergence** over states visited under $\pi_{\theta_k}$:

$$\bar{D}_{\mathrm{KL}}(\pi_{\theta_k} \| \pi_\theta) = \mathbb{E}_{s \sim \pi_{\theta_k}} \left[ D_{\mathrm{KL}}\big(\pi_{\theta_k}(\cdot \mid s) \;\|\; \pi_\theta(\cdot \mid s)\big) \right]$$

The hyperparameter $\delta$ controls the **radius of the trust region** — how far the new policy is allowed to deviate from the old one. Typical values are $\delta \in [0.001, 0.05]$.

**Why KL divergence?** KL divergence measures how different two probability distributions are. Unlike Euclidean distance in parameter space, it directly captures how differently the two policies *behave*. A small KL divergence guarantees that the surrogate objective remains a faithful approximation.

### 4.3 Solving the constrained problem efficiently

Directly solving the constrained optimization above would require computing and inverting the Hessian matrix $\hat{H}_k$ of the KL divergence — an $n \times n$ matrix where $n$ is the number of policy parameters. For neural network policies with millions of parameters, this is infeasible.

TRPO's practical contribution is a two-part trick:

1. **Conjugate gradient (CG) method** — computes $\hat{H}_k^{-1} \hat{g}_k$ *without ever forming or storing* $\hat{H}_k$. CG only needs **Hessian-vector products** $\hat{H}_k v$, which can be computed efficiently via automatic differentiation.

2. **Backtracking line search** — after computing the search direction, TRPO tries progressively smaller steps until it finds one that (a) actually improves the surrogate objective and (b) satisfies the KL constraint.

---

## 5. The TRPO Algorithm

Here is the full algorithm, with each step explained:

---

**Input:** Initial policy parameters $\theta_0$, initial value function parameters $\phi_0$

**Hyperparameters:** KL-divergence limit $\delta$, backtracking coefficient $\alpha$, maximum backtracking steps $K$

**For** $k = 0, 1, 2, \ldots$ **do:**

### Step 1 — Collect trajectories

> Run the current policy $\pi_k = \pi(\theta_k)$ in the environment to collect a batch of trajectories $\mathcal{D}_k = \{\tau_i\}$.

Each trajectory $\tau_i = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ is a complete episode (or a fixed-length rollout). This is the "on-policy" part: we use fresh data from the current policy at every iteration.

### Step 2 — Compute rewards-to-go

> For each time step $t$ in each trajectory, compute $\hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$.

The reward-to-go tells us the total (discounted) return the agent actually received from time $t$ onward. This serves as a target for fitting the value function.

### Step 3 — Compute advantage estimates

> Compute advantage estimates $\hat{A}_t$ based on the current value function $V_{\phi_k}$.

The advantage $\hat{A}_t \approx Q(s_t, a_t) - V(s_t)$ measures how much better the action taken was compared to the average. Common methods include:

- **Simple:** $\hat{A}_t = \hat{R}_t - V_{\phi_k}(s_t)$
- **GAE (Generalized Advantage Estimation):** a weighted blend of multi-step advantage estimates that trades off bias and variance via a parameter $\lambda$.

### Step 4 — Estimate the policy gradient

> Compute:
>
> $$\hat{g}_k = \frac{1}{|\mathcal{D}_k|} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big|_{\theta_k} \hat{A}_t$$

This is the standard policy gradient (the "score function" estimator). Each term $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ points in the direction that increases the probability of action $a_t$, and it is weighted by $\hat{A}_t$ so that *good* actions are reinforced and *bad* actions are suppressed.

### Step 5 — Compute the search direction via conjugate gradient

> Approximately solve for $\hat{x}_k$:
>
> $$\hat{x}_k \approx \hat{H}_k^{-1} \hat{g}_k$$
>
> where $\hat{H}_k$ is the Hessian of the sample average KL divergence with respect to $\theta$.

This is the **natural gradient** direction. While the ordinary gradient $\hat{g}_k$ gives the steepest ascent direction in parameter space, the natural gradient $\hat{H}_k^{-1} \hat{g}_k$ gives the steepest ascent direction in **policy distribution space** — accounting for the geometry of the probability manifold.

The conjugate gradient algorithm computes this in roughly 10–20 iterations, each requiring one Hessian-vector product (computed via automatic differentiation), without ever forming the full Hessian matrix.

### Step 6 — Update the policy via backtracking line search

> Set:
>
> $$\theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2\delta}{\hat{x}_k^T \hat{H}_k \hat{x}_k}} \; \hat{x}_k$$
>
> where $j \in \{0, 1, 2, \ldots, K\}$ is the smallest value such that the update (a) improves the surrogate loss and (b) satisfies $\bar{D}_{\mathrm{KL}}(\pi_{\theta_k} \| \pi_{\theta_{k+1}}) \leq \delta$.

**Breaking this down:**

- The term $\sqrt{\frac{2\delta}{\hat{x}_k^T \hat{H}_k \hat{x}_k}}$ computes the **maximum step size** that would exactly hit the trust region boundary $\delta$, assuming a local quadratic approximation to the KL divergence.
- The factor $\alpha^j$ (with $\alpha < 1$, e.g., $\alpha = 0.5$) **shrinks** the step if the full step doesn't satisfy the constraints. The algorithm tries $j = 0$ (full step) first, then $j = 1$ (half step), then $j = 2$ (quarter step), etc.
- This ensures we take the **largest safe step** that genuinely improves the policy.

### Step 7 — Update the value function

> Fit the value function by minimizing mean-squared error:
>
> $$\phi_{k+1} = \arg\min_\phi \frac{1}{|\mathcal{D}_k| T} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^{T} \left( V_\phi(s_t) - \hat{R}_t \right)^2$$
>
> typically via a few steps of gradient descent (e.g., Adam).

The value function is used to compute advantage estimates in the next iteration. A better value function leads to lower-variance advantage estimates, which leads to more stable policy updates.

---

## 6. TRPO vs. Vanilla Policy Gradient

| Aspect | Vanilla Policy Gradient | TRPO |
|--------|------------------------|------|
| **Update rule** | Fixed learning rate: $\theta + \alpha \hat{g}$ | Constrained: maximize surrogate s.t. KL $\leq \delta$ |
| **Step size control** | Manual tuning of $\alpha$ | Automatic via trust region |
| **Gradient type** | Euclidean gradient | Natural gradient ($H^{-1}g$) |
| **Stability** | Prone to catastrophic updates | Monotonic improvement guarantee (in theory) |
| **Computational cost** | Cheap (one gradient) | More expensive (CG + line search per update) |
| **Sensitivity to hyperparameters** | Very sensitive to $\alpha$ | Robust; $\delta$ is more interpretable |

### Why TRPO is a significant improvement

1. **Monotonic improvement guarantee.** The theoretical foundation of TRPO ([Kakade & Langford, 2002][kakade-langford]; [Schulman et al., 2015][trpo-paper]) shows that constraining the KL divergence provides a **lower bound** on the true policy improvement. Each update is guaranteed (in the exact case) to not make the policy worse.

2. **Invariance to parameterization.** The natural gradient is invariant to how the policy is parameterized. If you reparameterize the same policy (e.g., change the network architecture in a way that represents the same function), the natural gradient direction stays the same — whereas the vanilla gradient can change dramatically.

3. **Adaptive effective step size.** In regions where the policy is sensitive (small changes in $\theta$ cause large changes in behavior), TRPO automatically takes smaller steps. In flat regions, it takes larger steps. This is exactly the behavior we want.

---

## 7. Limitations of TRPO

Despite its theoretical elegance, TRPO has practical drawbacks that motivated the development of PPO:

- **Computational overhead.** The conjugate gradient computation and line search add significant per-iteration cost compared to a simple gradient step.
- **Implementation complexity.** Correctly implementing Hessian-vector products, conjugate gradient, and line search is non-trivial and error-prone.
- **Incompatibility with shared architectures.** When the policy and value function share parameters (common in practice), the KL constraint applies only to the policy head, making the optimization awkward.
- **Sample efficiency.** As an on-policy method, TRPO discards all collected data after each update. Off-policy methods (like SAC) can reuse old data.

These limitations are precisely what PPO addresses — see the [PPO tutorial](03_PPO.md) for details.

---

## 8. Connection to Actor-Critic Methods

TRPO is naturally an **actor-critic** algorithm:

- The **actor** is the policy $\pi_\theta$, updated via the constrained surrogate objective.
- The **critic** is the value function $V_\phi$, updated via regression on rewards-to-go.

The critic's role is to reduce variance in the policy gradient estimate by providing a baseline. Without a value function baseline, the policy gradient has high variance and learning is slow. The advantage estimate $\hat{A}_t = Q(s_t, a_t) - V(s_t)$ centers the gradient signal around zero, so that actions better than average get positive reinforcement and actions worse than average get negative reinforcement.

> *A dedicated tutorial on the actor-critic framework and its variants is planned for this series.*

---

## 9. Summary

TRPO's contribution is a **principled framework for stable policy optimization**:

1. It replaces the fragile fixed learning rate with a **trust region constraint** in policy space.
2. It uses the **natural gradient** (via conjugate gradient) to account for the geometry of probability distributions.
3. It provides **theoretical monotonic improvement guarantees** under the KL constraint.
4. It introduces the **surrogate objective** with importance sampling, which PPO later simplifies.

The key takeaway: **don't limit how far you move in parameter space — limit how much the policy's behavior changes.** This single insight underlies both TRPO and PPO, and is one of the most important ideas in modern deep RL.

---

## References

- **[Schulman et al., 2015]** — *Trust Region Policy Optimization.* ICML 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- **[Kakade & Langford, 2002]** — *Approximately Optimal Approximate Reinforcement Learning.* ICML 2002.
- **[Schulman et al., 2016]** — *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- **[OpenAI Spinning Up: TRPO]** — [spinningup.openai.com/en/latest/algorithms/trpo.html](https://spinningup.openai.com/en/latest/algorithms/trpo.html)

---

*Previous: [Policy Gradient Fundamentals](01_PG.md)* | *Next: [Proximal Policy Optimization (PPO)](03_PPO.md) — how to get TRPO-level stability with a simple clipped objective.*

[trpo-paper]: https://arxiv.org/abs/1502.05477
[kakade-langford]: https://dl.acm.org/doi/10.5555/645531.656005
