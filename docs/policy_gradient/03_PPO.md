# Proximal Policy Optimization (PPO)

**TRPO's stability with first-order simplicity**

> *Prerequisites: [Policy Gradient Fundamentals](01_PG.md), [TRPO](02_TRPO.md)*

---

## 1. Introduction

Proximal Policy Optimization (PPO) was introduced by [Schulman et al. (2017)][ppo-paper] as a simpler alternative to TRPO that achieves comparable or better performance. It has since become the **default policy gradient algorithm** in deep RL, powering everything from game-playing agents to RLHF for large language models.

PPO's core insight: replace TRPO's computationally expensive constrained optimization (conjugate gradient + line search) with a **clipped surrogate objective** that can be optimized with standard first-order methods (SGD, Adam). The clipping acts as a soft trust region — it doesn't guarantee the KL constraint is satisfied, but in practice it provides similar stability.

---

## 2. From TRPO to PPO: The Motivation

Recall from [Chapter 02](02_TRPO.md) that TRPO solves:

$$\max_\theta \; \mathbb{E}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} \hat{A}_t \right] \quad \text{subject to} \quad \bar{D}_{\mathrm{KL}}(\pi_{\theta_k} \| \pi_\theta) \leq \delta$$

This works well but has practical drawbacks:

- **Second-order optimization**: requires computing the Fisher information matrix (via conjugate gradient)
- **Line search**: requires multiple forward passes to find a valid step size
- **Implementation complexity**: Hessian-vector products, CG iterations, and backtracking are non-trivial to implement correctly
- **Incompatible with shared architectures**: when policy and value function share a neural network backbone

PPO asks: **can we get the same stability with just gradient descent?**

---

## 3. The Probability Ratio

The central object in both TRPO and PPO is the **importance sampling ratio** (introduced in [Chapter 01, Section 7](01_PG.md)):

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

This ratio measures how much the new policy $\pi_\theta$ differs from the old policy $\pi_{\theta_k}$ for the specific action $a_t$ that was taken:

- $r_t(\theta) = 1$: new policy assigns the same probability as the old policy
- $r_t(\theta) > 1$: new policy is **more likely** to take this action
- $r_t(\theta) < 1$: new policy is **less likely** to take this action

The surrogate objective from TRPO can be written as:

$$L^{\text{CPI}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \, \hat{A}_t \right]$$

(CPI = conservative policy iteration.) Without any constraint, maximizing $L^{\text{CPI}}$ would lead to excessively large policy updates — the same destructive update problem from [Chapter 01, Section 5](01_PG.md).

---

## 4. The PPO-Clip Objective

PPO's key contribution is replacing the KL constraint with a **clipped objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\Big( r_t(\theta) \, \hat{A}_t, \;\; \text{clip}\big(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon\big) \, \hat{A}_t \Big) \right]$$

where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.2$).

### 4.1 What does the clipping do?

The clip function restricts the ratio to the interval $[1-\epsilon, \; 1+\epsilon]$:

$$\text{clip}(r, 1-\epsilon, 1+\epsilon) = \begin{cases} 1-\epsilon & \text{if } r < 1-\epsilon \\ r & \text{if } 1-\epsilon \leq r \leq 1+\epsilon \\ 1+\epsilon & \text{if } r > 1+\epsilon \end{cases}$$

With $\epsilon = 0.2$, the ratio is clamped to $[0.8, 1.2]$. The new policy cannot assign more than 20% more (or less) probability to any action before the clipping kicks in.

### 4.2 Why the minimum?

The $\min$ is the subtle and important part. It creates different behavior depending on the sign of the advantage:

**Case 1: $\hat{A}_t > 0$ (good action — we want to increase its probability)**

- The unclipped term $r_t \hat{A}_t$ rewards increasing $r_t$ (making the action more likely)
- The clipped term caps the reward at $(1+\epsilon) \hat{A}_t$
- The $\min$ ensures that once $r_t > 1+\epsilon$, there is **no further incentive** to increase the ratio

**Case 2: $\hat{A}_t < 0$ (bad action — we want to decrease its probability)**

- The unclipped term $r_t \hat{A}_t$ rewards decreasing $r_t$ (making the action less likely)
- The clipped term caps the reward at $(1-\epsilon) \hat{A}_t$
- The $\min$ ensures that once $r_t < 1-\epsilon$, there is **no further incentive** to decrease the ratio

In both cases, the clipping creates a **flat region** in the objective beyond the trust region boundary. The gradient becomes zero once the policy has moved "far enough," which naturally stops the optimizer from taking destructive steps.

### 4.3 The pessimistic bound

By taking the minimum of the clipped and unclipped objectives, PPO always uses the **more conservative** estimate. This creates a **lower bound** on the true policy improvement — the agent never gets credit for policy changes beyond the trust region. This is the same conservatism principle that underlies TRPO, implemented through a different mechanism.

---

## 5. The Full PPO Algorithm

### 5.1 The combined loss

In practice, PPO optimizes a combined objective that includes three terms:

$$L(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_1 \, L^{\text{VF}}(\phi) + c_2 \, S[\pi_\theta]$$

where:

- $L^{\text{CLIP}}(\theta)$: the clipped surrogate objective (Section 4)
- $L^{\text{VF}}(\phi) = \frac{1}{|\mathcal{D}_k| T} \sum_{\tau} \sum_t (V_\phi(s_t) - \hat{R}_t)^2$: value function loss (mean squared error)
- $S[\pi_\theta] = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$: entropy bonus (encourages exploration)
- $c_1, c_2$: weighting coefficients (typically $c_1 = 0.5$, $c_2 = 0.01$)

The entropy bonus prevents the policy from collapsing to a deterministic policy too early, which would stop exploration.

### 5.2 The algorithm

**Input:** Initial policy parameters $\theta_0$, initial value function parameters $\phi_0$

**Hyperparameters:** Clip range $\epsilon$, number of epochs $K$, minibatch size $M$, coefficients $c_1, c_2$

**For** $k = 0, 1, 2, \ldots$ **do:**

1. **Collect** trajectories $\mathcal{D}_k = \{\tau_i\}$ by running $\pi_{\theta_k}$ in the environment
2. **Compute** rewards-to-go $\hat{R}_t$
3. **Compute** advantage estimates $\hat{A}_t$ using GAE (see [Chapter 01, Section 6](01_PG.md))
4. **For** $e = 1, \ldots, K$ **epochs:**
   - Shuffle $\mathcal{D}_k$ into minibatches of size $M$
   - For each minibatch, compute and ascend the gradient of $L(\theta, \phi)$:

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta L^{\text{CLIP}}(\theta)$$

$$\phi \leftarrow \phi - \alpha \, \nabla_\phi L^{\text{VF}}(\phi)$$

### 5.3 Multiple epochs on the same data

A key practical feature of PPO: it performs **multiple gradient steps** (epochs) on the same batch of collected data. This is more sample-efficient than vanilla policy gradient or TRPO, which use each batch for a single update.

The clipping mechanism makes this safe — even after several gradient steps, the ratio $r_t(\theta)$ is prevented from drifting too far from 1, so the surrogate objective remains a reasonable approximation.

---

## 6. PPO-Penalty: The KL Alternative

PPO also proposed a second variant that uses an adaptive KL penalty instead of clipping:

$$L^{\text{KLPEN}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \, \hat{A}_t - \beta \, D_{\mathrm{KL}}(\pi_{\theta_k} \| \pi_\theta) \right]$$

where $\beta$ is adapted based on the observed KL divergence:

- If $D_{\mathrm{KL}} > 1.5 \, d_{\text{targ}}$: increase $\beta$ (penalize more — policy moved too far)
- If $D_{\mathrm{KL}} < d_{\text{targ}} / 1.5$: decrease $\beta$ (penalize less — policy is too conservative)

In practice, **PPO-Clip outperforms PPO-Penalty** on most benchmarks and is simpler to implement. PPO-Clip is what people mean when they say "PPO."

---

## 7. PPO vs. TRPO: A Comparison

| Aspect | TRPO | PPO-Clip |
|--------|------|----------|
| **Trust region mechanism** | Hard KL constraint | Soft clipping on ratio |
| **Optimization** | Second-order (CG + line search) | First-order (Adam) |
| **Implementation** | Complex (~500 lines) | Simple (~100 lines) |
| **Multiple epochs per batch** | No (single update) | Yes (typically 3-10 epochs) |
| **Shared policy-value network** | Awkward | Natural (combined loss) |
| **Theoretical guarantees** | Monotonic improvement (exact) | Empirical stability (no formal guarantee) |
| **Wall-clock speed** | Slower per iteration | Faster per iteration |
| **Performance** | Strong | Comparable or better |

### Why PPO won

PPO's dominance is not because it is theoretically superior — TRPO has stronger guarantees. PPO won because:

1. **Simplicity**: easy to implement, debug, and modify
2. **Compatibility**: works naturally with shared architectures, recurrent policies, and multi-task setups
3. **Efficiency**: multiple epochs per batch extract more learning per environment interaction
4. **Robustness**: less sensitive to hyperparameters than TRPO in practice

---

## 8. PPO in the RLHF Era

PPO became the dominant algorithm for **Reinforcement Learning from Human Feedback (RLHF)**, the technique used to align large language models (GPT, Claude, etc.):

1. **Pre-train** a language model on text data (supervised learning)
2. **Train a reward model** from human preference comparisons
3. **Fine-tune the LM with PPO** using the reward model as the environment

In this setting:

- The **policy** $\pi_\theta$ is the language model
- The **action** is the next token (or full response)
- The **reward** comes from the reward model
- The **KL penalty** (PPO-Penalty variant) is often added to prevent the LM from drifting too far from the pre-trained model

PPO's stability is critical here — a destructive update to a language model can cause it to produce gibberish, and recovery is expensive.

### Challenges of PPO for LLMs

Despite its success, PPO for RLHF has known difficulties:

- **High computational cost**: requires maintaining 4 models simultaneously (policy, reference policy, reward model, value function)
- **Reward hacking**: the policy can exploit weaknesses in the reward model
- **Sensitivity to hyperparameters**: the clip range, KL coefficient, and learning rate all interact in complex ways
- **Variance**: advantage estimation in the LLM setting can be noisy

These challenges have motivated alternatives like DPO (Direct Preference Optimization) and GRPO (Group Relative Policy Optimization), discussed in [Chapter 04](04_PPO_variants.md).

---

## 9. Practical Tips

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Clip range $\epsilon$ | 0.1 - 0.2 | 0.2 is the most common default |
| GAE $\lambda$ | 0.95 | Bias-variance tradeoff for advantages |
| Discount $\gamma$ | 0.99 | Standard for most tasks |
| Epochs per batch $K$ | 3 - 10 | More epochs = more sample-efficient but risk overfitting |
| Minibatch size $M$ | 64 - 4096 | Larger for more stable gradients |
| Entropy coefficient $c_2$ | 0.01 | Increase if policy collapses too early |
| Value loss coefficient $c_1$ | 0.5 | Standard |
| Learning rate | $3 \times 10^{-4}$ | Often with linear decay |

### Common pitfalls

- **Advantage normalization**: Always normalize advantages to zero mean and unit variance within each batch. This stabilizes training significantly.
- **Value function clipping**: Some implementations also clip the value function loss, though this is debated.
- **Learning rate schedule**: Linear decay to zero over training often helps.
- **Observation normalization**: Running mean/std normalization of observations improves stability.

---

## 10. Summary

### PPO in three sentences

PPO replaces TRPO's KL divergence constraint with a clipped surrogate objective that limits how much the policy can change per update. The clipping creates a flat region in the objective beyond the trust region, so the optimizer naturally stops before taking destructive steps. This achieves TRPO-level stability with first-order optimization, making it simple, fast, and the default choice for policy gradient RL.

### Key equations

**Probability ratio:**

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

**PPO-Clip objective:**

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\Big( r_t(\theta) \, \hat{A}_t, \;\; \text{clip}\big(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon\big) \, \hat{A}_t \Big) \right]$$

**Combined loss:**

$$L(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_1 \, L^{\text{VF}}(\phi) + c_2 \, S[\pi_\theta]$$

---

## References

- **[Schulman et al., 2017]** — *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) (PPO)
- **[Schulman et al., 2015]** — *Trust Region Policy Optimization.* ICML 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) (TRPO)
- **[Schulman et al., 2016]** — *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) (GAE)
- **[Zheng et al., 2023]** — *Secrets of RLHF in Large Language Models Part I: PPO.* [arXiv:2307.04964](https://arxiv.org/abs/2307.04964)
- **[OpenAI Spinning Up: PPO]** — [spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

---

*Previous: [Trust Region Policy Optimization (TRPO)](02_TRPO.md)* | *Next: [PPO Variants and Modern Descendants](04_PPO_variants.md)*

*Supplement: [Where Is the Policy Gradient in PPO?](03a_pg_in_ppo.md) — why $\nabla_\theta$ seems to disappear in the ratio form*

[ppo-paper]: https://arxiv.org/abs/1707.06347
