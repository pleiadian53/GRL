# GRPO Without a Critic: How Advantages Are Computed

**The mechanics of critic-free advantage estimation and why it works for LLMs**

> *Prerequisites: [PPO vs GRPO: Credit Assignment](04a_PPO_vs_GRPO_in_LLM.md)*

---

## 1. The Question

A common source of confusion: papers say "GRPO requires no value model," yet GRPO clearly produces advantage estimates that drive the policy gradient. If there is no critic, where do the advantages come from?

The answer is that GRPO replaces the **learned** value function with a **computed** baseline derived from the policy's own samples. This section details exactly how.

---

## 2. PPO's Advantage Pipeline (For Comparison)

PPO computes advantages through a multi-step pipeline:

$$\text{Trajectory} \xrightarrow{V_\psi} \text{TD errors } \delta_t \xrightarrow{\text{GAE}} \hat{A}_t$$

Step by step:

1. Collect trajectory $(s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
2. For each timestep, query the value network: $V_\psi(s_t)$
3. Compute TD errors: $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$
4. Compute GAE: $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$

This requires a trained $V_\psi$ — a neural network as large as the policy itself in the LLM setting.

---

## 3. GRPO's Advantage Pipeline

GRPO computes advantages through a fundamentally simpler pipeline:

$$\text{Group of responses} \xrightarrow{\text{reward model}} \text{scores } r_i \xrightarrow{\text{z-score}} \hat{A}_i$$

Step by step:

1. For prompt $q$, sample $G$ complete responses: $o_1, \ldots, o_G \sim \pi_{\theta_k}(\cdot \mid q)$
2. Score each response with the reward model: $r_1, \ldots, r_G$
3. Compute group statistics: $\mu = \frac{1}{G}\sum_i r_i$, $\;\sigma = \sqrt{\frac{1}{G}\sum_i (r_i - \mu)^2}$
4. Normalize: $\hat{A}_i = \frac{r_i - \mu}{\sigma}$
5. Apply $\hat{A}_i$ uniformly to all tokens in response $i$

No neural network is trained. No backpropagation through a critic. The baseline $\mu$ is the group mean — a simple statistic.

### 3.1 Why Z-Scoring?

The z-score normalization serves three purposes:

- **Zero mean**: the average advantage across the group is zero, so the gradient update is balanced (some responses are reinforced, others are suppressed)
- **Unit variance**: the gradient magnitude is stable regardless of the reward scale
- **Prompt-specific**: each group is conditioned on the same prompt, so the baseline is relevant to that specific context

---

## 4. The Spectrum of Critic-Free Baselines

GRPO's group mean is one of several critic-free baseline strategies. Each trades off bias, variance, and compute:

| Method | Baseline $b$ | Samples needed | Variance | Bias |
|--------|-------------|----------------|----------|------|
| **No baseline** | $b = 0$ | 1 | Very high | None |
| **ReMax** | $b = r(\bar{o})$ where $\bar{o}$ is greedy | 1 + 1 greedy | High | Low (greedy is deterministic) |
| **RLOO** | $b_i = \frac{1}{G-1}\sum_{j \neq i} r_j$ | $G$ | Moderate | None (unbiased) |
| **GRPO** | $b = \frac{1}{G}\sum_j r_j$ (z-scored) | $G$ | Low | Slight (includes $r_i$ in its own baseline) |
| **PPO (GAE)** | $b = V_\psi(s_t)$ (learned) | 1 + critic training | Lowest (when critic is good) | Depends on critic quality |

Key observations:

- **RLOO** is technically unbiased (the baseline for response $i$ excludes $r_i$), while GRPO's z-score includes $r_i$ in the mean, introducing a slight bias. In practice, for $G \geq 8$, this bias is negligible.
- **ReMax** is the cheapest (only one extra forward pass for the greedy response) but has higher variance.
- **PPO's GAE** has the lowest variance *when the critic is well-trained*, but this condition is rarely met for LLMs.

---

## 5. Computational Cost Comparison

For a model with $P$ parameters generating responses of average length $L$ tokens:

| Operation | PPO | GRPO |
|-----------|-----|------|
| Policy forward passes | $1$ per response | $G$ per prompt |
| Critic forward passes | $L$ per response (one per token) | $0$ |
| Critic backward passes | $L$ per response | $0$ |
| Reward model calls | $1$ per response | $G$ per prompt |
| Models in GPU memory | $4P$ (policy + ref + critic + reward) | $3P$ (policy + ref + reward) |

GRPO trades **more inference** (sampling $G$ responses) for **no critic training**. Since inference is parallelizable and critic training is sequential, this is often a favorable trade — especially at scale where the critic's memory footprint is the binding constraint.

---

## 6. When the Critic-Free Approach Breaks Down

GRPO's response-level advantages have known limitations:

### 6.1 Uniform Reward Distributions

If all $G$ responses receive similar rewards, the z-scored advantages are near zero and the gradient signal vanishes. This happens when:

- The task is too easy (all responses correct) or too hard (all incorrect)
- The reward model has low discriminative power

**Mitigation**: use prompts with moderate difficulty, or increase $G$ to capture more variance.

### 6.2 Long-Horizon Credit Assignment

For very long responses where early tokens have outsized impact on the final reward, response-level advantages cannot distinguish which part of the response was responsible. A correct first step followed by a wrong conclusion gets the same per-token advantage as a wrong first step followed by a lucky recovery.

**Mitigation**: use **process reward models** (PRMs) that score intermediate steps, providing denser reward signals that GRPO can use at the step level rather than the response level.

### 6.3 Multi-Objective Rewards

When the reward is a weighted combination of multiple criteria (helpfulness, safety, style), the z-score treats them as a single scalar. There is no mechanism to separately credit tokens that contributed to helpfulness vs. safety.

---

## 7. Summary

GRPO's advantage computation is remarkably simple:

$$\hat{A}_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}$$

This single equation replaces an entire neural network (the critic), its training loop, and its memory footprint. It works because:

1. The group mean is a **relevant, prompt-specific baseline**
2. The z-score provides **stable, scale-invariant gradients**
3. The probability ratio $r_{i,t}(\theta)$ provides **implicit token-level differentiation**
4. For verifiable tasks, the **binary reward signal is strong enough** that response-level credit suffices

The tradeoff is clear: GRPO sacrifices fine-grained temporal credit assignment for simplicity, stability, and memory efficiency. For the tasks where it is most commonly applied (math, code, factual reasoning), this tradeoff is overwhelmingly favorable.

---

## References

- **[Shao et al., 2024]** — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (GRPO)
- **[Ahmadian et al., 2024]** — *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.* [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) (RLOO)
- **[Li et al., 2023]** — *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models.* [arXiv:2310.10505](https://arxiv.org/abs/2310.10505) (ReMax)
- **[Lightman et al., 2023]** — *Let's Verify Step by Step.* [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) (Process reward models)

---

*Previous: [PPO vs GRPO: Credit Assignment](04a_PPO_vs_GRPO_in_LLM.md)* | *Back to: [PPO Variants](04_PPO_variants.md)*
