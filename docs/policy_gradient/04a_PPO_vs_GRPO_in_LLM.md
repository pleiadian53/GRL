# PPO vs GRPO: Credit Assignment in LLM Fine-Tuning

**How reward signals flow from response-level scores to token-level updates**

> *Prerequisites: [PPO](03_PPO.md), [PPO Variants](04_PPO_variants.md)*

---

## 1. The Credit Assignment Problem in LLMs

When fine-tuning a language model with RL, the fundamental challenge is **credit assignment**: a reward model scores the *entire response*, but the policy made a decision at *every token*. Which tokens were responsible for the good (or bad) outcome?

Consider a model generating a 200-token math solution. The reward is binary: 1 if the final answer is correct, 0 otherwise. The model needs to learn that token 47 (where it chose the right algebraic step) mattered far more than token 150 (where it wrote "therefore"). But the reward signal says nothing about individual tokens.

This is the **temporal credit assignment problem** — familiar from RL in general, but especially acute in LLMs because:

- Sequences are long (hundreds to thousands of tokens)
- Rewards are sparse (typically given only at the end)
- The action space is enormous (vocabulary size ~32K–128K per step)

PPO and GRPO handle this problem very differently.

---

## 2. PPO's Approach: Learned Value Function

In the standard RLHF pipeline, PPO trains a **value network** $V_\psi(s_t)$ that predicts the expected return from each token position $t$:

$$V_\psi(s_t) = \mathbb{E}\left[ R \mid \text{tokens generated so far} = (o_1, \ldots, o_t) \right]$$

The advantage at each token is then computed via GAE:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \, \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

In principle, this gives **token-level credit** — each token gets its own advantage. In practice, there are two problems:

**Problem 1 — The value function is hard to train.** The "state" $s_t$ is the entire token sequence so far. The value network must generalize across all possible prefixes of all possible responses to all possible prompts. This is a harder learning problem than the policy itself.

**Problem 2 — Sparse rewards degrade the signal.** When the reward is given only at the final token, the TD errors $\delta_t$ for intermediate tokens depend entirely on the value function's predictions. If $V_\psi$ is inaccurate (and it usually is early in training), the per-token advantages are noisy. Many implementations fall back to **response-level advantages** — assigning the same advantage to every token in the sequence:

$$\hat{A}_t \approx R - b \quad \text{for all } t$$

where $b$ is some baseline (e.g., the mean reward in the batch). This is simpler but throws away all temporal structure.

---

## 3. GRPO's Approach: Group-Normalized, No Critic

GRPO eliminates the value network entirely. For each prompt $q$, it samples a group of $G$ responses and computes a **response-level** advantage via z-scoring:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

This advantage $\hat{A}_i$ is the **same for all tokens** in response $i$. GRPO does not assign per-token credit through a value function.

### 3.1 Clarifying a Common Confusion

There is a widespread misconception that "GRPO uses token-level credit assignment." This confusion arises from two different meanings of "value model":

**Meaning A — A learned critic network $V_\psi(s_t)$:**

A separate neural network trained to predict values at each token position. This is what PPO uses. GRPO does **not** have this. This is what papers mean when they say "GRPO requires no value model."

**Meaning B — Any mechanism that produces per-token advantage signals:**

All RL algorithms need *some* way to weight the policy gradient at each token. GRPO applies the same response-level advantage $\hat{A}_i$ to every token in the response, but the PPO-style clipped ratio $r_{i,t}(\theta)$ is still computed **per token**:

$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_k}(o_{i,t} \mid q, o_{i,<t})}$$

So the *gradient signal* varies across tokens (because the ratio varies), even though the *advantage* is constant. This can look like "token-level credit" in practice, but the mechanism is fundamentally different from a learned critic.

### 3.2 The Correct Statement

| Concept | PPO | GRPO |
|---------|-----|------|
| Trainable value network $V_\psi$ | Yes | No |
| Advantage granularity | Token-level (via GAE) or response-level (fallback) | Response-level (group z-score) |
| Per-token gradient variation | Via ratio $\times$ advantage | Via ratio $\times$ (constant advantage) |
| Memory cost | High (critic = another LLM-sized model) | Low (no critic) |

**GRPO assigns response-level credit without training a value network. The per-token gradient variation comes from the probability ratio, not from a learned critic.**

---

## 4. Why GRPO Works Despite Coarser Credit

If GRPO's advantages are response-level (coarser than PPO's token-level GAE), why does it work so well?

### 4.1 Group Comparison Is a Strong Signal

For tasks with verifiable outcomes (math, code, factual QA), the reward is often binary: correct or incorrect. In this setting, the group comparison is highly informative:

- If 12 out of 16 responses are correct, the 4 incorrect ones get strongly negative advantages
- The model learns to avoid the *patterns* that distinguish incorrect responses from correct ones
- Over many prompts and groups, this signal is sufficient to identify which token-level behaviors matter

### 4.2 The Ratio Provides Implicit Token-Level Differentiation

Even with a constant advantage $\hat{A}_i$, the gradient at each token is:

$$\nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \cdot \hat{A}_i$$

Tokens where the policy is uncertain (high entropy) produce larger gradients than tokens where the policy is confident. This provides a natural form of token-level weighting without an explicit critic.

### 4.3 The Value Function Was the Bottleneck

In practice, PPO's value function for LLMs is often so poorly trained that its token-level advantages are noisier than GRPO's response-level advantages. A clean, unbiased response-level signal can outperform a noisy, biased token-level signal.

---

## 5. When Does Token-Level Credit Matter?

Response-level credit (GRPO) works well when:

- Rewards are sparse and binary (correct/incorrect)
- Responses are relatively short (< 500 tokens)
- The task has clear success/failure modes

Token-level credit (PPO with a good critic, or process reward models) becomes important when:

- Responses are very long (multi-step reasoning chains)
- Intermediate steps have independent quality (each step in a proof can be right or wrong)
- The task requires fine-grained behavioral shaping (tone, style, safety at specific points)

This is why **process reward models** (PRMs) — which score intermediate reasoning steps, not just the final answer — are an active research direction. They provide dense reward signals that can be used with either PPO or GRPO.

---

## 6. The Full Picture: Post-Training Pipeline

The modern LLM post-training pipeline typically follows:

$$\text{Pre-training} \to \text{SFT} \to \text{RL alignment}$$

where the RL alignment step uses one of:

| Method | Reward source | Credit assignment | Critic needed | Exploration |
|--------|--------------|-------------------|---------------|-------------|
| **PPO** | Reward model | Token-level (GAE) or response-level (fallback) | Yes | Online |
| **GRPO** | Reward model | Response-level (group z-score) | No | Online |
| **DPO** | Preference pairs | Implicit (in loss) | No | Offline |
| **RLOO** | Reward model | Response-level (leave-one-out) | No | Online |

GRPO has emerged as the preferred method for **reasoning tasks** (math, code) where rewards are verifiable and binary. PPO remains relevant for **open-ended generation** where dense, nuanced credit assignment matters. DPO is used when preference data is available but online generation is too expensive.

---

## 7. Summary

The core difference between PPO and GRPO in the LLM setting is not *whether* they assign token-level credit, but *how*:

- **PPO** trains a separate value network to estimate per-token advantages. This is powerful but expensive (4 models in memory) and fragile (the critic is hard to train for variable-length text).
- **GRPO** uses response-level group-normalized advantages with no critic. The per-token gradient variation comes from the probability ratio, not a learned value function. This is cheaper, simpler, and often more stable.

Both methods use the same PPO-style clipped objective and the same score function $\nabla_\theta \log \pi_\theta$ at their core. The difference is entirely in how the advantage signal is computed.

---

## References

- **[Shao et al., 2024]** — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (GRPO)
- **[Zheng et al., 2023]** — *Secrets of RLHF in Large Language Models Part I: PPO.* [arXiv:2307.04964](https://arxiv.org/abs/2307.04964)
- **[Lightman et al., 2023]** — *Let's Verify Step by Step.* [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) (Process reward models)

---

*Previous: [PPO Variants and Modern Descendants](04_PPO_variants.md)* | *See also: [Where Is the Policy Gradient in PPO?](03a_pg_in_ppo.md)*
