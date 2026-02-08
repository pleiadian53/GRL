# PPO Variants and Modern Descendants

**Where PPO breaks, why GRPO exists, and what "state of the art" means in 2026**

> *Prerequisites: [PPO](03_PPO.md), [Where Is the Policy Gradient in PPO?](03a_pg_in_ppo.md)*

---

## 1. Why PPO Needed Successors

PPO ([Chapter 03](03_PPO.md)) is remarkably robust for a first-order method. But as it was applied to increasingly demanding problems — large language models, high-dimensional robotics, multi-agent systems — several failure modes became clear.

### 1.1 The Four-Model Burden (RLHF)

In the RLHF pipeline for LLM alignment, PPO requires **four models in memory simultaneously**:

1. **Policy** $\pi_\theta$ — the language model being fine-tuned
2. **Reference policy** $\pi_{\text{ref}}$ — the pre-trained model (frozen, for KL penalty)
3. **Reward model** $r_\phi$ — trained from human preferences
4. **Value function** $V_\psi$ — the critic, needed for advantage estimation

For a 70B-parameter LLM, this means ~280B parameters in GPU memory. The computational cost is enormous, and the value function $V_\psi$ is notoriously hard to train well in the LLM setting (the "state" is a variable-length token sequence, and the reward is sparse — given only at the end of a full response).

### 1.2 Ratio Variance in High Dimensions

Recall the PPO ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

For an LLM generating a response of $T$ tokens, the full-sequence ratio is a **product** of per-token ratios:

$$r(\theta) = \prod_{t=1}^T \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

Products of many terms near 1.0 can still produce extreme values. If each per-token ratio deviates by $\pm 5\%$, the full-sequence ratio after 100 tokens can range from $0.006$ to $170$. Clipping helps, but it also **kills the gradient** for many samples (see [Chapter 03a, Section 5](03a_pg_in_ppo.md)), reducing sample efficiency.

### 1.3 Advantage Estimation Difficulty

GAE ([Chapter 01, Section 6](01_PG.md)) requires a good value function $V(s)$. In robotics, the state is a fixed-size vector and the value function is easy to learn. In LLM alignment:

- The "state" is a variable-length token sequence
- The reward is typically given only at the end of a full response (sparse)
- The value function must generalize across diverse prompts and response lengths

A poor value function produces noisy advantages, which produce noisy gradients, which produce unstable training. This is the single biggest practical complaint about PPO for RLHF.

---

## 2. GRPO: Group Relative Policy Optimization

**GRPO** (Shao et al., 2024) was developed at DeepSeek specifically to address the value function problem in LLM alignment. Its key insight: **replace the learned value function with a group-based baseline computed from the policy's own samples.**

### 2.1 The Core Idea

Instead of training a separate critic $V_\psi(s)$, GRPO:

1. For each prompt $q$, samples a **group** of $G$ responses: $\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_k}(\cdot \mid q)$
2. Scores each response with the reward model: $r_1, r_2, \ldots, r_G$
3. Computes a **group-normalized advantage** for each response:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

This is a simple z-score within the group. No value function needed.

### 2.2 The GRPO Objective

GRPO uses a PPO-style clipped objective, but with the group-normalized advantage:

$$L^{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}} \; \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\Big( r_{i,t}(\theta) \, \hat{A}_i, \;\; \text{clip}\big(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon\big) \, \hat{A}_i \Big) \right] - \beta \, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where:

- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_k}(o_{i,t} \mid q, o_{i,<t})}$ is the per-token ratio for the $i$-th response
- $\hat{A}_i$ is the **same** group-normalized advantage for all tokens in response $i$ (response-level, not token-level)
- $\beta$ controls the KL penalty against the reference policy $\pi_{\text{ref}}$
- $|o_i|$ is the length of response $i$ (the $1/|o_i|$ normalizes across different response lengths)

### 2.3 What GRPO Eliminates

| Component | PPO | GRPO |
|-----------|-----|------|
| Policy $\pi_\theta$ | Yes | Yes |
| Reference policy $\pi_{\text{ref}}$ | Yes (for KL) | Yes (for KL) |
| Reward model $r_\phi$ | Yes | Yes |
| Value function $V_\psi$ | **Yes** | **No** |
| Models in memory | 4 | 3 |

Removing the value function saves ~25% of GPU memory and eliminates the hardest training signal in the pipeline.

### 2.4 Why Group Normalization Works

The group-normalized advantage $\hat{A}_i$ is a **relative** ranking within the group. It doesn't estimate absolute value — it only says "response $i$ is better/worse than the group average for this prompt."

This works because:

- **Prompt-specific**: each group is conditioned on the same prompt, so the baseline is relevant
- **Self-normalizing**: the z-score has zero mean and unit variance by construction, which stabilizes gradients
- **No generalization needed**: unlike $V_\psi(s)$, which must generalize across all prompts, the group baseline is computed fresh for each prompt

The tradeoff: GRPO requires $G$ samples per prompt (typically $G = 4$ to $16$), which increases inference cost. But inference is cheaper than training a value function, especially for large models.

### 2.5 GRPO in Practice

GRPO was used to train **DeepSeek-R1** and its variants. Key practical details:

- **Group size**: $G = 16$ (larger groups give better baselines but cost more inference)
- **KL penalty**: $\beta$ is annealed during training
- **Token-level vs. response-level**: the advantage $\hat{A}_i$ is response-level (same for all tokens in a response), which is simpler but coarser than PPO's token-level GAE advantages
- **Outcome supervision**: GRPO naturally supports outcome-based rewards (e.g., "is the final answer correct?") without needing dense per-token rewards

---

## 3. DPO: Direct Preference Optimization

**DPO** (Rafailov et al., 2023) takes a more radical approach: **eliminate the reward model entirely** and optimize directly from preference data.

### 3.1 The Insight

The standard RLHF pipeline is:

$$\text{Preferences} \xrightarrow{\text{train}} \text{Reward Model} \xrightarrow{\text{PPO}} \text{Aligned Policy}$$

DPO collapses this into a single step:

$$\text{Preferences} \xrightarrow{\text{DPO}} \text{Aligned Policy}$$

The key mathematical insight: the optimal policy under a KL-constrained reward maximization objective has a closed-form relationship with the reward function.

### 3.2 The Derivation

The RLHF objective is:

$$\max_{\pi_\theta} \; \mathbb{E}_{q \sim \mathcal{D}, \, o \sim \pi_\theta(\cdot|q)} \big[ r(q, o) \big] - \beta \, D_{\mathrm{KL}}\big(\pi_\theta(\cdot|q) \| \pi_{\text{ref}}(\cdot|q)\big)$$

The optimal policy for this objective is (derivable via calculus of variations):

$$\pi^*(o \mid q) = \frac{1}{Z(q)} \pi_{\text{ref}}(o \mid q) \exp\!\left(\frac{r(q, o)}{\beta}\right)$$

where $Z(q)$ is a normalizing constant. Rearranging for the reward:

$$r(q, o) = \beta \log \frac{\pi^*(o \mid q)}{\pi_{\text{ref}}(o \mid q)} + \beta \log Z(q)$$

Now substitute this into the **Bradley-Terry preference model** $p(o_w \succ o_l \mid q) = \sigma(r(q, o_w) - r(q, o_l))$, where $\sigma$ is the sigmoid. The $Z(q)$ terms cancel:

$$p(o_w \succ o_l \mid q) = \sigma\!\left( \beta \log \frac{\pi^*(o_w \mid q)}{\pi_{\text{ref}}(o_w \mid q)} - \beta \log \frac{\pi^*(o_l \mid q)}{\pi_{\text{ref}}(o_l \mid q)} \right)$$

### 3.3 The DPO Loss

Replace $\pi^*$ with the learnable policy $\pi_\theta$ and maximize the log-likelihood of observed preferences:

$$L^{\text{DPO}}(\theta) = -\mathbb{E}_{(q, o_w, o_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(o_w \mid q)}{\pi_{\text{ref}}(o_w \mid q)} - \beta \log \frac{\pi_\theta(o_l \mid q)}{\pi_{\text{ref}}(o_l \mid q)} \right) \right]$$

This is a **supervised learning loss** — no RL loop, no sampling, no reward model, no value function.

### 3.4 What DPO Eliminates

| Component | PPO | GRPO | DPO |
|-----------|-----|------|-----|
| Policy $\pi_\theta$ | Yes | Yes | Yes |
| Reference policy $\pi_{\text{ref}}$ | Yes | Yes | Yes |
| Reward model $r_\phi$ | Yes | Yes | **No** |
| Value function $V_\psi$ | Yes | No | **No** |
| RL training loop | Yes | Yes | **No** |
| Models in memory | 4 | 3 | 2 |

### 3.5 DPO's Relationship to Policy Gradient

Despite looking like supervised learning, DPO is implicitly performing policy optimization. The gradient of the DPO loss is:

$$\nabla_\theta L^{\text{DPO}} = -\beta \, \mathbb{E} \Big[ \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{weight}} \Big( \nabla_\theta \log \pi_\theta(o_w \mid q) - \nabla_\theta \log \pi_\theta(o_l \mid q) \Big) \Big]$$

where $\hat{r}_i = \beta \log \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{ref}}(o_i \mid q)}$ is the implicit reward.

The structure is recognizable: **score functions** $\nabla_\theta \log \pi_\theta$ weighted by an advantage-like term. DPO increases the probability of the preferred response and decreases the probability of the dispreferred one, with a weight that is larger when the model currently disagrees with the preference label.

### 3.6 Limitations of DPO

- **Offline only**: DPO trains on a fixed dataset of preferences. It cannot explore — if the preference data doesn't cover a behavior, DPO cannot learn it.
- **Distribution shift**: as $\pi_\theta$ moves away from the policy that generated the preference data, the implicit reward becomes unreliable.
- **No iterative improvement**: PPO and GRPO can generate new data and improve iteratively. DPO is a single-pass optimization.

These limitations have led to **online DPO variants** (OAIF, online DPO) that alternate between generating new responses and optimizing preferences — effectively re-introducing the RL loop that DPO was designed to eliminate.

---

## 4. Other Notable Variants

### 4.1 ReMax: Reinforce-Style Baseline for LLMs

**ReMax** (Li et al., 2023) takes a different approach to eliminating the value function. Instead of group normalization (GRPO), it uses a **REINFORCE-style baseline**:

$$\hat{A}(o) = r(o) - r(\bar{o})$$

where $\bar{o}$ is the **greedy response** (generated by argmax decoding from the current policy). This requires only one extra forward pass (no sampling a group), making it cheaper than GRPO.

### 4.2 RLOO: REINFORCE Leave-One-Out

**RLOO** (Ahmadian et al., 2024) uses a leave-one-out baseline: for each response $o_i$ in a group, the baseline is the mean reward of the *other* responses:

$$\hat{A}_i = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j$$

This is an unbiased estimator with lower variance than a single-sample baseline (ReMax) but doesn't require a learned value function (like PPO). It sits between ReMax and GRPO in the bias-variance tradeoff.

### 4.3 RPO: Relative Policy Optimization

**RPO** modifies the PPO objective to include a **contrastive term** that directly compares preferred and dispreferred responses within the clipped objective. It can be seen as a hybrid of PPO's ratio-based update and DPO's preference-based signal.

### 4.4 KTO: Kahneman-Tversky Optimization

**KTO** (Ethayarajh et al., 2024) is inspired by prospect theory. Instead of requiring paired preferences $(o_w, o_l)$, KTO works with **unpaired** binary feedback (good/bad labels on individual responses). The loss function is asymmetric, reflecting the psychological finding that humans weight losses more heavily than gains:

$$L^{\text{KTO}}(\theta) = \mathbb{E}_{o \sim \mathcal{D}} \Big[ w(o) \cdot \big(1 - \sigma(r_\theta(o) - z_{\text{ref}})\big) \Big]$$

where $w(o)$ is a label-dependent weight (higher for "bad" examples) and $z_{\text{ref}}$ is a reference point.

### 4.5 RLHF-Era PPO Tweaks

Several practical modifications to PPO have emerged from the RLHF community:

- **Reward whitening**: normalize rewards across the batch (similar in spirit to GRPO's group normalization, but applied to PPO)
- **Reward clipping**: cap extreme reward model outputs to prevent ratio explosion
- **KL penalty scheduling**: start with a high $\beta$ (stay close to reference) and anneal down
- **Response-level vs. token-level KL**: penalizing KL at the response level is more stable than per-token KL for long sequences
- **Separate reference model**: keep $\pi_{\text{ref}}$ frozen throughout training (don't update it periodically)

---

## 5. The Landscape: A Comparative View

### 5.1 What Each Method Optimizes

All these methods ultimately optimize the same underlying objective — maximize reward while staying close to a reference policy — but they differ in *how* they decompose the problem:

| Method | Reward Signal | Baseline / Advantage | Trust Region | Training Loop |
|--------|--------------|---------------------|--------------|---------------|
| **PPO** | Reward model | Learned $V_\psi$ (GAE) | Clipping | Online RL |
| **GRPO** | Reward model | Group mean (z-score) | Clipping + KL | Online RL |
| **ReMax** | Reward model | Greedy response | Clipping | Online RL |
| **RLOO** | Reward model | Leave-one-out mean | REINFORCE-style | Online RL |
| **DPO** | Implicit (from preferences) | Implicit (in loss) | KL (implicit) | Supervised |
| **KTO** | Binary feedback | Reference point | KL (implicit) | Supervised |

### 5.2 The Fundamental Tradeoff

There is a spectrum from **more RL** to **more supervised**:

```text
PPO ← GRPO ← RLOO ← ReMax ← Online DPO ← DPO ← KTO
 │                                                    │
 │  More exploration, more compute, more stable        │
 │  Better for iterative improvement                   │
 │                                                    │
 │                    Less compute, simpler pipeline   │
 │                    But offline, no exploration      │
```

Methods on the left (PPO, GRPO) can **explore** — they generate new responses, evaluate them, and improve. Methods on the right (DPO, KTO) are **offline** — they optimize a fixed dataset.

For frontier LLM training (GPT-4, Claude, Gemini, DeepSeek), the trend is toward **online methods with cheap baselines** (GRPO, RLOO) — they retain the exploration benefits of RL while eliminating the value function bottleneck.

---

## 6. Where PPO's Ratio Still Hides the Gradient

A unifying observation across all these variants (except pure DPO/KTO): the **probability ratio** $r_t(\theta) = \pi_\theta / \pi_{\theta_k}$ remains the core mechanism. As shown in [Chapter 03a](03a_pg_in_ppo.md), differentiating this ratio recovers the score function:

$$\nabla_\theta \, r_t(\theta) = r_t(\theta) \; \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

GRPO, ReMax, and RLOO all use this same ratio — they differ only in how they estimate the advantage $\hat{A}_t$. The gradient machinery is identical to PPO's.

DPO is the exception: it replaces the ratio with a **log-ratio difference** between preferred and dispreferred responses. But even there, the gradient contains $\nabla_\theta \log \pi_\theta$ — the same score function that drives all policy gradient methods.

**The score function $\nabla_\theta \log \pi_\theta$ is the universal engine.** What varies across methods is the signal (advantage, preference, binary feedback) that multiplies it.

---

## 7. Open Questions and Future Directions

### 7.1 Beyond Scalar Rewards

All current methods assume a scalar reward. But human preferences are multidimensional (helpfulness, harmlessness, honesty, style, ...). **Multi-objective RLHF** is an active research area — how to balance competing objectives without collapsing them into a single number.

### 7.2 Process Supervision vs. Outcome Supervision

PPO with GAE provides **token-level** (process) supervision. GRPO provides **response-level** (outcome) supervision. Which is better? Process supervision gives denser signal but requires a harder-to-train value function. Outcome supervision is simpler but coarser. The optimal granularity likely depends on the task.

### 7.3 Scaling Laws for RL Fine-Tuning

We have scaling laws for pre-training (Chinchilla, etc.) but not for RL fine-tuning. How does the optimal group size $G$ in GRPO scale with model size? How many RL steps are needed? These questions are largely unanswered.

### 7.4 Connection to Control and Robotics

The RLHF variants (GRPO, DPO, etc.) were developed for LLMs, but their ideas apply to any policy gradient setting. Group-normalized advantages could replace GAE in robotics. DPO-style preference learning could train robot policies from human demonstrations without reward engineering. This cross-pollination is just beginning.

---

## 8. Summary

### The evolution

| Era | Method | Key Innovation |
|-----|--------|---------------|
| 2017 | **PPO** | Clipped ratio for stable updates |
| 2023 | **DPO** | Eliminate reward model; train from preferences directly |
| 2024 | **GRPO** | Eliminate value function; group-normalized advantages |
| 2024 | **KTO** | Work with unpaired binary feedback |
| 2024+ | **Online DPO, RLOO** | Hybrid: online exploration + cheap baselines |

### The pattern

Each successor removes a component that was hard to train or expensive to maintain:

- PPO → GRPO: remove the value function
- PPO → DPO: remove the reward model *and* the value function *and* the RL loop
- DPO → Online DPO: add back exploration (because offline-only is limiting)

The field is converging toward methods that are **online** (can explore and improve iteratively) but **critic-free** (don't need a learned value function). GRPO is currently the best representative of this sweet spot.

### What hasn't changed

Across all variants, the fundamental mechanism is the same: **compute a score function** $\nabla_\theta \log \pi_\theta$, **multiply by an advantage signal**, and **constrain the update** to prevent catastrophic policy changes. The score function is the invariant; everything else is engineering.

---

## References

- **[Schulman et al., 2017]** — *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) (PPO)
- **[Rafailov et al., 2023]** — *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) (DPO)
- **[Shao et al., 2024]** — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (GRPO)
- **[Ahmadian et al., 2024]** — *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.* [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) (RLOO)
- **[Li et al., 2023]** — *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models.* [arXiv:2310.10505](https://arxiv.org/abs/2310.10505) (ReMax)
- **[Ethayarajh et al., 2024]** — *KTO: Model Alignment as Prospect Theoretic Optimization.* [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) (KTO)
- **[Zheng et al., 2023]** — *Secrets of RLHF in Large Language Models Part I: PPO.* [arXiv:2307.04964](https://arxiv.org/abs/2307.04964)

---

*Previous: [Proximal Policy Optimization (PPO)](03_PPO.md)* | *Next: [Actions as Operators](05_actions_as_operators.md)*

*Supplements: [PPO vs GRPO in LLMs](04a_PPO_vs_GRPO_in_LLM.md) | [GRPO Without a Critic](04b_GRPO_QA.md)*