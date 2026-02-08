# Where Is the Policy Gradient in PPO?

**Unpacking the ratio form: why $\nabla_\theta$ seems to disappear**

> *Prerequisites: [Policy Gradient Fundamentals](01_PG.md), [PPO](03_PPO.md)*

---

## 1. The Puzzle

In [Chapter 01](01_PG.md), the policy gradient is written with an explicit gradient operator inside the expectation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s_t, a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

The gradient $\nabla_\theta$ is right there — loud and visible.

But in [Chapter 03](03_PPO.md), PPO writes down an **objective function** with no gradient in sight:

$$L(\theta) = \mathbb{E}_t \left[ r_t(\theta) \, \hat{A}_t \right], \quad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

Where did the gradient go? If we're doing gradient ascent, there must be a $\nabla_\theta$ somewhere. Is the ratio form hiding something?

**Yes — and this chapter shows exactly what it's hiding and why.**

---

## 2. Two Equivalent Views of Policy Gradient

The key insight is that there are two mathematically equivalent ways to describe the same algorithm.

### View A: Gradient-first (textbook REINFORCE)

Write the gradient formula directly:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

You never define a scalar loss function. You just have a gradient expression, and you follow it.

This is common in theory papers and early RL (Williams, 1992; Sutton et al., 2000).

### View B: Loss-first (modern deep learning style)

Define a scalar objective first:

$$L(\theta) = \mathbb{E}\left[ \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

Then take its gradient:

$$\nabla_\theta L(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

**Same gradient. Different presentation.** In View B, you hand the scalar loss $L(\theta)$ to an optimizer (Adam, SGD), and autograd computes $\nabla_\theta L$ for you. You never write the gradient formula explicitly.

PPO and TRPO live entirely in **View B**. That's why the gradient seems to disappear — it's handled by the autodiff framework, not written in the math.

---

## 3. Why the Ratio Appears: Importance Sampling

Now comes the subtle step that introduces the ratio.

In View B, the on-policy loss is:

$$L^{\text{on-policy}}(\theta) = \mathbb{E}_{a_t \sim \pi_\theta}\left[ \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

But there's a problem: **we collected data using the old policy** $\pi_{\theta_k}$, not the current policy $\pi_\theta$. The expectation is over the wrong distribution.

**Importance sampling** fixes this mismatch. For any function $f$:

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_k}}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} \, f(a) \right]$$

The fraction $\frac{\pi_\theta}{\pi_{\theta_k}}$ reweights samples from the old distribution to be valid under the new distribution. This fraction is exactly the **probability ratio** $r_t(\theta)$.

Applying this to the policy objective gives the **surrogate objective**:

$$L(\theta) = \mathbb{E}_{a_t \sim \pi_{\theta_k}}\left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \; \hat{A}_t \right] = \mathbb{E}_t\left[ r_t(\theta) \; \hat{A}_t \right]$$

This is the objective that TRPO and PPO optimize. It looks gradient-free, but it isn't — the gradient is hiding inside $r_t(\theta)$.

---

## 4. Where the Gradient Is Hiding: The Derivation

Let's take $\nabla_\theta$ of the surrogate objective step by step.

### Step 1: Move the gradient inside the expectation

Since the expectation is over data from $\pi_{\theta_k}$ (which is **fixed** — it doesn't depend on $\theta$), we can move the gradient inside:

$$\nabla_\theta L(\theta) = \nabla_\theta \, \mathbb{E}_t\left[ r_t(\theta) \, \hat{A}_t \right] = \mathbb{E}_t\left[ \nabla_\theta \, r_t(\theta) \; \hat{A}_t \right]$$

Note: this step would **not** be valid if the expectation were over $\pi_\theta$ (because then the distribution itself depends on $\theta$). The importance sampling rewrite is what makes this legal.

### Step 2: Differentiate the ratio

Now we need $\nabla_\theta \, r_t(\theta)$. Recall:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

The denominator $\pi_{\theta_k}(a_t \mid s_t)$ is a **constant** (it was computed using the old parameters $\theta_k$, which are frozen). So:

$$\nabla_\theta \, r_t(\theta) = \frac{\nabla_\theta \, \pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$$

### Step 3: Apply the log-derivative trick

The standard identity $\nabla_\theta \, \pi_\theta = \pi_\theta \, \nabla_\theta \log \pi_\theta$ (the same "log trick" used to derive the original policy gradient) gives:

$$\nabla_\theta \, \pi_\theta(a_t \mid s_t) = \pi_\theta(a_t \mid s_t) \; \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Substituting:

$$\nabla_\theta \, r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \; \nabla_\theta \log \pi_\theta(a_t \mid s_t) = r_t(\theta) \; \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

### Step 4: Assemble the full gradient

Plugging back into the expectation:

$$\boxed{\nabla_\theta L(\theta) = \mathbb{E}_t\left[ r_t(\theta) \; \nabla_\theta \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]}$$

**There it is.** The score function $\nabla_\theta \log \pi_\theta$ was hiding inside the ratio all along.

### Step 5: Verify consistency with vanilla policy gradient

At $\theta = \theta_k$ (the beginning of each PPO iteration), the ratio equals 1:

$$r_t(\theta_k) = \frac{\pi_{\theta_k}(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} = 1$$

So the gradient reduces to:

$$\nabla_\theta L(\theta)\Big|_{\theta = \theta_k} = \mathbb{E}_t\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

This is **exactly the vanilla policy gradient**. The surrogate objective and the true objective have the same gradient at the current policy — they are first-order equivalent.

As $\theta$ moves away from $\theta_k$, the ratio $r_t(\theta)$ deviates from 1 and **reweights** the gradient: actions that became more likely under the new policy get amplified, and actions that became less likely get dampened. This is the importance sampling correction at work.

---

## 5. What Clipping Does to the Gradient

In PPO-Clip, the objective is:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\Big( r_t(\theta) \, \hat{A}_t, \;\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \hat{A}_t \Big) \right]$$

What happens to the gradient?

**Inside the trust region** ($1-\epsilon \leq r_t \leq 1+\epsilon$): the clip is inactive, so the gradient is exactly as derived above:

$$\nabla_\theta L^{\text{CLIP}} = \mathbb{E}_t\left[ r_t(\theta) \; \nabla_\theta \log \pi_\theta(a_t \mid s_t) \; \hat{A}_t \right]$$

**Outside the trust region** ($r_t > 1+\epsilon$ or $r_t < 1-\epsilon$): the clipped term becomes a constant times $\hat{A}_t$, so its gradient with respect to $\theta$ is **zero**. The $\min$ selects this zero-gradient term, effectively **killing the gradient** for that sample.

This is how clipping enforces the trust region: it doesn't add a penalty — it simply **turns off the learning signal** for actions where the policy has already changed too much. The optimizer receives zero gradient and stops moving in that direction.

---

## 6. The Gradient Flow in Practice

In a modern deep learning framework (PyTorch, JAX), the entire process is:

1. **Forward pass**: compute $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$ and the clipped objective $L^{\text{CLIP}}$
2. **Backward pass**: autograd computes $\nabla_\theta L^{\text{CLIP}}$ automatically via the chain rule through $r_t(\theta)$, through $\pi_\theta(a_t \mid s_t)$, through the neural network
3. **Optimizer step**: Adam updates $\theta$

The programmer never writes $\nabla_\theta \log \pi_\theta$ explicitly. It emerges from autodiff applied to the ratio. This is why PPO code looks like a standard supervised learning loop — define a loss, call `loss.backward()`, call `optimizer.step()`.

---

## 7. Summary

### The answer to the puzzle

The policy gradient $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is not missing from PPO — it is **inside the ratio** $r_t(\theta)$, extracted automatically by the chain rule when you differentiate:

$$\nabla_\theta \, r_t(\theta) = r_t(\theta) \; \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

### Why PPO is written as a loss, not a gradient

Modern deep learning works by defining scalar objectives and letting autograd handle differentiation. PPO embraces this paradigm: define $L^{\text{CLIP}}(\theta)$, let the framework compute $\nabla_\theta L^{\text{CLIP}}$, and update with Adam.

> **Policy gradient algorithms are not defined by their gradient formulas — they are defined by the objective whose gradient they follow.**

### The progression

| Algorithm | Objective | Gradient | Trust region |
|-----------|-----------|----------|--------------|
| **Vanilla PG** | $\mathbb{E}[\log \pi_\theta \, \hat{A}_t]$ | Explicit $\nabla_\theta \log \pi_\theta$ | None |
| **TRPO** | $\mathbb{E}[r_t(\theta) \, \hat{A}_t]$ | Hidden in $r_t(\theta)$ | KL constraint |
| **PPO** | $\mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(\ldots))]$ | Hidden in $r_t(\theta)$, killed by clip | Soft (clipping) |

In all three cases, the same score function $\nabla_\theta \log \pi_\theta$ drives the update. The difference is how aggressively the update is constrained.

---

*Previous: [Proximal Policy Optimization (PPO)](03_PPO.md)* | *Next: [PPO Variants and Modern Descendants](04_PPO_variants.md)*
