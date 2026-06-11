# How Gradients Flow

**Series:** Learning with Action Operators
**Part:** 2 of 3
**Prerequisites:** [Part 0: Overview](00-overview.md), [Part 1: Actor–critic and its components](01-actor-critic-and-components.md).

Part 1 ended with the actor step: *adjust the policy so it picks operators whose outcomes the critic rates highly.* This part shows exactly how that adjustment is computed — by differentiating through the operator pipeline. This is the heart of "learning with action operators," and it is what makes the operator view distinctive.

We keep the 1-D navigation example from Part 1 (a point agent moving toward a goal).

---

## 1. Why the pipeline is differentiable at all

In classical RL the next state comes from the environment's transition $s' = T(s, a)$, a black box. You cannot ask "how would the reward change if I nudged the action?" *through* the environment, because you do not have $\partial s' / \partial a$. (That is why classical policy-gradient methods use the sampling-based estimator in Section 4 instead.)

In the operator view the agent produces the next state with its **own** differentiable function — the generator $\Phi$:

$$
s' = \hat{O}_\theta(s) + \xi = \Phi(\theta, s) + \xi.
$$

Because $\Phi$ is a neural network the agent controls, the sensitivity $\partial s' / \partial \theta$ is available — it is ordinary backpropagation through $\Phi$. The transition has become a differentiable layer *inside* the agent. That single fact is what lets a learning signal flow all the way from the next state back into the policy.

---

## 2. The pathwise gradient (the main method)

Train the policy by **maximizing the critic's value of the state the policy moves to**, minus a small penalty preferring gentle operators. Writing the produced next state as $s'_\psi = \Phi(\pi_\psi(s), s)$, the actor's objective averaged over states is:

$$
J(\psi) = \operatorname{mean}_{s}\Big[\, Q_\phi\big(s,\ s'_\psi\big) \;-\; \lambda\, E(\hat{O}_\theta) \,\Big].
$$

| Symbol | Reads as | Meaning |
|---|---|---|
| $J(\psi)$ | "J of psi" | The actor's objective — what we increase by adjusting $\psi$. |
| $Q_\phi(s, s')$ | "Q-sub-phi" | The critic's value of being at $s$ and moving to $s'$ (Part 1). |
| $E(\hat{O}_\theta)$ | "energy of the operator" | A penalty measuring how far the operator is from the identity $I$ (the "do-nothing" map $I(s) = s$); e.g. $\lVert \hat{O}_\theta - I \rVert$. |
| $\lambda$ | "lambda" | Weight of that penalty, $\lambda \ge 0$ (the "least-action" preference). |

To improve the policy we ascend this objective: compute $\partial J / \partial \psi$ and step $\psi$ in that direction. The gradient is just the **chain rule** along the pipeline, $s' \leftarrow \theta \leftarrow \psi$:

$$
\frac{\partial J}{\partial \psi}
= \operatorname{mean}_{s}\Big[\,
\underbrace{J_\pi^{\top}}_{\partial \theta / \partial \psi}\;
\underbrace{J_\Phi^{\top}}_{\partial s' / \partial \theta}\;
\underbrace{\nabla_{s'} Q_\phi}_{\text{value gradient}}
\;-\; \lambda\, J_\pi^{\top}\, \nabla_\theta E
\,\Big].
$$

The three factors, each tied to one arrow of the pipeline:

| Factor | Reads as | What it is | Which arrow |
|---|---|---|---|
| $\nabla_{s'} Q_\phi$ | "grad of Q w.r.t. s-prime" | How the value rises as the next state moves — the direction in state space that is "better." | the critic scoring $s'$ |
| $J_\Phi$ | "Jacobian of Phi" | $\partial s' / \partial \theta$: how the next state changes as the operator parameters change. (A *Jacobian* is the matrix of partial derivatives of a vector output w.r.t. a vector input.) | $\theta \xrightarrow{\Phi} s'$ |
| $J_\pi$ | "Jacobian of pi" | $\partial \theta / \partial \psi$: how the parameters change as the policy weights change. | $s \xrightarrow{\pi_\psi} \theta$ |

(The superscript $\top$ is "transpose"; it just arranges the chain rule for a backward pass.) In words: the critic says which direction in next-state space is better ($\nabla_{s'}Q$); the generator's Jacobian translates that into "which way to move the operator parameters" ($J_\Phi$); the policy's Jacobian carries it the rest of the way into the weights ($J_\pi$). One backward pass through the same pipeline used in the forward direction — automatic-differentiation libraries compute it for you.

This is the operator-valued cousin of deterministic-policy-gradient methods (DDPG and its relatives): the actor outputs operator parameters instead of a raw action vector, and the differentiable transition is the agent's own generator.

---

## 3. A worked example

Take the simplest operator on the line — a **displacement** $\hat{O}_\theta(s) = s + \theta$ (move by $\theta$), so $\Phi(\theta, s) = s + \theta$. Then $\partial s' / \partial \theta = 1$, i.e. $J_\Phi = 1$. The actor gradient (ignoring the penalty for a moment) reduces to

$$
\frac{\partial J}{\partial \theta} = J_\Phi \cdot \nabla_{s'} Q_\phi = 1 \cdot \nabla_{s'} Q_\phi = \nabla_{s'} Q_\phi.
$$

So the move $\theta$ is pushed *in the direction that increases the critic's value*. Concretely: the agent is at position $s = 7$, goal at $10$; the critic has learned that higher positions (closer to the goal) are better, so $\nabla_{s'} Q > 0$ there. The gradient therefore increases $\theta$ — the policy learns to step rightward, toward the goal. The penalty term $-\lambda \nabla_\theta E$ pulls $\theta$ gently back toward $0$ (no movement), so the agent settles on the *smallest* step that still gains enough value.

For a richer operator the Jacobian is no longer $1$. Take an affine operator $\hat{O}_\theta(s) = \theta_1 s + \theta_2$ with $\theta = (\theta_1, \theta_2)$. Now $\partial s'/\partial \theta = (s,\ 1)$, so the gradient becomes $(\,s \cdot \nabla_{s'}Q,\ \nabla_{s'}Q\,)$ — the update to the *scaling* part $\theta_1$ is weighted by the current position $s$, while the *shift* part $\theta_2$ updates directly. That position-weighting is exactly $J_\Phi$ doing its job: routing the value gradient correctly to each parameter.

---

## 4. The alternative: the score-function gradient

The pathwise method needs a differentiable critic (it differentiates *through* $Q_\phi$). When you only have *samples* of return and cannot differentiate the reward — a more general but higher-variance situation — use the classic **score-function** estimator. For a stochastic policy $\pi_\psi(\theta \mid s)$ (one that outputs a distribution over $\theta$):

$$
\frac{\partial J}{\partial \psi} = \operatorname{mean}\Big[\, \nabla_\psi \log \pi_\psi(\theta \mid s)\,\big(G - b(s)\big) \,\Big],
$$

where $G$ is the observed return and $b(s)$ is a **baseline** (any state-dependent reference value subtracted to reduce variance without biasing the result). This needs only samples — no $\partial s'/\partial \theta$, no differentiable reward. It is more broadly applicable but noisier; prefer the pathwise method when a differentiable critic is available (which, in actor–critic, it is).

**A useful middle ground (for a stochastic policy with low variance).** Make the policy output a mean $\mu_\psi(s)$ and a spread $\sigma_\psi(s)$, and sample $\theta = \mu_\psi(s) + \sigma_\psi(s)\,\varepsilon$ with $\varepsilon$ a standard-normal random draw (zero mean, unit variance). Because $\theta$ is now a *differentiable function* of $\psi$ and a separate noise source $\varepsilon$, you can use the pathwise gradient of Section 2 *through* the sampled $\theta$ — keeping exploration while enjoying low variance. (This "reparameterization" also handles the transition noise $\xi$: since $\xi$ is added on, $\partial s'/\partial \theta$ does not depend on it, so gradients pass straight through.)

| Symbol | Reads as | Meaning |
|---|---|---|
| $G$ | "return" | Observed total discounted reward of a trajectory (Part 1). |
| $b(s)$ | "baseline" | A reference value subtracted to reduce variance; does not bias the gradient. |
| $\mu_\psi(s),\ \sigma_\psi(s)$ | "mu", "sigma" | Mean and spread of operator parameters a stochastic policy outputs at $s$. |
| $\varepsilon$ | "epsilon" | A standard-normal random draw used to sample $\theta$. |

---

## 5. What the pathwise gradient assumes

Two things are worth being explicit about, since they shape when the method applies.

**It is a faithful learning signal to the extent the generator models the real transition.** The gradient pulls $\partial s'/\partial \theta$ from the generator $\Phi$. That equals the true sensitivity of the next state to $\theta$ only when $s' = \Phi(\theta, s)$ is genuinely how the world evolves. When the operator family *is* the environment's dynamics, this holds by construction. When $\Phi$ is instead a *learned model* of the dynamics, the gradient is taken through the model, so it is only as good as the model — the same trade-off as any model-based method. It is a modeling assumption to state, not to hide.

**The generator's Jacobian appears because the critic is scored on the outcome $s'$.** If the critic is written $Q_\phi(s, \theta)$ — conditioned on the parameters directly — the actor gradient is simply $\nabla_\theta Q_\phi \cdot J_\pi$, with no transition Jacobian at all. Conditioning on $s'$ instead (as in Section 2) is what brings in $J_\Phi$, and it buys a pleasant reading: the value gradient $\nabla_{s'}Q$ is a *field over state space* that gets pulled back through the operator. Both choices give the same update; differentiating *through the transition* becomes essential only when you look more than one step ahead (planning through several operator applications) or backpropagate through a true differentiable reward rather than a learned critic.

---

## 6. Looking ahead: the same idea in representation space

The pathwise gradient is even more natural when the "transition" lives in a learned representation rather than raw state. In a joint-embedding predictive setup, an encoder maps observations to a latent code, and a **predictor** maps the current code to the next one — and that predictor is already a differentiable transition. Making the predictor *operator-conditioned* — the policy emits $\theta$, and the latent transition becomes $\hat{U}_\theta$ acting on codes — drops this entire mechanism into representation space, with the encoder learned by a predictive objective rather than assumed to be the true dynamics. That is the bridge from operator-RL to predictive representation learning, and a later part of this series develops it.

---

## 7. Symbol summary

| Symbol | Meaning |
|---|---|
| $J(\psi)$ | actor objective (critic value minus operator penalty) |
| $\nabla_{s'}Q_\phi$ | gradient of the critic w.r.t. the next state (the "better-direction" field) |
| $J_\Phi$ | Jacobian of the generator, $\partial s'/\partial \theta$ |
| $J_\pi$ | Jacobian of the policy, $\partial \theta/\partial \psi$ |
| $E(\hat{O}_\theta),\ \lambda,\ I$ | operator energy penalty, its weight, the identity operator |
| $G,\ b(s)$ | observed return; variance-reducing baseline |
| $\mu_\psi, \sigma_\psi, \varepsilon$ | mean/spread of a stochastic policy; standard-normal draw |

---

**One-line summary:** because the agent builds the transition with its own differentiable generator, improving the policy is a single backward pass — the critic's value gradient is routed through the generator's Jacobian and the policy's Jacobian into the policy weights — which is why operator policies can be trained by smooth gradient ascent rather than discrete search.

**Previous:** [Part 1: Actor–critic and its components](01-actor-critic-and-components.md) · **Series:** [Overview](00-overview.md)
