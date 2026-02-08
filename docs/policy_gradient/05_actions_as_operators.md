# Actions as Operators: From Flat Vectors to Structured Control

**Why parameterizing the distribution over actions is not enough — and what happens when we parameterize the actions themselves**

> *Prerequisites: [Parameterizing the Policy](01a_parameterizing_policy.md), [PPO Variants](04_PPO_variants.md)*
>
> *Bridge to: [GRL Core Concepts](../GRL0/tutorials/01-core-concepts.md), [Reinforcement Field](../GRL0/tutorials/04-reinforcement-field.md)*

---

## 1. The Gap in Standard Policy Gradient

Chapters 01–04 of this series developed the full policy gradient toolkit: REINFORCE, baselines, GAE, TRPO, PPO, GRPO, DPO. All of these methods optimize a parameterized policy:

$$\pi_\theta(a \mid s)$$

In [Chapter 01a](01a_parameterizing_policy.md), we saw how this distribution can be implemented — Gaussian, mixture, normalizing flow, trajectory-level. But notice what all these parameterizations have in common: **the action $a$ is always a flat vector**.

- A Gaussian policy outputs $a \in \mathbb{R}^d$ — a vector of torques, forces, or velocities
- A categorical policy outputs $a \in \{1, \ldots, K\}$ — an index
- Even a trajectory-level policy outputs $\tau \in \mathbb{R}^{H \times 2}$ — a sequence of waypoints, flattened

The policy gradient machinery treats $a$ as an opaque object. It only needs $\log \pi_\theta(a \mid s)$ and its gradient. It never asks *what $a$ means* — what physical effect it produces, how it composes with other actions, or what structure it carries.

This is both a strength (generality) and a limitation (missed structure).

---

## 2. What's Missing: Three Limitations of Flat Actions

### 2.1 No Notion of Similarity

In a standard continuous policy, actions $a$ and $a'$ are "similar" only if they are close in Euclidean distance: $\|a - a'\|$ is small. But Euclidean distance in action space often has no physical meaning.

**Example — robotic arm:**

- Action $a_1 = (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)$: apply torque only to shoulder
- Action $a_2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)$: apply torque only to wrist

These are equidistant in $\mathbb{R}^7$ from the zero action, but they produce completely different physical effects. The policy gradient treats them as equally "far" from each other — it has no notion of functional similarity.

### 2.2 No Compositionality

Real-world actions are often **composed** from simpler primitives:

- "Reach for the cup" = move arm to position + open gripper
- "Overtake" = accelerate + steer left + steer right + decelerate
- "Walk" = cyclic composition of stance, swing, push-off phases

A flat action vector $a \in \mathbb{R}^d$ cannot represent this compositional structure. Every action is independent — there is no way to say "this action is a combination of these two simpler actions."

### 2.3 No Physical Semantics

A torque vector $\tau \in \mathbb{R}^7$ is not just a number — it is a **physical operator** that acts on the robot's state through the equations of motion:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

The action $\tau$ has meaning only in the context of the dynamics. A policy that outputs $\tau$ without knowing the dynamics is flying blind — it must learn the entire mapping from torques to outcomes through trial and error.

---

## 3. Actions as Operators: The Key Idea

What if, instead of treating actions as flat vectors, we treat them as **parameterized operators** that act on the environment state?

$$\theta_a \;\longrightarrow\; \hat{O}(\theta_a)$$

where:

- $\theta_a \in \Theta$ is the **action parameter vector** (the "knobs" the agent controls)
- $\hat{O}(\theta_a)$ is the **operator** specified by those parameters (the physical effect)

The distinction is subtle but important:

| Flat action | Action as operator |
|-------------|-------------------|
| $a \in \mathbb{R}^d$ | $\theta_a \in \Theta \to \hat{O}(\theta_a)$ |
| Opaque vector | Parameterized physical effect |
| Similarity = Euclidean distance | Similarity = effect similarity |
| No internal structure | Composable, interpretable |
| Policy learns mapping blindly | Structure aids generalization |

### 3.1 Concrete Examples

| Domain | Parameters $\theta_a$ | Operator $\hat{O}(\theta_a)$ | What structure buys you |
|--------|----------------------|------------------------------|------------------------|
| 2D navigation | $(F, \alpha)$ | Force of magnitude $F$ at angle $\alpha$ | Nearby angles → similar trajectories |
| Robotic arm | $(\mathbf{p}_{\text{target}}, \text{stiffness})$ | Impedance controller to target pose | Smooth interpolation between targets |
| Autonomous driving | $(v_{\text{ref}}, \kappa)$ | Track reference velocity $v_{\text{ref}}$ along curvature $\kappa$ | Curvature space is geometrically meaningful |
| Manipulation | $(\text{grasp\_type}, \mathbf{p}, \phi)$ | Grasp object at position $\mathbf{p}$ with orientation $\phi$ | Discrete grasp type + continuous placement |

In each case, the parameters $\theta_a$ have **physical meaning**, and the operator $\hat{O}(\theta_a)$ specifies a well-defined physical effect. Nearby parameters produce similar effects — not because of Euclidean distance, but because of the operator's structure.

### 3.2 The Operator Hierarchy

Actions can be organized into a hierarchy of abstraction:

**Level 0 — Raw actuator commands:**

$$a = \tau \in \mathbb{R}^d \quad \text{(joint torques, motor voltages)}$$

This is what standard policy gradient operates on. No structure, no semantics.

**Level 1 — Parameterized controllers:**

$$\theta_a = (\mathbf{p}_{\text{target}}, K_p, K_d) \quad \longrightarrow \quad \hat{O}(\theta_a) = K_p(\mathbf{p}_{\text{target}} - \mathbf{p}) + K_d(\dot{\mathbf{p}}_{\text{target}} - \dot{\mathbf{p}})$$

The agent chooses *where to go* and *how stiffly*, not individual torques. The PD controller handles the low-level execution.

**Level 2 — Parameterized skills:**

$$\theta_a = (\text{skill\_id}, \text{duration}, \text{target}) \quad \longrightarrow \quad \hat{O}(\theta_a) = \text{execute skill for duration toward target}$$

The agent composes pre-learned skills (reach, grasp, place) with continuous parameters (where, how long).

**Level 3 — Parameterized plans:**

$$\theta_a = (\text{waypoints}, \text{timing}, \text{constraints}) \quad \longrightarrow \quad \hat{O}(\theta_a) = \text{trajectory satisfying constraints}$$

The agent specifies a full motion plan. The operator is a trajectory optimizer or motion planner.

Each level adds structure and reduces the dimensionality of what the policy must learn, at the cost of reduced flexibility.

---

## 4. How This Changes Policy Gradient

### 4.1 Policy Over Operator Parameters

Instead of $\pi_\theta(a \mid s)$, we write:

$$\pi_\theta(\theta_a \mid s)$$

The policy outputs a distribution over **operator parameters**, not raw actions. Everything from Chapters 01–04 still applies — we still need $\log \pi_\theta(\theta_a \mid s)$, we still compute the ratio $r_t(\theta)$, we still clip.

But the gradient now flows through a more structured space. If the operator parameters have physical meaning, the policy gradient "knows" that changing the target position slightly will produce a slightly different trajectory — because the operator enforces this smoothness.

### 4.2 The Value Function Over Augmented Space

Here is where the connection to GRL becomes direct.

In standard RL, the value function is $Q(s, a)$ — value of taking flat action $a$ in state $s$.

If actions are parameterized operators, we can define:

$$Q^+(s, \theta_a) \quad \text{— value of executing operator } \hat{O}(\theta_a) \text{ in state } s$$

This is a function over the **augmented space** $\mathcal{Z} = \mathcal{S} \times \Theta$, where states and action parameters live together. This is exactly the construction in GRL ([Chapter 01: Core Concepts](../GRL0/tutorials/01-core-concepts.md)):

$$z = (s, \theta_a) \in \mathcal{Z}$$

The value function $Q^+(z)$ is smooth over this joint space, enabling:

- **Generalization across actions**: similar operator parameters → similar values
- **Gradient-based action selection**: $\theta_a^* = \arg\max_{\theta_a} Q^+(s, \theta_a)$ via gradient ascent
- **Kernel-based similarity**: the kernel $k(z, z')$ defines a meaningful notion of similarity between state-action pairs

### 4.3 Two Approaches to Optimization

With actions as operators, there are two complementary ways to find good actions:

**Approach A — Policy gradient (this series):**

Learn an explicit policy $\pi_\theta(\theta_a \mid s)$ and optimize it via PPO/GRPO:

$$\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(\theta_a \mid s) \; \hat{A}_t \right]$$

The policy is a neural network that maps states to distributions over operator parameters. Standard PG machinery applies unchanged.

**Approach B — Field navigation (GRL):**

Learn the value landscape $Q^+(s, \theta_a)$ and derive the policy implicitly:

$$\theta_a^* = \arg\max_{\theta_a} Q^+(s, \theta_a) \quad \text{or} \quad \theta_a \sim \exp\!\left(\frac{Q^+(s, \theta_a)}{\lambda}\right)$$

No explicit policy network. The "policy" is the act of navigating the reinforcement field — following the gradient $\nabla_{\theta_a} Q^+(s, \theta_a)$ toward high-value regions.

This is the central insight of GRL ([Chapter 04: Reinforcement Field](../GRL0/tutorials/04-reinforcement-field.md)):

> **The policy is not a function to be learned. It is a trajectory induced by the reinforcement field geometry.**

### 4.4 Can We Combine Both?

Yes — and this is a natural research direction. Consider an **actor-critic architecture** where:

- **Critic**: the reinforcement field $Q^+(s, \theta_a)$, represented non-parametrically via particles in RKHS (GRL's approach)
- **Actor**: an explicit policy $\pi_\phi(\theta_a \mid s)$, trained via policy gradient using the field as the critic

The actor provides efficient sampling (single forward pass), while the critic provides a smooth, non-parametric value landscape with kernel-induced generalization. The policy gradient update becomes:

$$\phi \leftarrow \phi + \beta \, \nabla_\phi \log \pi_\phi(\theta_a \mid s) \; \hat{A}_t$$

where the advantage $\hat{A}_t$ is computed from the reinforcement field:

$$\hat{A}_t = Q^+(s_t, \theta_{a,t}) - V(s_t), \quad V(s) = \mathbb{E}_{\theta_a \sim \pi_\phi}[Q^+(s, \theta_a)]$$

This is already sketched in [GRL Chapter 07a: Continuous Policy Inference](../GRL0/tutorials/07a-continuous-policy-inference.md) as "Actor-Critic in RKHS."

---

## 5. From Structured Actions to Structured Policies

### 5.1 Factorized Operators

When the operator has internal structure, the policy can mirror that structure:

$$\hat{O}(\theta_a) = \hat{O}_{\text{move}}(\theta_{\text{move}}) \circ \hat{O}_{\text{grasp}}(\theta_{\text{grasp}})$$

The policy factorizes accordingly:

$$\pi_\theta(\theta_a \mid s) = \pi_{\text{move}}(\theta_{\text{move}} \mid s) \cdot \pi_{\text{grasp}}(\theta_{\text{grasp}} \mid s, \theta_{\text{move}})$$

This is a **conditional factorization** — the grasp parameters depend on where the arm moved. Policy gradient handles this naturally (log-probs add, gradients flow through the chain).

### 5.2 Hierarchical Operators

For temporally extended actions (options, skills), the operator includes a **duration** or **termination condition**:

$$\theta_a = (\text{skill\_id}, \theta_{\text{skill}}, T_{\text{max}})$$

The high-level policy selects which skill to execute and with what parameters. The low-level skill executes for up to $T_{\text{max}}$ steps. This is the **options framework** (Sutton et al., 1999) expressed in the language of parameterized operators.

Policy gradient at the high level uses the cumulative reward over the skill's execution as the return signal. GRPO's group-normalized advantages are particularly natural here — sample multiple skill parameterizations for the same state and rank them by outcome.

---

## 6. The Spectrum of Action Representations

Pulling together the entire series, we can now see a spectrum of how actions are represented in RL:

| Representation | Example | Policy gradient | Value function | Generalization |
|---------------|---------|----------------|---------------|----------------|
| Discrete symbols | $a \in \{1, \ldots, K\}$ | Categorical $\pi_\theta$ | $Q(s, a)$ table | None across actions |
| Flat continuous | $a \in \mathbb{R}^d$ | Gaussian $\pi_\theta$ | $Q(s, a)$ network | Euclidean only |
| Parameterized operators | $\theta_a \to \hat{O}(\theta_a)$ | $\pi_\theta(\theta_a \mid s)$ | $Q^+(s, \theta_a)$ in augmented space | Operator-induced |
| Field-navigated operators | $\theta_a \to \hat{O}(\theta_a)$ | Implicit (from field) | $Q^+(z)$ in RKHS | Kernel-induced |

Moving down the table, we gain more structure, more generalization, and more physical meaning — but also more design choices and more assumptions about the domain.

Standard policy gradient (Chapters 01–04) operates in the first two rows. GRL operates in the last two. This chapter is the bridge.

---

## 7. Practical Implications

### 7.1 When to Use Structured Actions

**Use flat actions when:**

- The action space is low-dimensional ($d \leq 10$)
- You have abundant training data (millions of environment steps)
- The task doesn't require compositional or hierarchical behavior
- You want maximum generality with minimal domain knowledge

**Use parameterized operators when:**

- The action space is high-dimensional or has known structure
- You have domain knowledge about what "good" actions look like (controllers, skills)
- The task requires compositional behavior (reach + grasp + place)
- Sample efficiency matters (structured actions reduce the search space)

### 7.2 The Design Choice

Choosing the operator parameterization is an engineering decision analogous to choosing the neural network architecture. Too little structure (flat actions) wastes data on learning what the domain already knows. Too much structure (rigid skill library) prevents the agent from discovering novel behaviors.

The sweet spot depends on the domain, the data budget, and the desired level of autonomy.

---

## 8. Summary

### What this chapter adds to the series

The policy gradient series (Chapters 01–04) developed the *optimization machinery* — how to compute gradients, stabilize updates, and scale to large models. This chapter adds the *representation question* — what the policy is optimizing *over*.

### The key insight

**Standard PG parameterizes the distribution over actions. GRL parameterizes the actions themselves.** These are complementary:

- Policy gradient provides the optimization algorithm (PPO, GRPO)
- Parametric actions provide the representation (operators, augmented space, kernel similarity)

Combining them — policy gradient over operator parameters, with a reinforcement field as the critic — is a natural synthesis that inherits the strengths of both.

### The series so far

| Chapter | Topic | Key contribution |
|---------|-------|-----------------|
| [01](01_PG.md) | Policy gradient fundamentals | The score function $\nabla_\theta \log \pi_\theta$ |
| [01a](01a_parameterizing_policy.md) | Parameterizing the policy | Concrete distribution families |
| [02](02_TRPO.md) | TRPO | KL-constrained trust regions |
| [03](03_PPO.md) | PPO | Clipped ratio for stability |
| [03a](03a_pg_in_ppo.md) | PG in PPO | Where $\nabla_\theta$ hides in the ratio |
| [04](04_PPO_variants.md) | PPO variants | GRPO, DPO, critic-free methods |
| **05** | **Actions as operators** | **From flat vectors to structured control; bridge to GRL** |

---

## References

- **[Masson et al., 2016]** — *Reinforcement Learning with Parameterized Actions.* AAAI 2016.
- **[Sutton et al., 1999]** — *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning.* Artificial Intelligence. (Options framework)
- **[Hausman et al., 2018]** — *Learning an Embedding Space for Transferable Robot Skills.* ICLR 2018.
- **[Dalal et al., 2021]** — *Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives.* CoRL 2021.
- **GRL Tutorials** — [Core Concepts](../GRL0/tutorials/01-core-concepts.md), [Reinforcement Field](../GRL0/tutorials/04-reinforcement-field.md), [Continuous Policy Inference](../GRL0/tutorials/07a-continuous-policy-inference.md)

---

*Previous: [PPO Variants and Modern Descendants](04_PPO_variants.md)* | *Bridge to: [GRL Tutorial Series](../GRL0/tutorials/README.md)*
