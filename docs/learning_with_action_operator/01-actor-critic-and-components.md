# Actor–Critic and Its Components

**Series:** Learning with Action Operators
**Part:** 1 of 3
**Prerequisites:** [Part 0: Overview](00-overview.md). Basic RL (states, actions, rewards).

This part defines the machinery that trains an operator policy — the **actor–critic** loop and every component it uses. Modern RL borrows a specific vocabulary (replay buffer, twin critics, target network, the `done` flag) that is rarely spelled out; we define each term from scratch and illustrate it on one running example.

**Running example — 1-D navigation.** A point agent lives on the line. Its state $s$ is its position; an operator moves it; it gets reward for approaching a goal at position $10$ and the episode ends when it arrives. We will reuse this throughout.

---

## 1. The big picture: two cooperating networks

Actor–critic training uses two neural networks:

- the **critic** $Q_\phi$ — learns *how good* a move is (a value estimate), with trainable weights $\phi$;
- the **actor** $\pi_\psi$ — the policy; learns to *pick* operators the critic rates highly, with trainable weights $\psi$.

They improve in tandem: the critic is fit to observed rewards; the actor is then adjusted to exploit the critic. The rest of this part is the bookkeeping that makes that stable.

---

## 2. The transition tuple $(s, \theta, r, s', \texttt{done})$

Every environment step produces one **transition** — a single record of experience:

| Symbol | Reads as | Meaning |
|---|---|---|
| $s$ | "state" | The state the agent was in. |
| $\theta$ | "theta" | The operator parameters it chose (the "action"). |
| $r$ | "reward" | The scalar reward received for the step. |
| $s'$ | "next state" | The state it moved to, $s' = \hat{O}_\theta(s) + \xi$. |
| $\texttt{done}$ | "done" | A 0/1 flag: did the episode end on this step? |

**The `done` flag.** Many tasks are *episodic* — they end (reach a goal, fail, or time out). `done = 1` marks the final transition of an episode (its $s'$ is a *terminal* state); otherwise `done = 0`. Its sole purpose is to switch off the "future" term of the value target at episode ends (Section 5). In a *continuing* task with no episodes, `done` is always 0.

*Navigation:* one step records $s$ = current position, $\theta$ = the chosen move, $r = -1$ (a small per-step cost) or $+1$ on reaching the goal, $s'$ = new position, and `done` $= 1$ only on the step that reaches position $10$.

> Implementation aside: some operator critics are written $Q(s, s')$ — conditioned on the *outcome* $s'$ rather than the parameters $\theta$ — because $s' = \hat{O}_\theta(s)$ already encodes the operator's effect. Then the stored tuple is just $(s, r, s', \texttt{done})$, with no separate $\theta$. Both conventions appear; we keep $\theta$ explicit here for clarity.

---

## 3. Reward, return, and the discount $\gamma$

A single step gives a scalar **reward** $r$. What the agent optimizes, though, is the **return** — the whole discounted stream of future rewards from a given time $t$ onward:

$$
G_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k \ge 0} \gamma^k\, r_{t+k}.
$$

| Symbol | Reads as | Meaning |
|---|---|---|
| $G_t$ | "return at time t" | Total discounted future reward from step $t$. |
| $\gamma$ | "gamma" | **Discount factor**, $\gamma \in [0, 1)$. A reward $k$ steps ahead counts for $\gamma^k$ of its value. |

The discount $\gamma$ does two jobs: it says "sooner is better," and (because $\gamma < 1$) it keeps the infinite sum finite.

---

## 4. Value functions: $V$ and $Q$

A **value** estimates expected return — but it always assumes *whose behavior you follow after the current step*. That assumption is written as a superscript and matters:

- **Value under a policy** $V^\pi(s)$ — expected return starting from $s$ and **following the current policy $\pi$ thereafter**. This is what an actor–critic's critic estimates (it evaluates the *current* policy).
- **Optimal value** $V^{*}(s)$ — expected return starting from $s$ and **acting optimally thereafter** (following the best possible policy $\pi^{*}$). It is the greedy maximum $V^{*}(s) = \max_\theta Q^{*}(s, \hat{O}_\theta(s))$.
- **Action value (critic)** $Q_\phi(s, \theta)$ — expected return that takes the move $\theta$ *now* and then follows $\pi$ (so it is a $Q^\pi$). "How good is this particular move?"

| Symbol | Reads as | Meaning |
|---|---|---|
| $V^\pi(s)$ | "V under pi" | Value of state $s$ following the current policy. |
| $V^{*}(s)$ | "V star" | Value of state $s$ acting optimally. |
| $\pi^{*}$ | "pi star" | The optimal policy (highest return from every state). |
| $Q_\phi(s,\theta)$ | "Q-sub-phi" | Critic estimate of the move's value; weights $\phi$. |

$V^\pi$ and $V^{*}$ coincide only at the end of learning, when the policy has become optimal ($\pi = \pi^{*}$). During training the critic tracks $V^\pi$ for the policy as it currently is, and the actor keeps improving $\pi$ — so $V^\pi$ climbs toward $V^{*}$. (This alternation — evaluate the current policy, then improve it — is called *generalized policy iteration*.)

---

## 5. The Bellman target and bootstrapping

The critic is trained by **regression**: for each stored transition we build a target value and push the critic's estimate toward it. The target is the **Bellman target**:

$$
y = r + \gamma\,(1 - \texttt{done})\,V(s').
$$

Read it as: *"the value of this move = the reward just received, plus the discounted value of where I ended up."*

- If `done = 0` (episode continues): $y = r + \gamma\,V(s')$ — the normal case.
- If `done = 1` (episode ended): the $(1 - \texttt{done})$ factor zeroes the future term, so $y = r$. You do **not** look past the end of an episode, because a terminal state has no future.

Using your own current estimate $V(s')$ inside the target is called **bootstrapping** — it lets value learning proceed step-by-step instead of waiting for whole episodes to finish. The mismatch $\delta = y - Q_\phi(s, \theta)$ is the **temporal-difference (TD) error** — "how wrong the estimate was"; training shrinks it.

| Symbol | Reads as | Meaning |
|---|---|---|
| $y$ | "the target" | Bellman regression target for the critic. |
| $\delta$ | "delta" | TD error, $\delta = y - Q_\phi(s,\theta)$. |

*Navigation, concretely.* Suppose $\gamma = 0.9$, the agent takes a step with reward $r = -1$, lands one step from the goal where it currently estimates $V(s') = 8$. Not terminal, so $y = -1 + 0.9 \times 8 = 6.2$. The critic's estimate for that move is pushed toward $6.2$. On the *final* step into the goal, with $r = +1$ and `done = 1`, the target is simply $y = +1$ — no future added.

---

## 6. The replay buffer $\mathcal{D}$

Rather than learn only from the latest step, the agent stores transitions in a **replay buffer** $\mathcal{D}$ (a large list) and trains on random **minibatches** $B$ drawn from it ($|B|$ = how many transitions in the batch). Two benefits: each experience is reused many times (sample-efficient), and random sampling breaks the strong correlation between consecutive steps (more stable regression). Because the stored data was generated by an older version of the policy, this is called **off-policy** learning.

| Symbol | Reads as | Meaning |
|---|---|---|
| $\mathcal{D}$ | "the buffer" | Replay buffer: stored past transitions. |
| $B,\ |B|$ | "minibatch", "batch size" | A random subset of $\mathcal{D}$ used for one update, and its size. |

---

## 7. Twin critics (reducing over-optimism)

Bootstrapping with a "best-case" estimate tends to **over-estimate** values: optimistic noise gets baked in and amplified. A standard remedy is to keep **two** critics, $Q_1$ and $Q_2$, with independent weights, and use the **smaller** of the two wherever a value is needed:

$$
V(s) \approx \min\big(Q_1(s, \pi(s)),\ Q_2(s, \pi(s))\big).
$$

Taking the minimum of two independently-trained estimates cancels much of the optimistic bias. It is a stability device, nothing more.

---

## 8. Stable targets: the target network and EMA

There is a subtlety in Section 5: the target $y$ contains a value $V(s')$ computed by the *same* critic we are training. If we used the live, changing weights, we would be chasing a moving goal — prediction and target shift together, which destabilizes learning.

The fix is a **target network**: a slowly-updated copy of the critic, with weights $\bar\phi$, used only to compute targets and never differentiated through (held as a fixed constant during the update — "stop-gradient"). It is kept close to the live critic by an **exponential moving average (EMA)**, also called a **Polyak average**:

$$
\bar\phi \leftarrow (1 - \tau)\,\bar\phi + \tau\,\phi, \qquad \tau \text{ small (e.g. } 0.005).
$$

| Symbol | Reads as | Meaning |
|---|---|---|
| $\bar\phi$ | "phi-bar" | Target-network weights — a slow copy of $\phi$. |
| $\tau$ | "tau" | EMA / Polyak rate; small $\tau$ = slower, more stable target. |

Each step, the target weights move a tiny fraction $\tau$ toward the live weights, so the target drifts slowly and learning has a stable goal to aim at. This same EMA-target idea recurs widely in machine learning — for example the target/teacher encoder in self-supervised methods (BYOL, DINO, JEPA) is an EMA of the online encoder — so it is worth recognizing as a general stabilization pattern.

---

## 9. Putting it together: one training iteration

With every term defined, the actor–critic loop reads cleanly:

1. **Collect.** Run the policy in the environment; at each step apply the chosen operator (with exploration noise $\xi$) and store $(s, \theta, r, s', \texttt{done})$ in $\mathcal{D}$.
2. **Evaluate (critic step).** Sample a minibatch from $\mathcal{D}$. Build targets $y = r + \gamma(1 - \texttt{done})\,V_{\bar\phi}(s')$ using the *target* network, then push the critic(s) toward $y$ by regression (twin-critic minimum). No gradient reaches the policy here.
3. **Improve (actor step).** Adjust the policy weights $\psi$ so the operators it picks lead to states the critic rates highly (the next part shows exactly how the gradient flows). Often plus a small "least-action" penalty preferring gentle operators.
4. **Stabilize.** Nudge the target network: $\bar\phi \leftarrow (1 - \tau)\bar\phi + \tau\phi$.

The critic learns *what is valuable*; the actor is pulled toward operators whose outcomes the critic values; the target network keeps the whole thing from chasing itself.

---

## 10. Symbol summary

| Symbol | Meaning |
|---|---|
| $s, s'$ | state, next state |
| $\theta$ | operator parameters (the action) |
| $r$ | reward; $G_t = \sum_k \gamma^k r_{t+k}$ the return |
| $\gamma$ | discount factor, $\in [0,1)$ |
| $\texttt{done}$ | episode-end flag (0/1) |
| $\pi_\psi,\ \psi$ | policy (actor) and its weights |
| $\pi^{*}$ | optimal policy |
| $Q_\phi,\ \phi$ | critic and its weights |
| $V^\pi(s),\ V^{*}(s)$ | state value under current vs optimal policy |
| $y,\ \delta$ | Bellman target; TD error $y - Q$ |
| $\mathcal{D},\ B,\ |B|$ | replay buffer; minibatch and its size |
| $Q_1, Q_2$ | twin critics |
| $\bar\phi,\ \tau$ | target-network weights; EMA rate |

---

**Next:** [Part 2: How gradients flow](02-how-gradients-flow.md) — how the actor step (Section 9.3) actually computes its update by differentiating through the operator pipeline.
