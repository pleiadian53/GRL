# GRL Theory: Actions as Operators

## Overview

Generalized Reinforcement Learning (GRL) reconceptualizes the notion of "action" in reinforcement learning. Instead of treating actions as discrete indices or fixed-dimensional vectors, GRL models actions as **functional operators** on the state space.

## The Core Insight

In classical RL, an agent selects an action $a \in \mathcal{A}$ which influences the next state through the transition dynamics $T(s' | s, a)$. The action is external to the state transformation.

In GRL, the action **is** the transformation:

$$
\hat{O}: \mathcal{S} \to \mathcal{S}
$$

The agent's policy generates an operator $\hat{O}$ that directly maps the current state to the next state.

## The GRL Tuple

Classical MDP: $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$

GRL: $(\mathcal{S}, \mathcal{O}, T, R, \gamma)$

Where:
- $\mathcal{S}$: State space
- $\mathcal{O}$: Space of operators on $\mathcal{S}$
- $T$: Transition function (often deterministic: $s' = \hat{O}(s)$)
- $R$: Reward function
- $\gamma$: Discount factor

## Operator Generator (Policy)

The policy in GRL is an **operator generator**:

$$
\pi_\theta: \mathcal{S} \to \mathcal{O}
$$

Given state $s$, the policy produces an operator $\hat{O} = \pi_\theta(s)$ which is then applied:

$$
s' = \hat{O}(s)
$$

This is fundamentally different from classical policies which output action vectors.

## Least-Action Principle

Inspired by physics, GRL incorporates the **principle of least action** as a regularizer:

$$
\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda \mathcal{E}(\hat{O})
$$

Where $\mathcal{E}(\hat{O})$ is the "energy" of the operator, encouraging:
- Smooth transformations
- Minimal perturbations
- Physically plausible behavior

## Classical RL as a Special Case

Standard continuous-action RL is recovered when operators are restricted to displacements:

$$
\hat{O}_a(s) = s + a
$$

Here $a$ is the classical action vector, and the operator simply translates the state.

## Next Steps

- [Operator Families](operator_families.md): Explore different operator architectures
- [Generalized Bellman](bellman.md): Value iteration with operators
- [Least-Action](least_action.md): Physics-inspired regularization
