# Part 1a: The Action Operator in GRL0

**Series:** Action Operators in Generalized Reinforcement Learning
**Part:** 1a of 4 (supplement to [Part 1: The GRL0 Gap](./01-the-grl0-gap.md))
**Audience:** Readers who want to see how far GRL0 already took the action-operator idea.

GRL0 introduced the action operator as a first-class concept and demonstrated it concretely. This note collects what the original paper contains — the worked example, the named definition, the matrix form, and an early version of the generator — so the relationship to the fully developed formalism in [Part 2](./02-action-operator-formalization.md) is clear.

---

## 1. The 2D navigation operator (Figs 1, 3, 4)

The paper's running example is a 2D navigation domain (Fig 1). Each action is a parameter vector $\mathbf{x}_a = (\Delta r, \Delta\theta)$ — a radius and an angle increment — indexed by the 12 clock directions ($a^{(i)}$, $i = 1 \ldots 12$).

The state-update mapping is written out explicitly: from position $(u, v)$, an action moves the agent to

$$(u + \Delta r \cos\Delta\theta,\; v + \Delta r \sin\Delta\theta).$$

This is a concrete action operator. The same domain carries the rest of the paper's intuition: Fig 3 illustrates policy inference where the local control policies are known a priori (query points A/B/C favor the 2-, 10-, and 6-o'clock moves), and Fig 4 is the navigation path-finding task (start → flag) used in the empirical study.

---

## 2. A named, first-class concept

The action operator is central, not incidental: the term appears in the paper's **title** and is the subject of **Section III.A**, where an action is defined as a construct that

> "specifies how an action *operates* on its input state, alters its configuration stochastically, and then produces an output state,"

and the navigation case is called "an application of such action operator."

---

## 3. The matrix / linear-operator form (Eq. 11)

The navigation update is also expressed as a matrix acting on the state — a linear operator:

$$s' = A_\theta \cdot s, \qquad \Delta x = \Delta r \cos\Delta\theta,\ \ \Delta y = \Delta r \sin\Delta\theta,$$

"a linear operator that takes an input state $(x, y)$ and produces the next state $(x + \Delta x, y + \Delta y)$." This is the *affine* operator family (the $b = 0$ case) — one of the structured families introduced in [Part 2](./02-action-operator-formalization.md) §3.1.

---

## 4. The generator, in nascent form

GRL0 also describes the *map from parameters to operators* — a higher-order function whose outputs are themselves functions:

> "an action operator [is] a **higher-order generative function** $f_\theta : \mathbf{x}_a \mapsto a(\mathbf{x}_a)$ that maps the action parameter $\mathbf{x}_a \in A_\theta$ to another function $a(\mathbf{x}_a): s \mapsto s'$ that acts on an input state $s$ and produces an output state $s'$ … $f$ is a higher-order function because its codomain again consists of functions; it is also **generative** in the sense that, depending on the action parameters, $f$ produces different actions, as functions."

This is the same idea later formalized as the **operator generator** $\Phi: \Theta \times \mathcal{S} \to \mathcal{S}$, with $\hat{O}_\theta(s) = \Phi(\theta, s)$. The paper also anticipates the general case — operators that "allow any arbitrary mappings including non-linear operations" — and refers to "a family of action operators."

---

## 5. Operators given a priori — as in standard RL

In GRL0's examples the operator is specified a priori, as part of the model. This is the same status that action semantics have in standard RL: in a classical MDP, the transition $T(s, a)$ and the meaning of each action are part of the problem specification — a modeling choice made by the designer, not something the framework derives or the agent learns. Giving the navigation operator a priori is the normal way action effects enter an RL problem.

---

## 6. From GRL0's sketch to the full formalism

GRL0 named and demonstrated the action operator and an early generator. [Part 2](./02-action-operator-formalization.md) develops these into a general, executable structure:

- a general operator space $\mathcal{O}$ with concrete **families** derived from choices of $\Phi$ (affine, field, potential, kernel, …);
- **algebraic structure** — composition, identity, monoid/group/Lie — so that skills and hierarchies are operator products;
- an **energy functional** $E(\hat{O})$ enabling least-action regularization;
- a **differentiable end-to-end pipeline** $\pi_\psi \to \theta \to \Phi \to \hat{O}_\theta \to s'$ trained by gradients.

---

## 7. Summary

| | GRL0 | Full formalism (Part 2) |
|---|---|---|
| Action as a function $s \mapsto s'$ | Yes — Section III.A, $a(\mathbf{x}_a)$ | The foundation |
| Concrete mapping $\theta \mapsto s'$ | Yes — navigation $(u + \Delta r\cos\Delta\theta, \ldots)$; matrix Eq. 11 | Multiple structured families |
| Parameter → operator map (generator) | Yes, in nascent form — $f_\theta$ | Formal, reusable, differentiable $\Phi$ |
| General nonlinear operators / families | Anticipated | Defined and instantiated |
| Operator space, algebra, energy | — | Monoid/group/Lie, $E(\hat{O})$ |
| Differentiable end-to-end pipeline | — | Yes |

---

**Back to:** [Part 1: The GRL0 Gap](./01-the-grl0-gap.md) · **Next:** [Part 2: Action Operator Formalization](./02-action-operator-formalization.md)
