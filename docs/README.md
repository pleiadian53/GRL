# GRL Documentation

You are browsing the documentation source. The rendered version is at **[pleiadian53.github.io/GRL](https://pleiadian53.github.io/GRL/)**, which is what most readers want.

Start at **[index.md](index.md)** for the project overview, or **[ROADMAP.md](ROADMAP.md)** for what is written, what is next, and what is planned.

## What is in here

| Directory | Contents |
|---|---|
| **[GRL0/](GRL0/)** | The tutorial paper on Reinforcement Fields, based on [arXiv:2208.04822](https://arxiv.org/abs/2208.04822). Part I (particle-based learning) in [`tutorials/`](GRL0/tutorials/), the [`quantum_inspired/`](GRL0/quantum_inspired/) extensions, [`implementation/`](GRL0/implementation/) specs, and [`recovering_classical_rl.md`](GRL0/recovering_classical_rl.md) |
| **[action_operator/](action_operator/)** | Actions as operators, Parts 1-4. The formalism, the operator families and Lie group structure, the learned-kernel extension, and [Part 4](action_operator/04-the-minimum-viable-experiment.md) on the first experiment |
| **[learning_with_action_operator/](learning_with_action_operator/)** | How an operator policy actually learns: actor-critic components and gradient flow |
| **[policy_gradient/](policy_gradient/)** | Background series: policy gradients, TRPO, PPO and variants, GRPO, and the bridge to operators |
| **[theory/](theory/)** | Short theory overview. The full treatment is in [action_operator/](action_operator/) |
| **[tutorials/](tutorials/)** | Quickstart |
| **[notebooks/](notebooks/)** | The field series: vector fields, functional fields, reinforcement fields |
| **[runpods/](runpods/)** | GPU pod setup notes |

`javascripts/` and `stylesheets/` are site assets. `CONTRIBUTING.md` and `LICENSE.md` are what they sound like.

## Conventions

**Status claims live in [ROADMAP.md](ROADMAP.md) and nowhere else.** This page and `index.md` deliberately carry no chapter counts or completion percentages. They previously drifted into four different counts for a single series, so the numbers now have exactly one home.

**This directory is public.** Private development notes, paper drafts, and planning live in `dev/`, which is gitignored. Do not link from `docs/` into `dev/`; those links cannot resolve on the published site.
