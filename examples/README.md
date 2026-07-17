# GRL Examples

**Status: none of these are written yet.** This directory holds the plan; the scripts do not exist. It is here so the intended shape of the examples layer is on record, not because there is anything to run today.

## What this directory is for

`examples/<topic>/` develops concrete workflows, scripts, and use cases *while* the reusable abstractions are built under `src/grl/`. The examples are not demos filed after the fact. They put the code into real use, which is how the abstractions get tested for whether they are correct, useful, and behaving as expected. Once a group of use cases matures, it refactors into the application layer.

That convention has a precondition: there must be something under `src/` worth exercising, and a question worth answering by exercising it.

## Planned

The underlying library code for most of these already exists, so they are cheap to write once there is a reason to.

| Planned script | Exercises | Backing code |
|---|---|---|
| `01_basic_operator.py` | Construct and apply each operator family; visualize what each does to a state | `src/grl/operators/{base,affine,field,kernel}.py` |
| `02_field_navigation.py` | Train a GRL agent on 2D navigation; visualize the learned field | `src/grl/envs/field_navigation.py` |
| `03_pendulum_control.py` | Operator-based pendulum control with torque-field visualization | `src/grl/envs/operator_pendulum.py` |
| `04_custom_operator.py` | Define a custom operator type for a specific domain | `src/grl/operators/base.py` |
| `05_baseline_comparison.py` | GRL vs SAC: learning curves, trajectory smoothness, final performance | `src/grl/algorithms/oac.py` |

## The next topic directory

The first `examples/<topic>/` to be created should be a **sequential-action** topic, not any of the above.

The reason is in [Action Operators, Part 4](../docs/action_operator/04-the-minimum-viable-experiment.md): the operator formalism is being validated in the sibling [ssl-lab](https://github.com/pleiadian53/ssl-lab) project, on a domain where interventions are simultaneous. That setting can observe only the swap-even half of the bracket algebra, which reads as epistasis. The swap-odd half, which reads as path-dependence and is GRL's own claim, becomes observable only when actions are sequenced. ssl-lab structurally cannot reach it. GRL can.

**Trigger:** create that topic directory when the first generator / exponential-map abstraction lands in `src/grl/operators/`. Until then the plan lives in `dev/planning/action_operator/`.

## Existing runnable code

The CLI entry points in `pyproject.toml` are real:

```bash
mamba activate grl
grl-train --env field_navigation --episodes 1000 --save-dir checkpoints/
grl-evaluate checkpoints/final_checkpoint.pt --episodes 100
grl-visualize --checkpoint checkpoints/final_checkpoint.pt
```

See [`docs/tutorials/quickstart.md`](../docs/tutorials/quickstart.md).
