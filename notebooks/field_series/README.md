# Field Series: From Vector Fields to Reinforcement Fields

**Understanding GRL's Core Concept Through Progressive Visualization**

This series of notebooks builds intuition for GRL's **reinforcement field** by starting with familiar concepts (classical vector fields) and progressively introducing the abstract notion of **functional fields**.

---

## Series Overview

```
┌─────────────────────────────────────────┐
│  Notebook 1: Classical Vector Fields    │
│  (Concrete: arrows at points)           │
└──────────────┬──────────────────────────┘
               │ You understand: arrows at points
               ↓
┌─────────────────────────────────────────┐
│  Notebook 2: Functional Fields          │
│  (Abstract: functions as vectors)       │
└──────────────┬──────────────────────────┘
               │ You understand: functions at points
               ↓
┌─────────────────────────────────────────┐
│  Notebook 3: Reinforcement Fields       │
│  (GRL's Q⁺ field in RKHS)               │
└─────────────────────────────────────────┘
               │ You understand: GRL's learning mechanism!
```

---

## Notebooks

| # | Notebook | Status | Description |
|---|----------|--------|-------------|
| 0 | `00_intro_vector_fields.ipynb` | ✅ Complete | Gentle intro with real-world examples (optional) |
| 1 | `01_classical_vector_fields.ipynb` | ✅ Complete | Gradient fields, rotational fields, superposition, trajectories |
| 1a | `01a_vector_fields_and_odes.ipynb` | ✅ Complete | ODEs, numerical solvers (Euler/RK4), flow matching connection 🔗 |
| 2 | `02_functional_fields.ipynb` | ✅ Complete | Functions as vectors, kernels, RKHS intuition |
| 3 | `03_reinforcement_fields/` | ✅ Complete | GRL's Q⁺ field, 2D navigation domain, policy inference 📁 |

**Note**: Notebook 3 is in a subdirectory with supplementary materials:
- `03_reinforcement_fields/03_reinforcement_fields.ipynb` — Main notebook
- `03_reinforcement_fields/03a_particle_coverage_effects.ipynb` — Visual proof of particle coverage effects
- `03_reinforcement_fields/particle_vs_gradient_fields.md` — Theory note

> 📋 **See [ROADMAP.md](ROADMAP.md)** for planned future notebooks (Policy Inference, Memory Update, RF-SARSA)

---

## Key Concepts

### Notebook 1: Classical Vector Fields

- **Vector field definition**: $\mathbf{F}(x, y) = [F_x(x,y), F_y(x,y)]^T$
- **Gradient fields**: $\nabla V$ points uphill on potential $V$
- **Rotational fields**: Circular flow, curl
- **Superposition**: $\mathbf{F}_{\text{total}} = \mathbf{F}_1 + \mathbf{F}_2$
- **Trajectories**: Following the field to find extrema

### Notebook 2: Functional Fields

- **Functions as vectors**: Addition, scaling, inner products
- **Kernel functions**: $k(x, x')$ as similarity measure
- **RKHS**: Reproducing Kernel Hilbert Space
- **Functional gradient**: Gradient in function space

### Notebook 3: Reinforcement Fields

- **Augmented space**: $z = (s, \theta)$ — state-action pairs
- **Particle memory**: $\{(z_i, w_i)\}$ — weighted experience points
- **Reinforcement field**: $Q^+(z) = \sum_i w_i \, k(z, z_i)$
- **Policy inference**: Reading the field to choose actions

---

## Related Projects

### 🔗 genai-lab — Generative AI & Diffusion Models

Notebook 1a bridges to the [genai-lab](https://github.com/pleiadian53/genai-lab) project, which covers **flow matching** and **diffusion models** — both built on the same vector field / ODE foundations:

| Topic | genai-lab Document |
|-------|-------------------|
| Flow Matching | `docs/flow_matching/01_flow_matching_foundations.md` |
| Diffusion Models | `docs/DDPM/01_ddpm_foundations.md` |
| Diffusion Transformers | `docs/diffusion/DiT/diffusion_transformer.md` |

**Shared concepts**: Velocity fields, ODE solvers (Euler, RK4), probability transport, gradient-based sampling.

---

## Learning Paths

### Quick Visual Tour (~45 min)
Run through all three notebooks, focusing on visualizations.

### Deep Understanding (~2 hours)
Combine notebooks with tutorial chapters:
- Notebook 2 → [Tutorial Ch 2: RKHS Foundations](../../docs/GRL0/tutorials/02-rkhs-foundations.md)
- Notebook 3 → [Tutorial Ch 4: Reinforcement Field](../../docs/GRL0/tutorials/04-reinforcement-field.md)

---

## Running the Notebooks

```bash
cd GRL/notebooks/field_series
conda activate grl  # or your environment
jupyter lab
```

### Dependencies

```
numpy
matplotlib
seaborn
ipywidgets (optional, for interactivity)
```

---

## Related Documentation

- [Tutorial Series](../../docs/GRL0/tutorials/)
- [RKHS Foundations](../../docs/GRL0/tutorials/02-rkhs-foundations.md)
- [Reinforcement Field](../../docs/GRL0/tutorials/04-reinforcement-field.md)
- [Original Paper](https://arxiv.org/abs/2208.04822)

---

**Last Updated**: January 15, 2026
