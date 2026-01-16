# Notebook 3: Reinforcement Fields in GRL

This directory contains the main notebook and supplementary materials for understanding GRL's reinforcement fields.

## Contents

| File | Description |
|------|-------------|
| `03_reinforcement_fields.ipynb` | **Main notebook**: 2D navigation domain, particles, Q⁺ field, policy inference |
| `03a_particle_coverage_effects.ipynb` | **Supplementary**: Visual proof of how particle coverage affects policy field behavior |
| `particle_vs_gradient_fields.md` | **Theory note**: Detailed comparison of particle-based Q⁺ vs true gradient fields |

## Key Question Addressed

**Why do policy arrows appear parallel instead of converging on the goal?**

The supplementary notebook (`03a`) provides visual proof that:

- **Sparse particles** → parallel arrows (limited directional info)
- **Rich particles** → converging arrows (diverse directional info)
- **True gradient** → perfect convergence (geometric, not learned)

## Related

- Field series overview: `../README.md`
- Theory docs: `../../../docs/theory/particle_vs_gradient_fields.md` (canonical location)
