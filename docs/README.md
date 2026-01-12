# GRL Documentation

This directory contains public, shareable documentation for the Generalized Reinforcement Learning (GRL) project.

## Original Publication

**Chiu, P.-H., & Huber, M. (2022).** *Generalized Reinforcement Learning: Experience Particles, Action Operator, Reinforcement Field, Memory Association, and Decision Concepts.* arXiv:2208.04822.

**[Read on arXiv →](https://arxiv.org/abs/2208.04822)** (37 pages, 15 figures)

---

## Contents

### Tutorial Papers: Reinforcement Fields

**[GRL0/](GRL0/)** — Two-part tutorial series based on the original paper:

**Part I: Particle-Based Learning** (6/10 chapters complete)
- Augmented state-action space
- Particle memory as belief state
- MemoryUpdate and RF-SARSA algorithms
- POMDP interpretation

**Part II: Emergent Structure & Spectral Abstraction** (Planned)
- Functional clustering in RKHS
- Spectral concept discovery
- Hierarchical policy organization

**[Start Learning →](GRL0/tutorials/00-overview.md)**

---

### Extensions: Actions as Operators (In Development)

Future documentation for operator-based extensions (Papers A, B, C):

#### Theory
Mathematical foundations of operator-based GRL:
- GRL Overview - Introduction to actions as operators
- Operator Families - Types of operators (affine, field, kernel)
- Generalized Bellman Equation - Value iteration with operators
- Least-Action Principle - Physics-inspired regularization

#### Algorithms
Training algorithms for operator-based GRL:
- Operator-Actor-Critic (OAC) - Core algorithm
- Operator-SAC - Soft actor-critic variant
- Policy Gradients - REINFORCE-style updates

#### Tutorials
Getting started guides:
- Quick Start - First GRL agent in 5 minutes
- Field Navigation - Visual navigation demo
- Custom Operators - Defining new operator types

#### API Reference
- Operators
- Policies
- Environments

---

## Documentation Organization

**Public documentation** (this directory):
- Tutorial papers, guides, API references
- Intended for sharing with the research community

**Private development notes** (`dev/`):
- Research drafts, brainstorming, work-in-progress
- Not intended for public sharing

---

**Last Updated**: January 12, 2026
