# GRL Documentation

This directory contains public, shareable documentation for the Generalized Reinforcement Learning (GRL) project.

## Contents

### Theory
Mathematical foundations of GRL:
- [GRL Overview](theory/README.md) - Introduction to actions as operators
- [Operator Families](theory/operator_families.md) - Types of operators (affine, field, kernel)
- [Generalized Bellman Equation](theory/bellman.md) - Value iteration with operators
- [Least-Action Principle](theory/least_action.md) - Physics-inspired regularization

### Algorithms
Training algorithms for GRL:
- [Operator-Actor-Critic](algorithms/oac.md) - Core algorithm
- [Operator-SAC](algorithms/sac.md) - Soft actor-critic variant
- [Policy Gradients](algorithms/policy_gradient.md) - REINFORCE-style updates

### Tutorials
Getting started guides:
- [Quick Start](tutorials/quickstart.md) - First GRL agent in 5 minutes
- [Field Navigation](tutorials/field_navigation.md) - Visual navigation demo
- [Custom Operators](tutorials/custom_operators.md) - Defining new operator types

### API Reference
- [Operators](api/operators.md)
- [Policies](api/policies.md)
- [Environments](api/environments.md)

## Private Development Notes

Private research notes are kept in `dev/` and are not intended for public sharing.
