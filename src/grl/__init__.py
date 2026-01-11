"""
GRL: Generalized Reinforcement Learning

Actions as Operators on State Space.

This package provides the core implementation of GRL, including:
- Operator definitions and generators
- Operator-based policy networks
- Training algorithms (OAC, Operator-SAC)
- Custom environments for GRL experiments
- Evaluation and visualization utilities
"""

__version__ = "0.1.0"

from grl.operators import (
    ActionOperator,
    AffineOperator,
    FieldOperator,
    KernelOperator,
)
from grl.policies import OperatorPolicy, OperatorGenerator

__all__ = [
    "__version__",
    # Operators
    "ActionOperator",
    "AffineOperator", 
    "FieldOperator",
    "KernelOperator",
    # Policies
    "OperatorPolicy",
    "OperatorGenerator",
]
