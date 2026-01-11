"""
GRL environments.

Custom environments designed to showcase operator-based policies:
- Field navigation (potential fields)
- Operator-based control (pendulum, cartpole)
- PDE control environments
"""

from grl.envs.field_navigation import FieldNavigationEnv
from grl.envs.operator_pendulum import OperatorPendulumEnv

__all__ = [
    "FieldNavigationEnv",
    "OperatorPendulumEnv",
]
