"""
Training algorithms for GRL.

Includes operator-based variants of standard RL algorithms:
- Operator-Actor-Critic (OAC)
- Operator-SAC
- Operator Policy Gradient
"""

from grl.algorithms.oac import OperatorActorCritic
from grl.algorithms.losses import (
    compute_td_loss,
    compute_policy_loss,
    compute_least_action_loss,
)

__all__ = [
    "OperatorActorCritic",
    "compute_td_loss",
    "compute_policy_loss", 
    "compute_least_action_loss",
]
