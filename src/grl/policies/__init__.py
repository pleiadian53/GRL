"""
Operator-based policies for GRL.

These policies generate operators instead of actions.
"""

from grl.policies.generator import OperatorGenerator
from grl.policies.policy import OperatorPolicy

__all__ = [
    "OperatorGenerator",
    "OperatorPolicy",
]
