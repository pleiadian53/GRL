"""
Evaluation utilities for GRL.

Includes:
- Performance metrics
- Visualization tools
- Comparison with baselines
"""

from grl.eval.metrics import (
    compute_episode_return,
    compute_success_rate,
    compute_trajectory_smoothness,
)
from grl.eval.visualization import (
    plot_operator_field,
    plot_trajectory,
    plot_training_curves,
)

__all__ = [
    "compute_episode_return",
    "compute_success_rate",
    "compute_trajectory_smoothness",
    "plot_operator_field",
    "plot_trajectory",
    "plot_training_curves",
]
