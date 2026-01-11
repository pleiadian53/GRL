"""
High-level workflows for GRL.

Entry points for training, evaluation, and visualization.
"""

from grl.workflows.train import train, main as train_main
from grl.workflows.evaluate import evaluate, main as evaluate_main
from grl.workflows.visualize import visualize_operator, main as visualize_main

__all__ = [
    "train",
    "train_main",
    "evaluate",
    "evaluate_main",
    "visualize_operator",
    "visualize_main",
]
