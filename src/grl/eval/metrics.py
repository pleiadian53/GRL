"""
Evaluation metrics for GRL agents.
"""

from typing import Dict, List

import numpy as np
import torch


def compute_episode_return(rewards: List[float], gamma: float = 0.99) -> float:
    """
    Compute discounted return from a list of rewards.
    
    Args:
        rewards: List of rewards from an episode
        gamma: Discount factor
        
    Returns:
        Discounted return
    """
    ret = 0.0
    for r in reversed(rewards):
        ret = r + gamma * ret
    return ret


def compute_success_rate(
    episode_infos: List[Dict],
    success_key: str = "reached_goal",
) -> float:
    """
    Compute success rate from episode info dicts.
    
    Args:
        episode_infos: List of info dicts from episodes
        success_key: Key indicating success in info dict
        
    Returns:
        Success rate in [0, 1]
    """
    if not episode_infos:
        return 0.0
    
    successes = sum(1 for info in episode_infos if info.get(success_key, False))
    return successes / len(episode_infos)


def compute_trajectory_smoothness(trajectory: np.ndarray) -> Dict[str, float]:
    """
    Compute smoothness metrics for a trajectory.
    
    Smoother trajectories indicate the least-action principle
    is working effectively.
    
    Args:
        trajectory: Array of shape (T, state_dim)
        
    Returns:
        Dict with smoothness metrics
    """
    if len(trajectory) < 3:
        return {"velocity_std": 0.0, "acceleration_std": 0.0, "jerk": 0.0}
    
    trajectory = np.array(trajectory)
    
    # Velocity (first derivative)
    velocity = np.diff(trajectory, axis=0)
    velocity_std = np.std(np.linalg.norm(velocity, axis=1))
    
    # Acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    acceleration_std = np.std(np.linalg.norm(acceleration, axis=1))
    
    # Jerk (third derivative) - lower is smoother
    if len(trajectory) >= 4:
        jerk = np.diff(acceleration, axis=0)
        jerk_mean = np.mean(np.linalg.norm(jerk, axis=1))
    else:
        jerk_mean = 0.0
    
    return {
        "velocity_std": float(velocity_std),
        "acceleration_std": float(acceleration_std),
        "jerk": float(jerk_mean),
    }


def compute_operator_energy_stats(
    energies: List[float],
) -> Dict[str, float]:
    """
    Compute statistics on operator energies during an episode.
    
    Args:
        energies: List of operator energy values
        
    Returns:
        Dict with energy statistics
    """
    if not energies:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    
    energies = np.array(energies)
    return {
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "max": float(np.max(energies)),
        "min": float(np.min(energies)),
    }


def compare_with_baseline(
    grl_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compare GRL agent with baseline.
    
    Args:
        grl_metrics: Metrics from GRL agent
        baseline_metrics: Metrics from baseline (e.g., SAC)
        
    Returns:
        Dict with comparison ratios
    """
    comparison = {}
    
    for key in grl_metrics:
        if key in baseline_metrics and baseline_metrics[key] != 0:
            comparison[f"{key}_ratio"] = grl_metrics[key] / baseline_metrics[key]
        else:
            comparison[f"{key}_ratio"] = float("nan")
    
    return comparison
