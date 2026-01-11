"""
Evaluation workflow for GRL agents.
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from grl.envs.field_navigation import FieldNavigationEnv
from grl.eval.metrics import (
    compute_episode_return,
    compute_success_rate,
    compute_trajectory_smoothness,
)
from grl.operators.field import FieldOperator
from grl.policies.policy import OperatorPolicy
from grl.utils.reproducibility import set_seed, get_device


def evaluate(
    checkpoint_path: str,
    env_name: str = "field_navigation",
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 0,
    device: Optional[str] = None,
    render: bool = False,
) -> Dict:
    """
    Evaluate a trained GRL agent.
    
    Args:
        checkpoint_path: Path to checkpoint file
        env_name: Environment name
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        device: Compute device
        render: Whether to render episodes
        
    Returns:
        Evaluation metrics
    """
    set_seed(seed)
    device = get_device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create environment
    if env_name == "field_navigation":
        render_mode = "rgb_array" if render else None
        env = FieldNavigationEnv(max_steps=max_steps, render_mode=render_mode)
        state_dim = 4
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Create and load policy
    policy = OperatorPolicy(
        state_dim=state_dim,
        operator_class=FieldOperator,
        hidden_dims=[256, 256],
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    # Evaluation loop
    episode_returns = []
    episode_infos = []
    all_smoothness = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_rewards = []
        trajectory = []
        
        for step in range(max_steps):
            with torch.no_grad():
                next_state = policy.act(state, deterministic=True)
            
            action = (next_state - state).squeeze(0).cpu().numpy()[:2]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            trajectory.append(obs[:2])
            
            if render:
                env.render()
            
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            if terminated or truncated:
                break
        
        episode_returns.append(compute_episode_return(episode_rewards))
        episode_infos.append(info)
        all_smoothness.append(compute_trajectory_smoothness(trajectory))
    
    # Aggregate metrics
    metrics = {
        "mean_return": sum(episode_returns) / len(episode_returns),
        "std_return": (sum((r - sum(episode_returns) / len(episode_returns)) ** 2 
                          for r in episode_returns) / len(episode_returns)) ** 0.5,
        "success_rate": compute_success_rate(episode_infos),
        "mean_jerk": sum(s["jerk"] for s in all_smoothness) / len(all_smoothness),
    }
    
    print("\nEvaluation Results:")
    print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Mean Jerk: {metrics['mean_jerk']:.4f}")
    
    return metrics


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GRL agent")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--env", default="field_navigation", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        env_name=args.env,
        num_episodes=args.episodes,
        render=args.render,
    )


if __name__ == "__main__":
    main()
