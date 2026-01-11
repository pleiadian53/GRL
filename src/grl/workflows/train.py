"""
Training workflow for GRL agents.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm

from grl.algorithms.oac import OperatorActorCritic
from grl.envs.field_navigation import FieldNavigationEnv
from grl.eval.metrics import compute_episode_return, compute_trajectory_smoothness
from grl.operators.field import FieldOperator
from grl.policies.policy import OperatorPolicy
from grl.utils.reproducibility import set_seed, get_device


def train(
    env_name: str = "field_navigation",
    num_episodes: int = 1000,
    max_steps: int = 200,
    seed: int = 42,
    device: Optional[str] = None,
    save_dir: Optional[str] = None,
    log_interval: int = 10,
) -> Dict:
    """
    Train a GRL agent.
    
    Args:
        env_name: Environment name
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        device: Compute device
        save_dir: Directory to save checkpoints
        log_interval: How often to log metrics
        
    Returns:
        Training metrics
    """
    set_seed(seed)
    device = get_device(device)
    
    # Create environment
    if env_name == "field_navigation":
        env = FieldNavigationEnv(max_steps=max_steps)
        state_dim = 4
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Create policy and algorithm
    policy = OperatorPolicy(
        state_dim=state_dim,
        operator_class=FieldOperator,
        hidden_dims=[256, 256],
        exploration_noise=0.1,
        least_action_weight=0.01,
    ).to(device)
    
    agent = OperatorActorCritic(
        policy=policy,
        state_dim=state_dim,
        gamma=0.99,
        least_action_weight=0.01,
    ).to(device)
    
    # Training loop
    metrics = {
        "episode_returns": [],
        "episode_lengths": [],
        "smoothness": [],
        "success_rate": [],
    }
    
    recent_successes = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, info = env.reset(seed=seed + episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_rewards = []
        episode_states = []
        
        for step in range(max_steps):
            # Select action
            next_state = agent.select_action(state, deterministic=False)
            
            # Convert to numpy for env
            action = (next_state - state).squeeze(0).cpu().numpy()[:2]  # Take first 2 dims as velocity
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_states.append(obs[:2])  # Position only
            
            # Prepare for update
            next_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            done_t = torch.tensor([terminated or truncated], dtype=torch.float32, device=device)
            
            # Update agent
            agent.update(state, reward_t, next_obs, done_t)
            
            state = next_obs
            
            if terminated or truncated:
                break
        
        # Record metrics
        episode_return = compute_episode_return(episode_rewards)
        metrics["episode_returns"].append(episode_return)
        metrics["episode_lengths"].append(len(episode_rewards))
        
        smoothness = compute_trajectory_smoothness(episode_states)
        metrics["smoothness"].append(smoothness["jerk"])
        
        recent_successes.append(info.get("reached_goal", False))
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        metrics["success_rate"].append(sum(recent_successes) / len(recent_successes))
        
        # Log
        if (episode + 1) % log_interval == 0:
            avg_return = sum(metrics["episode_returns"][-log_interval:]) / log_interval
            success_rate = metrics["success_rate"][-1]
            print(f"\nEpisode {episode + 1}: Return={avg_return:.2f}, Success={success_rate:.2%}")
    
    # Save checkpoint
    if save_dir:
        save_path = Path(save_dir) / "final_checkpoint.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "agent_state_dict": agent.state_dict(),
            "metrics": metrics,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    return metrics


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GRL agent")
    parser.add_argument("--env", default="field_navigation", help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", default="checkpoints", help="Save directory")
    
    args = parser.parse_args()
    
    train(
        env_name=args.env,
        num_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
