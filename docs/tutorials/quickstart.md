# Quick Start: Your First GRL Agent

This tutorial walks through creating and training a GRL agent on the Field Navigation environment.

## Installation

```bash
# Clone and setup
git clone https://github.com/your-username/GRL.git
cd GRL

# Create environment
mamba env create -f environment.yml
mamba activate grl

# Install package
pip install -e .
```

## Basic Concepts

In GRL, agents don't output action vectors—they output **operators** that transform states:

```python
import torch
from grl.operators import FieldOperator

# Create a field operator
operator = FieldOperator(state_dim=4, hidden_dims=[64, 64])

# Apply operator to transform state
state = torch.randn(1, 4)
next_state = operator(state)

# Check operator energy (for least-action regularization)
energy = operator.energy()
print(f"Operator energy: {energy.item():.4f}")
```

## Creating an Agent

```python
from grl.policies import OperatorPolicy
from grl.algorithms import OperatorActorCritic

# Create an operator policy
policy = OperatorPolicy(
    state_dim=4,
    operator_class=FieldOperator,
    hidden_dims=[256, 256],
    exploration_noise=0.1,
    least_action_weight=0.01,
)

# Wrap in Operator-Actor-Critic
agent = OperatorActorCritic(
    policy=policy,
    state_dim=4,
    gamma=0.99,
    least_action_weight=0.01,
)
```

## Training on Field Navigation

```python
from grl.envs import FieldNavigationEnv

# Create environment
env = FieldNavigationEnv(arena_size=10.0, max_steps=200)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    episode_reward = 0
    
    for step in range(200):
        # Agent selects next state via operator
        next_state = agent.select_action(state)
        
        # Extract velocity for environment
        action = (next_state - state).squeeze(0).numpy()[:2]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Update agent
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        agent.update(
            state,
            torch.tensor([reward]),
            obs_tensor,
            torch.tensor([terminated or truncated]),
        )
        
        state = obs_tensor
        
        if terminated or truncated:
            break
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Return = {episode_reward:.2f}")
```

## Visualizing the Learned Field

```python
from grl.eval.visualization import plot_operator_field
import matplotlib.pyplot as plt

# Get the learned operator
operator = policy.get_operator()

# Plot the field
fig, ax = plot_operator_field(
    operator,
    xlim=(-10, 10),
    ylim=(-10, 10),
    resolution=20,
    title="Learned Navigation Field",
)

plt.savefig("learned_field.png")
```

## Using the CLI

```bash
# Train an agent
grl-train --env field_navigation --episodes 1000 --save-dir checkpoints/

# Evaluate
grl-evaluate checkpoints/final_checkpoint.pt --episodes 100

# Visualize
grl-visualize --checkpoint checkpoints/final_checkpoint.pt
```

## Next Steps

- [Field Navigation Tutorial](field_navigation.md): Deep dive into the navigation environment
- [Custom Operators](custom_operators.md): Define your own operator types
- [Comparing with Baselines](../algorithms/baselines.md): SAC/PPO comparison
