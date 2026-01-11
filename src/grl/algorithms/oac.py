"""
Operator-Actor-Critic (OAC) algorithm.

This is the core training algorithm for GRL, extending
actor-critic methods to operator-based policies.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from grl.policies.policy import OperatorPolicy


class ValueNetwork(nn.Module):
    """
    State value function V(s).
    
    Estimates the expected return from a given state under
    the current operator policy.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate."""
        return self.network(state).squeeze(-1)


class QNetwork(nn.Module):
    """
    State-operator value function Q(s, O).
    
    Since operators are parameterized, we estimate Q(s, next_s)
    where next_s = O(s). This simplifies to Q(s, s') which
    can be approximated by conditioning on both states.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        layers = []
        in_dim = state_dim * 2  # Concatenate s and s'
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self, 
        state: torch.Tensor, 
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value for state-operator pair."""
        x = torch.cat([state, next_state], dim=-1)
        return self.network(x).squeeze(-1)


class OperatorActorCritic(nn.Module):
    """
    Operator-Actor-Critic algorithm.
    
    Extends actor-critic to operator-based policies with:
    - Operator policy (actor)
    - Value function (critic)
    - Least-action regularization
    
    Args:
        policy: The operator policy to train
        state_dim: State dimensionality
        gamma: Discount factor
        tau: Target network update rate
        least_action_weight: Weight for operator energy regularization
        lr_policy: Learning rate for policy
        lr_value: Learning rate for value function
    """
    
    def __init__(
        self,
        policy: OperatorPolicy,
        state_dim: Optional[int] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        least_action_weight: float = 0.01,
        lr_policy: float = 3e-4,
        lr_value: float = 3e-4,
    ):
        super().__init__()
        
        self.policy = policy
        self.state_dim = state_dim or policy.state_dim
        self.gamma = gamma
        self.tau = tau
        self.least_action_weight = least_action_weight
        
        # Value networks
        self.value = ValueNetwork(self.state_dim)
        self.value_target = ValueNetwork(self.state_dim)
        self.value_target.load_state_dict(self.value.state_dict())
        
        # Q-networks (twin for stability)
        self.q1 = QNetwork(self.state_dim)
        self.q2 = QNetwork(self.state_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(
            list(self.value.parameters()) + 
            list(self.q1.parameters()) + 
            list(self.q2.parameters()),
            lr=lr_value
        )
    
    def update(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one update step.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            rewards: Batch of rewards (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            dones: Batch of done flags (batch_size,)
            
        Returns:
            Dict of training metrics
        """
        # ===== Critic Update =====
        with torch.no_grad():
            # Target value
            next_value = self.value_target(next_states)
            target_q = rewards + self.gamma * (1 - dones.float()) * next_value
        
        # Predicted next state from policy
        predicted_next, info = self.policy(states, deterministic=True)
        
        # Q-value estimates
        q1 = self.q1(states, next_states)
        q2 = self.q2(states, next_states)
        
        # Q-loss (MSE to target)
        q1_loss = nn.functional.mse_loss(q1, target_q)
        q2_loss = nn.functional.mse_loss(q2, target_q)
        
        # Value loss
        min_q = torch.min(q1, q2)
        value_pred = self.value(states)
        value_loss = nn.functional.mse_loss(value_pred, min_q.detach())
        
        critic_loss = q1_loss + q2_loss + value_loss
        
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()
        
        # ===== Policy Update =====
        # Generate next states from policy
        predicted_next, info = self.policy(states, deterministic=False)
        
        # Policy loss: maximize Q-value
        q_policy = self.q1(states, predicted_next)
        policy_loss = -q_policy.mean()
        
        # Least-action regularization
        energy_loss = self.least_action_weight * info["energy"]
        
        total_policy_loss = policy_loss + energy_loss
        
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()
        
        # ===== Target Update =====
        self._soft_update(self.value, self.value_target)
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "energy": info["energy"].item() if isinstance(info["energy"], torch.Tensor) else info["energy"],
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak averaging for target network."""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1 - self.tau) * tgt_param.data
            )
    
    def select_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> torch.Tensor:
        """Select action (next state) given current state."""
        with torch.no_grad():
            next_state, _ = self.policy(state, deterministic=deterministic)
        return next_state
