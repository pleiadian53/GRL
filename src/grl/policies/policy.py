"""
Operator policy for GRL.

The OperatorPolicy is the main interface for GRL agents.
It wraps an operator generator and provides methods for
action selection, exploration, and regularization.
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from grl.operators.base import ActionOperator
from grl.operators.field import FieldOperator
from grl.policies.generator import OperatorGenerator


class OperatorPolicy(nn.Module):
    """
    Policy that generates and applies operators.
    
    This is the GRL analogue of a policy network. Instead of outputting
    action vectors, it generates operators that transform the state space.
    
    Features:
    - Operator generation from state
    - Stochastic exploration via noise injection
    - Least-action regularization
    - Energy-based exploration bonus
    
    Args:
        state_dim: Dimensionality of the state space
        operator_class: Class of operator to use
        hidden_dims: Hidden dimensions for the generator
        exploration_noise: Std of Gaussian noise for exploration
        least_action_weight: Weight for least-action regularization
    """
    
    def __init__(
        self,
        state_dim: int,
        operator_class: Type[ActionOperator] = FieldOperator,
        hidden_dims: Optional[list] = None,
        exploration_noise: float = 0.1,
        least_action_weight: float = 0.01,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.exploration_noise = exploration_noise
        self.least_action_weight = least_action_weight
        
        # Core generator
        self.generator = OperatorGenerator(
            state_dim=state_dim,
            operator_class=operator_class,
            hidden_dims=hidden_dims,
        )
        
        # For direct field generation (simpler but less flexible)
        self.direct_field = operator_class(state_dim)
        self.use_direct = True  # Use direct field by default for simplicity
    
    def forward(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate operator and apply to get next state.
        
        Args:
            state: Current state (batch_size, state_dim)
            deterministic: If True, no exploration noise
            
        Returns:
            Tuple of:
                - next_state: Transformed state (batch_size, state_dim)
                - info: Dict with energy, noise, etc.
        """
        if self.use_direct:
            # Direct approach: single shared operator applied to all states
            next_state = self.direct_field(state)
            energy = self.direct_field.energy()
        else:
            # Generator approach: state-dependent operator
            next_state = self.generator.generate_and_apply(state)
            energy = torch.tensor(0.0)  # TODO: compute from generated operator
        
        # Add exploration noise
        if not deterministic and self.exploration_noise > 0:
            noise = torch.randn_like(next_state) * self.exploration_noise
            next_state = next_state + noise
        else:
            noise = torch.zeros_like(next_state)
        
        info = {
            "energy": energy,
            "noise": noise,
            "regularization": self.least_action_weight * energy,
        }
        
        return next_state, info
    
    def act(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get next state from current state (inference mode).
        
        Args:
            state: Current state
            deterministic: If True, no exploration
            
        Returns:
            Next state
        """
        next_state, _ = self.forward(state, deterministic=deterministic)
        return next_state
    
    def compute_regularization(self) -> torch.Tensor:
        """
        Compute the least-action regularization term.
        
        Returns:
            Regularization loss to be added to the main loss
        """
        if self.use_direct:
            return self.least_action_weight * self.direct_field.energy()
        else:
            return torch.tensor(0.0)
    
    def get_operator(self) -> ActionOperator:
        """Return the current operator for visualization."""
        if self.use_direct:
            return self.direct_field
        else:
            raise NotImplementedError("Generator-based operators not yet supported")


class StochasticOperatorPolicy(OperatorPolicy):
    """
    Stochastic operator policy with learned exploration.
    
    Instead of fixed Gaussian noise, this policy learns a
    distribution over operators, enabling principled exploration.
    
    The policy outputs:
    - Mean operator parameters
    - Log-std of operator parameters
    
    Sampling gives different operators for exploration.
    """
    
    def __init__(
        self,
        state_dim: int,
        operator_class: Type[ActionOperator] = FieldOperator,
        hidden_dims: Optional[list] = None,
        min_log_std: float = -20,
        max_log_std: float = 2,
    ):
        super().__init__(
            state_dim=state_dim,
            operator_class=operator_class,
            hidden_dims=hidden_dims,
            exploration_noise=0.0,  # Handled internally
        )
        
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Log-std head for stochastic operators
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.log_std_head = nn.Linear(hidden_dims[-1], self.generator.num_operator_params)
        nn.init.constant_(self.log_std_head.weight, 0)
        nn.init.constant_(self.log_std_head.bias, -1)  # Start with low variance
    
    def forward(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample operator from learned distribution.
        """
        # Get hidden representation
        hidden = self.generator.encoder(state)
        
        # Get mean and std of operator parameters
        mean = self.generator.param_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        
        # Sample or use mean
        if deterministic:
            params = mean
        else:
            params = mean + std * torch.randn_like(std)
        
        # Apply operator (simplified: use direct field with noise)
        next_state = self.direct_field(state)
        
        if not deterministic:
            next_state = next_state + params[:, :self.state_dim] * 0.1
        
        # Log probability for policy gradient
        log_prob = -0.5 * (
            ((params - mean) / (std + 1e-8)) ** 2 
            + 2 * log_std 
            + torch.log(torch.tensor(2 * 3.14159))
        ).sum(dim=-1)
        
        info = {
            "energy": self.direct_field.energy(),
            "log_prob": log_prob,
            "mean": mean,
            "std": std,
        }
        
        return next_state, info
