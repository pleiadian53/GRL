"""
Field operators for GRL.

Field operators generate vector fields over the state space:
    O(s) = s + F(s)

where F: S -> S is a learned vector field. This is particularly
powerful for navigation and control tasks where the field
represents forces, velocities, or potential gradients.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from grl.operators.base import ActionOperator


class FieldOperator(ActionOperator):
    """
    Field operator that adds a learned vector field to the state.
    
    O(s) = s + F_θ(s)
    
    The field F is represented by a neural network. The least-action
    energy is computed as the norm of the field output.
    
    Args:
        state_dim: Dimensionality of the state space
        hidden_dims: List of hidden layer dimensions for the field network
        activation: Activation function
        field_scale: Scaling factor for the field output
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "tanh",
        field_scale: float = 1.0,
    ):
        super().__init__(state_dim=state_dim)
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        self.field_scale = field_scale
        self._last_field = None  # Cache for energy computation
        
        # Build field network
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, state_dim))
        
        self.field_network = nn.Sequential(*layers)
        
        # Initialize output layer to small values for stability
        nn.init.zeros_(self.field_network[-1].weight)
        nn.init.zeros_(self.field_network[-1].bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply field operator: O(s) = s + scale * F(s)
        
        Args:
            state: Shape (batch_size, state_dim)
            
        Returns:
            Transformed state of shape (batch_size, state_dim)
        """
        field = self.field_network(state) * self.field_scale
        self._last_field = field  # Cache for energy
        return state + field
    
    def energy(self) -> torch.Tensor:
        """
        Compute energy as the mean squared norm of the field.
        
        Energy = E[||F(s)||^2]
        
        This encourages smooth, minimal-magnitude fields.
        """
        if self._last_field is None:
            return torch.tensor(0.0)
        return torch.mean(self._last_field ** 2)
    
    def get_field(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the field values without applying the transformation.
        
        Useful for visualization.
        
        Args:
            state: Query states of shape (batch_size, state_dim)
            
        Returns:
            Field vectors of shape (batch_size, state_dim)
        """
        return self.field_network(state) * self.field_scale


class PotentialFieldOperator(FieldOperator):
    """
    Potential field operator where the field is the gradient of a scalar potential.
    
    O(s) = s - ∇φ(s)
    
    This models conservative forces (like gravity or electrostatics).
    The negative gradient ensures the system evolves toward lower potential.
    
    Args:
        state_dim: Dimensionality of the state space
        hidden_dims: Hidden dimensions for the potential network
        step_size: Step size for gradient descent
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Optional[List[int]] = None,
        step_size: float = 0.1,
    ):
        # Don't call super().__init__ since we build a different network
        ActionOperator.__init__(self, state_dim=state_dim)
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        self.step_size = step_size
        self._last_gradient = None
        
        # Build potential network: S -> R
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.potential_network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply potential field: O(s) = s - η∇φ(s)
        
        Uses autograd to compute the gradient of the potential.
        """
        state = state.requires_grad_(True)
        potential = self.potential_network(state).sum()
        
        gradient = torch.autograd.grad(
            potential, state, create_graph=True
        )[0]
        
        self._last_gradient = gradient
        return state - self.step_size * gradient
    
    def energy(self) -> torch.Tensor:
        """Energy is the squared gradient norm."""
        if self._last_gradient is None:
            return torch.tensor(0.0)
        return torch.mean(self._last_gradient ** 2)
    
    def get_potential(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get potential values for visualization.
        
        Args:
            state: Query states of shape (batch_size, state_dim)
            
        Returns:
            Potential values of shape (batch_size, 1)
        """
        return self.potential_network(state)
