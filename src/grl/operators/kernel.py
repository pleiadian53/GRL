"""
Kernel operators for GRL.

Kernel operators transform states using kernel-weighted combinations
of basis transformations. They are particularly suited for:
- Spatially varying transformations
- Attention-based state modifications
- Non-local operator effects
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from grl.operators.base import ActionOperator


class KernelOperator(ActionOperator):
    """
    Kernel-based operator using learned attention over basis operators.
    
    O(s) = Σ_i α_i(s) * T_i(s)
    
    where α_i are attention weights computed from s, and T_i are basis
    transformation functions. This enables adaptive, context-dependent
    operator behavior.
    
    Args:
        state_dim: Dimensionality of the state space
        num_bases: Number of basis transformations
        hidden_dims: Hidden dimensions for basis networks
    """
    
    def __init__(
        self,
        state_dim: int,
        num_bases: int = 8,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__(state_dim=state_dim)
        
        if hidden_dims is None:
            hidden_dims = [64]
        
        self.num_bases = num_bases
        self._last_weights = None
        
        # Attention network: computes weights over bases
        attention_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            attention_layers.append(nn.Linear(in_dim, hidden_dim))
            attention_layers.append(nn.ReLU())
            in_dim = hidden_dim
        attention_layers.append(nn.Linear(in_dim, num_bases))
        self.attention_net = nn.Sequential(*attention_layers)
        
        # Basis transformations: simple linear for efficiency
        self.basis_transforms = nn.ModuleList([
            nn.Linear(state_dim, state_dim)
            for _ in range(num_bases)
        ])
        
        # Initialize bases near identity
        for i, transform in enumerate(self.basis_transforms):
            nn.init.eye_(transform.weight)
            nn.init.zeros_(transform.bias)
            # Add small perturbation for diversity
            transform.weight.data += 0.1 * torch.randn_like(transform.weight)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel operator: O(s) = Σ_i softmax(α)_i * T_i(s)
        
        Args:
            state: Shape (batch_size, state_dim)
            
        Returns:
            Transformed state of shape (batch_size, state_dim)
        """
        # Compute attention weights
        logits = self.attention_net(state)  # (batch, num_bases)
        weights = F.softmax(logits, dim=-1)
        self._last_weights = weights
        
        # Apply weighted combination of basis transforms
        result = torch.zeros_like(state)
        for i, transform in enumerate(self.basis_transforms):
            basis_output = transform(state)  # (batch, state_dim)
            result += weights[:, i:i+1] * basis_output
        
        return result
    
    def energy(self) -> torch.Tensor:
        """
        Energy based on entropy of attention weights.
        
        Low entropy (concentrated on one basis) = low energy
        High entropy (uniform) = high energy
        
        This encourages the operator to commit to specific transformations.
        """
        if self._last_weights is None:
            return torch.tensor(0.0)
        
        # Entropy: -Σ p log p
        entropy = -torch.sum(
            self._last_weights * torch.log(self._last_weights + 1e-8),
            dim=-1
        )
        return torch.mean(entropy)


class RBFKernelOperator(ActionOperator):
    """
    RBF (Radial Basis Function) kernel operator.
    
    O(s) = s + Σ_i w_i * exp(-||s - c_i||^2 / σ^2) * d_i
    
    where c_i are centers, d_i are displacement directions, and w_i are weights.
    This creates localized operator effects centered at learned locations.
    
    Useful for:
    - Navigation with localized attractors/repellers
    - Multi-modal control policies
    - Obstacle avoidance
    
    Args:
        state_dim: Dimensionality of the state space
        num_centers: Number of RBF centers
        sigma: RBF kernel width
        learnable_sigma: Whether sigma is learnable
    """
    
    def __init__(
        self,
        state_dim: int,
        num_centers: int = 16,
        sigma: float = 1.0,
        learnable_sigma: bool = False,
    ):
        super().__init__(state_dim=state_dim)
        
        self.num_centers = num_centers
        
        # RBF centers
        self.centers = nn.Parameter(torch.randn(num_centers, state_dim))
        
        # Displacement directions at each center
        self.directions = nn.Parameter(torch.randn(num_centers, state_dim) * 0.1)
        
        # Kernel width
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(sigma)))
        
        self._last_activations = None
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply RBF kernel operator.
        
        Args:
            state: Shape (batch_size, state_dim)
            
        Returns:
            Transformed state of shape (batch_size, state_dim)
        """
        # Compute distances to centers: (batch, num_centers)
        diff = state.unsqueeze(1) - self.centers.unsqueeze(0)  # (batch, centers, dim)
        distances_sq = torch.sum(diff ** 2, dim=-1)  # (batch, centers)
        
        # RBF activations
        activations = torch.exp(-distances_sq / (2 * self.sigma ** 2))
        self._last_activations = activations
        
        # Weighted sum of directions
        # (batch, centers, 1) * (1, centers, dim) -> (batch, dim)
        displacement = torch.sum(
            activations.unsqueeze(-1) * self.directions.unsqueeze(0),
            dim=1
        )
        
        return state + displacement
    
    def energy(self) -> torch.Tensor:
        """
        Energy based on magnitude of displacements and center spread.
        """
        direction_energy = torch.mean(self.directions ** 2)
        
        # Encourage diverse, spread-out centers
        center_distances = torch.cdist(self.centers, self.centers)
        center_energy = -torch.mean(center_distances)  # Negative: encourage spread
        
        return direction_energy + 0.1 * center_energy
