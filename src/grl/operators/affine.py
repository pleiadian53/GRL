"""
Affine operators for GRL.

Affine operators represent linear transformations plus translation:
    O(s) = As + b

These are the simplest non-trivial operators and recover classical
continuous-action RL when A = I (identity).
"""

from typing import Optional

import torch
import torch.nn as nn

from grl.operators.base import ActionOperator


class AffineOperator(ActionOperator):
    """
    Affine transformation operator: O(s) = As + b
    
    When A = I, this reduces to a displacement operator O(s) = s + b,
    which is equivalent to classical continuous actions.
    
    Args:
        state_dim: Dimensionality of the state space
        learnable_matrix: If True, A is learnable; if False, A = I (displacement only)
        init_identity: Initialize A close to identity for stability
    """
    
    def __init__(
        self,
        state_dim: int,
        learnable_matrix: bool = True,
        init_identity: bool = True,
    ):
        super().__init__(state_dim=state_dim)
        self.learnable_matrix = learnable_matrix
        
        if learnable_matrix:
            # Learnable transformation matrix
            self.A = nn.Parameter(torch.empty(state_dim, state_dim))
            if init_identity:
                # Initialize close to identity for stable training
                nn.init.eye_(self.A)
                self.A.data += 0.01 * torch.randn_like(self.A)
            else:
                nn.init.xavier_uniform_(self.A)
        else:
            # Fixed identity matrix (displacement-only mode)
            self.register_buffer("A", torch.eye(state_dim))
        
        # Translation vector (always learnable)
        self.b = nn.Parameter(torch.zeros(state_dim))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation: O(s) = As + b
        
        Args:
            state: Shape (batch_size, state_dim)
            
        Returns:
            Transformed state of shape (batch_size, state_dim)
        """
        # (batch, dim) @ (dim, dim).T + (dim,) -> (batch, dim)
        return torch.mm(state, self.A.T) + self.b
    
    def energy(self) -> torch.Tensor:
        """
        Compute operator energy based on deviation from identity.
        
        Energy = ||A - I||_F^2 + ||b||_2^2
        
        This encourages minimal transformations (least action).
        """
        if self.learnable_matrix:
            identity = torch.eye(self.state_dim, device=self.A.device)
            matrix_energy = torch.norm(self.A - identity, p="fro") ** 2
        else:
            matrix_energy = torch.tensor(0.0, device=self.b.device)
        
        translation_energy = torch.norm(self.b) ** 2
        
        return matrix_energy + translation_energy
    
    def spectral_norm(self) -> torch.Tensor:
        """Compute spectral norm of A for stability analysis."""
        if self.learnable_matrix:
            return torch.linalg.norm(self.A, ord=2)
        return torch.tensor(1.0, device=self.b.device)


class DisplacementOperator(AffineOperator):
    """
    Displacement operator: O(s) = s + b
    
    This is equivalent to classical continuous actions where the action
    is added to the state. A special case of AffineOperator with A = I.
    """
    
    def __init__(self, state_dim: int):
        super().__init__(state_dim=state_dim, learnable_matrix=False)
