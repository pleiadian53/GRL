"""
Operator generator networks for GRL.

An operator generator is a neural network that takes state as input
and outputs the parameters of an operator. This is the GRL analogue
of a policy network in classical RL.
"""

from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn

from grl.operators.base import ActionOperator
from grl.operators.affine import AffineOperator
from grl.operators.field import FieldOperator


class OperatorGenerator(nn.Module):
    """
    Neural network that generates operator parameters from state.
    
    Given a state s, the generator outputs parameters θ that define
    an operator O_θ. The operator is then applied: s' = O_θ(s).
    
    This is a meta-network: it produces functions (operators), not values.
    
    Args:
        state_dim: Dimensionality of the state space
        operator_class: Class of operator to generate
        hidden_dims: Hidden layer dimensions
        operator_kwargs: Additional kwargs passed to operator constructor
    """
    
    def __init__(
        self,
        state_dim: int,
        operator_class: Type[ActionOperator] = AffineOperator,
        hidden_dims: Optional[List[int]] = None,
        operator_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.operator_class = operator_class
        self.operator_kwargs = operator_kwargs or {}
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Compute number of parameters needed for the operator
        self.num_operator_params = self._compute_num_params()
        
        # Build encoder network: state -> hidden representation
        encoder_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Parameter head: hidden -> operator parameters
        self.param_head = nn.Linear(in_dim, self.num_operator_params)
        
        # Initialize to produce near-identity operators
        nn.init.zeros_(self.param_head.weight)
        nn.init.zeros_(self.param_head.bias)
    
    def _compute_num_params(self) -> int:
        """Compute number of parameters needed for the operator class."""
        # Create a dummy operator to count parameters
        dummy = self.operator_class(self.state_dim, **self.operator_kwargs)
        return sum(p.numel() for p in dummy.parameters())
    
    def forward(self, state: torch.Tensor) -> ActionOperator:
        """
        Generate an operator from the current state.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            An ActionOperator with parameters set from the network output
        """
        # Encode state to hidden representation
        hidden = self.encoder(state)
        
        # Generate operator parameters
        params = self.param_head(hidden)
        
        # Create operator and set parameters
        operator = self.operator_class(self.state_dim, **self.operator_kwargs)
        self._set_operator_params(operator, params)
        
        return operator
    
    def _set_operator_params(
        self, 
        operator: ActionOperator, 
        params: torch.Tensor
    ) -> None:
        """
        Set operator parameters from a flat parameter vector.
        
        Args:
            operator: The operator to modify
            params: Flat parameter vector of shape (batch_size, num_params)
        """
        offset = 0
        for p in operator.parameters():
            numel = p.numel()
            # Take mean over batch for shared operator
            # (In practice, you might want per-sample operators)
            p.data = params[:, offset:offset + numel].mean(dim=0).view(p.shape)
            offset += numel
    
    def generate_and_apply(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate an operator and apply it in one step.
        
        This is the main interface for training.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Next state tensor of shape (batch_size, state_dim)
        """
        operator = self.forward(state)
        return operator(state)


class HypernetworkGenerator(nn.Module):
    """
    Hypernetwork-based operator generator.
    
    Uses a hypernetwork to directly generate the weights of a target
    operator network. More expressive than parameter prediction but
    also more computationally expensive.
    
    Args:
        state_dim: Dimensionality of the state space
        target_hidden_dims: Hidden dims of the target operator network
        hyper_hidden_dims: Hidden dims of the hypernetwork
    """
    
    def __init__(
        self,
        state_dim: int,
        target_hidden_dims: Optional[List[int]] = None,
        hyper_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        if target_hidden_dims is None:
            target_hidden_dims = [64, 64]
        if hyper_hidden_dims is None:
            hyper_hidden_dims = [128, 128]
        
        self.target_hidden_dims = target_hidden_dims
        
        # Calculate target network parameter counts
        self.target_shapes = []
        in_dim = state_dim
        for hidden_dim in target_hidden_dims:
            self.target_shapes.append((in_dim, hidden_dim))  # weight
            self.target_shapes.append((hidden_dim,))  # bias
            in_dim = hidden_dim
        self.target_shapes.append((in_dim, state_dim))  # output weight
        self.target_shapes.append((state_dim,))  # output bias
        
        total_params = sum(
            s[0] * s[1] if len(s) == 2 else s[0] 
            for s in self.target_shapes
        )
        
        # Build hypernetwork
        hyper_layers = []
        in_dim = state_dim
        for hidden_dim in hyper_hidden_dims:
            hyper_layers.append(nn.Linear(in_dim, hidden_dim))
            hyper_layers.append(nn.ReLU())
            in_dim = hidden_dim
        hyper_layers.append(nn.Linear(in_dim, total_params))
        
        self.hypernet = nn.Sequential(*hyper_layers)
        
        # Initialize for small initial perturbations
        nn.init.zeros_(self.hypernet[-1].weight)
        nn.init.zeros_(self.hypernet[-1].bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate operator and apply it to state.
        
        Args:
            state: Shape (batch_size, state_dim)
            
        Returns:
            Next state of shape (batch_size, state_dim)
        """
        batch_size = state.shape[0]
        
        # Generate target network parameters
        params = self.hypernet(state)  # (batch, total_params)
        
        # Apply target network with generated parameters
        x = state
        offset = 0
        
        for i, shape in enumerate(self.target_shapes):
            if len(shape) == 2:  # Weight matrix
                in_dim, out_dim = shape
                numel = in_dim * out_dim
                W = params[:, offset:offset + numel].view(batch_size, in_dim, out_dim)
                offset += numel
                
                # Batch matrix multiply: (batch, 1, in) @ (batch, in, out) -> (batch, 1, out)
                x = torch.bmm(x.unsqueeze(1), W).squeeze(1)
            else:  # Bias
                dim = shape[0]
                b = params[:, offset:offset + dim]
                offset += dim
                x = x + b
                
                # Apply activation (except last layer)
                if i < len(self.target_shapes) - 1:
                    x = torch.tanh(x)
        
        # Add residual connection for stability
        return state + x
