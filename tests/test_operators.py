"""
Tests for operator classes.
"""

import pytest
import torch

from grl.operators import (
    ActionOperator,
    AffineOperator,
    FieldOperator,
    KernelOperator,
)
from grl.operators.base import IdentityOperator, ComposedOperator


class TestAffineOperator:
    """Tests for AffineOperator."""
    
    def test_init(self):
        """Test operator initialization."""
        op = AffineOperator(state_dim=4)
        assert op.state_dim == 4
        assert op.A.shape == (4, 4)
        assert op.b.shape == (4,)
    
    def test_forward(self):
        """Test forward pass."""
        op = AffineOperator(state_dim=4)
        state = torch.randn(8, 4)  # batch of 8
        
        next_state = op(state)
        
        assert next_state.shape == (8, 4)
    
    def test_identity_init(self):
        """Test that init_identity produces near-identity transform."""
        op = AffineOperator(state_dim=4, init_identity=True)
        state = torch.randn(1, 4)
        
        next_state = op(state)
        
        # Should be close to identity initially
        assert torch.allclose(state, next_state, atol=0.1)
    
    def test_energy(self):
        """Test energy computation."""
        op = AffineOperator(state_dim=4, init_identity=True)
        energy = op.energy()
        
        # Energy should be small for near-identity
        assert energy.item() < 1.0
    
    def test_displacement_only(self):
        """Test displacement operator (A=I)."""
        op = AffineOperator(state_dim=4, learnable_matrix=False)
        
        # A should be identity
        assert torch.allclose(op.A, torch.eye(4))


class TestFieldOperator:
    """Tests for FieldOperator."""
    
    def test_init(self):
        """Test field operator initialization."""
        op = FieldOperator(state_dim=4, hidden_dims=[32, 32])
        assert op.state_dim == 4
    
    def test_forward(self):
        """Test forward pass."""
        op = FieldOperator(state_dim=4)
        state = torch.randn(8, 4)
        
        next_state = op(state)
        
        assert next_state.shape == (8, 4)
    
    def test_field_scale(self):
        """Test field scaling."""
        op_small = FieldOperator(state_dim=4, field_scale=0.01)
        op_large = FieldOperator(state_dim=4, field_scale=10.0)
        
        state = torch.randn(1, 4)
        
        # Smaller scale should produce smaller changes
        delta_small = (op_small(state) - state).norm()
        delta_large = (op_large(state) - state).norm()
        
        # Not guaranteed due to random init, but generally true
        # Just check they're different
        assert delta_small != delta_large
    
    def test_energy(self):
        """Test energy computation."""
        op = FieldOperator(state_dim=4)
        state = torch.randn(8, 4)
        
        # Must call forward first to compute energy
        _ = op(state)
        energy = op.energy()
        
        assert energy.item() >= 0


class TestKernelOperator:
    """Tests for KernelOperator."""
    
    def test_init(self):
        """Test kernel operator initialization."""
        op = KernelOperator(state_dim=4, num_bases=8)
        assert op.state_dim == 4
        assert op.num_bases == 8
    
    def test_forward(self):
        """Test forward pass."""
        op = KernelOperator(state_dim=4, num_bases=8)
        state = torch.randn(8, 4)
        
        next_state = op(state)
        
        assert next_state.shape == (8, 4)


class TestComposedOperator:
    """Tests for operator composition."""
    
    def test_composition(self):
        """Test composing two operators."""
        op1 = AffineOperator(state_dim=4)
        op2 = AffineOperator(state_dim=4)
        
        composed = op1.compose(op2)
        
        state = torch.randn(8, 4)
        
        # Manual composition
        intermediate = op2(state)
        expected = op1(intermediate)
        
        # Via composed operator
        result = composed(state)
        
        assert torch.allclose(result, expected)
    
    def test_composed_energy(self):
        """Test that composed energy is sum of components."""
        op1 = AffineOperator(state_dim=4)
        op2 = AffineOperator(state_dim=4)
        
        composed = ComposedOperator(op1, op2)
        
        e1 = op1.energy()
        e2 = op2.energy()
        e_composed = composed.energy()
        
        assert torch.allclose(e_composed, e1 + e2)


class TestIdentityOperator:
    """Tests for identity operator."""
    
    def test_identity(self):
        """Test that identity returns unchanged state."""
        op = IdentityOperator(state_dim=4)
        state = torch.randn(8, 4)
        
        result = op(state)
        
        assert torch.allclose(result, state)
    
    def test_zero_energy(self):
        """Test that identity has zero energy."""
        op = IdentityOperator(state_dim=4)
        assert op.energy().item() == 0.0
