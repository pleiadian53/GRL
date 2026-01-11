"""
Tests for GRL environments.
"""

import pytest
import numpy as np

from grl.envs import FieldNavigationEnv, OperatorPendulumEnv


class TestFieldNavigationEnv:
    """Tests for FieldNavigationEnv."""
    
    def test_init(self):
        """Test environment initialization."""
        env = FieldNavigationEnv()
        assert env.observation_space.shape == (4,)
        assert env.action_space.shape == (2,)
    
    def test_reset(self):
        """Test reset."""
        env = FieldNavigationEnv()
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (4,)
        assert "trajectory" in info
    
    def test_step(self):
        """Test stepping."""
        env = FieldNavigationEnv()
        env.reset(seed=42)
        
        action = np.array([0.5, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_goal_reaching(self):
        """Test that reaching goal terminates episode."""
        env = FieldNavigationEnv(goal_radius=10.0)  # Large radius
        obs, _ = env.reset(seed=42)
        
        # Move directly toward goal
        goal = obs[2:4]
        pos = obs[:2]
        direction = goal - pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        for _ in range(1000):
            obs, reward, terminated, truncated, info = env.step(direction)
            if terminated:
                assert info["reached_goal"]
                break
    
    def test_arena_bounds(self):
        """Test that agent stays within arena."""
        env = FieldNavigationEnv(arena_size=5.0)
        env.reset(seed=42)
        
        # Try to go way out of bounds
        action = np.array([100.0, 100.0])
        obs, _, _, _, _ = env.step(action)
        
        pos = obs[:2]
        assert np.all(np.abs(pos) <= 5.0)


class TestOperatorPendulumEnv:
    """Tests for OperatorPendulumEnv."""
    
    def test_init(self):
        """Test environment initialization."""
        env = OperatorPendulumEnv()
        assert env.observation_space.shape == (3,)
        assert env.action_space.shape == (1,)
    
    def test_reset(self):
        """Test reset."""
        env = OperatorPendulumEnv()
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (3,)
        # cos^2 + sin^2 = 1
        assert np.isclose(obs[0]**2 + obs[1]**2, 1.0, atol=1e-5)
    
    def test_step(self):
        """Test stepping."""
        env = OperatorPendulumEnv()
        env.reset(seed=42)
        
        action = np.array([1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (3,)
        assert isinstance(reward, float)
        assert not terminated  # Pendulum never terminates early
    
    def test_torque_clipping(self):
        """Test that torque is clipped."""
        env = OperatorPendulumEnv(max_torque=2.0)
        env.reset(seed=42)
        
        # Apply large torque
        obs, _, _, _, info = env.step(np.array([100.0]))
        
        assert np.abs(info["torque"]) <= 2.0
