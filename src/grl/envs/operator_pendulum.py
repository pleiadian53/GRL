"""
Operator-based Pendulum Environment.

A pendulum control task where actions are operators
that modify the angular momentum or apply torque fields.

This demonstrates GRL on a classic control benchmark,
allowing direct comparison with SAC/PPO baselines.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OperatorPendulumEnv(gym.Env):
    """
    Pendulum with operator-based actions.
    
    The standard pendulum task, but framed for GRL:
    - State: [cos(θ), sin(θ), θ_dot]
    - Action: torque operator applied to angular velocity
    
    The operator can be thought of as a function that
    transforms the current (θ, θ_dot) state to the next.
    
    For GRL, we expose the full state and expect the policy
    to output the state transformation directly.
    
    Args:
        g: Gravity constant
        m: Pendulum mass
        l: Pendulum length
        dt: Time step
        max_torque: Maximum applicable torque
        max_speed: Maximum angular velocity
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
        dt: float = 0.05,
        max_torque: float = 2.0,
        max_speed: float = 8.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.g = g
        self.m = m
        self.l = l
        self.dt = dt
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.render_mode = render_mode
        
        # State: [cos(θ), sin(θ), θ_dot]
        high = np.array([1.0, 1.0, max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32,
        )
        
        # Action: torque (can be interpreted as momentum change)
        self.action_space = spaces.Box(
            low=-max_torque,
            high=max_torque,
            shape=(1,),
            dtype=np.float32,
        )
        
        self.theta = None
        self.theta_dot = None
        self.step_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Random initial state (mostly hanging down)
        high = np.array([np.pi, 1.0])
        self.theta, self.theta_dot = self.np_random.uniform(
            low=-high, high=high
        )
        
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply torque and simulate one step.
        
        For GRL interpretation: the action represents how
        the operator modifies the momentum state.
        """
        torque = np.clip(action[0], -self.max_torque, self.max_torque)
        
        # Physics update (Euler integration)
        # θ'' = -3g/(2l) * sin(θ) + 3/(ml²) * τ
        theta_acc = (
            -3 * self.g / (2 * self.l) * np.sin(self.theta)
            + 3 / (self.m * self.l ** 2) * torque
        )
        
        self.theta_dot = self.theta_dot + theta_acc * self.dt
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)
        self.theta = self.theta + self.theta_dot * self.dt
        
        # Normalize angle to [-π, π]
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi
        
        self.step_count += 1
        
        # Reward: upright position, low velocity, low torque
        # Angle cost: θ=0 is upright
        angle_cost = self.theta ** 2
        velocity_cost = 0.1 * self.theta_dot ** 2
        torque_cost = 0.001 * torque ** 2
        
        reward = -(angle_cost + velocity_cost + torque_cost)
        
        # No termination (infinite horizon like standard Pendulum)
        terminated = False
        truncated = self.step_count >= 200
        
        info = {
            "angle": self.theta,
            "velocity": self.theta_dot,
            "torque": torque,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            np.cos(self.theta),
            np.sin(self.theta),
            self.theta_dot,
        ], dtype=np.float32)
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self) -> np.ndarray:
        """Render pendulum as RGB array."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            
            # Pendulum rod
            x = self.l * np.sin(self.theta)
            y = -self.l * np.cos(self.theta)
            ax.plot([0, x], [0, y], 'b-', linewidth=3)
            
            # Pendulum bob
            ax.plot(x, y, 'ro', markersize=20)
            
            # Pivot
            ax.plot(0, 0, 'ko', markersize=10)
            
            # Target (upright)
            ax.axhline(y=self.l, color='g', linestyle='--', alpha=0.5)
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return frame
            
        except ImportError:
            return np.zeros((256, 256, 3), dtype=np.uint8)
