"""
Field Navigation Environment.

A 2D navigation task where the agent learns to generate
flow fields that guide it to a goal. This is the canonical
environment for demonstrating GRL's visual appeal.

The agent controls a point mass in 2D by emitting a velocity
field. The field is evaluated at the current position to
determine the next position.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FieldNavigationEnv(gym.Env):
    """
    2D field navigation environment.
    
    The agent starts at a random position and must reach a goal.
    Instead of discrete actions, the agent outputs a vector field
    that is evaluated at the current position.
    
    State: [x, y, goal_x, goal_y] (4D)
    Action: [vx, vy] velocity vector (2D)
    Reward: -distance to goal, +100 for reaching goal
    
    The key difference from standard navigation is that GRL
    policies learn a *field* F(x, y) that maps any position
    to a velocity, not just the current position.
    
    Args:
        arena_size: Size of the square arena
        goal_radius: Radius for goal reaching
        max_steps: Maximum steps per episode
        dt: Time step for integration
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        arena_size: float = 10.0,
        goal_radius: float = 0.5,
        max_steps: int = 200,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.arena_size = arena_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.dt = dt
        self.render_mode = render_mode
        
        # State: [x, y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=-arena_size,
            high=arena_size,
            shape=(4,),
            dtype=np.float32,
        )
        
        # Action: velocity vector [vx, vy]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        
        self.position = None
        self.goal = None
        self.step_count = 0
        self.trajectory = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to random start and goal."""
        super().reset(seed=seed)
        
        # Random start position
        self.position = self.np_random.uniform(
            -self.arena_size * 0.8,
            self.arena_size * 0.8,
            size=(2,),
        ).astype(np.float32)
        
        # Random goal (ensure some distance from start)
        while True:
            self.goal = self.np_random.uniform(
                -self.arena_size * 0.8,
                self.arena_size * 0.8,
                size=(2,),
            ).astype(np.float32)
            if np.linalg.norm(self.goal - self.position) > 2.0:
                break
        
        self.step_count = 0
        self.trajectory = [self.position.copy()]
        
        return self._get_obs(), {"trajectory": self.trajectory}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step using the velocity action.
        
        Args:
            action: Velocity vector [vx, vy]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Update position with velocity
        self.position = self.position + action * self.dt
        
        # Clip to arena
        self.position = np.clip(
            self.position,
            -self.arena_size,
            self.arena_size,
        )
        
        self.trajectory.append(self.position.copy())
        self.step_count += 1
        
        # Compute reward
        distance = np.linalg.norm(self.position - self.goal)
        reward = -distance * 0.01  # Shaping reward
        
        # Check termination
        reached_goal = distance < self.goal_radius
        if reached_goal:
            reward += 100.0
        
        terminated = reached_goal
        truncated = self.step_count >= self.max_steps
        
        info = {
            "distance": distance,
            "reached_goal": reached_goal,
            "trajectory": self.trajectory,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([self.position, self.goal]).astype(np.float32)
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            # Implement pygame rendering if needed
            pass
    
    def _render_frame(self) -> np.ndarray:
        """Render a frame as RGB array."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Draw arena boundary
            ax.set_xlim(-self.arena_size, self.arena_size)
            ax.set_ylim(-self.arena_size, self.arena_size)
            ax.set_aspect('equal')
            
            # Draw trajectory
            if len(self.trajectory) > 1:
                traj = np.array(self.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=2)
            
            # Draw goal
            goal_circle = plt.Circle(
                self.goal, self.goal_radius, 
                color='green', alpha=0.5
            )
            ax.add_patch(goal_circle)
            
            # Draw agent
            ax.plot(
                self.position[0], self.position[1], 
                'ro', markersize=10
            )
            
            # Convert to array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return frame
            
        except ImportError:
            return np.zeros((256, 256, 3), dtype=np.uint8)


class MultiGoalNavigationEnv(FieldNavigationEnv):
    """
    Navigation with multiple goals and obstacles.
    
    Extends the basic navigation to include:
    - Multiple goal positions
    - Obstacle regions to avoid
    - Curriculum over difficulty
    """
    
    def __init__(
        self,
        num_goals: int = 3,
        num_obstacles: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        
        # Extend observation space
        # [position, goals..., obstacles...]
        obs_dim = 2 + num_goals * 2 + num_obstacles * 3  # obstacle: x, y, radius
        self.observation_space = spaces.Box(
            low=-self.arena_size,
            high=self.arena_size,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        self.goals = []
        self.obstacles = []
        self.reached_goals = set()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        # Generate multiple goals
        self.goals = []
        for _ in range(self.num_goals):
            goal = self.np_random.uniform(
                -self.arena_size * 0.7,
                self.arena_size * 0.7,
                size=(2,),
            ).astype(np.float32)
            self.goals.append(goal)
        
        # Generate obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = self.np_random.uniform(
                -self.arena_size * 0.5,
                self.arena_size * 0.5,
                size=(2,),
            ).astype(np.float32)
            radius = self.np_random.uniform(0.5, 1.5)
            self.obstacles.append((pos, radius))
        
        self.reached_goals = set()
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = [self.position]
        for goal in self.goals:
            obs.append(goal)
        for pos, radius in self.obstacles:
            obs.append(np.concatenate([pos, [radius]]))
        return np.concatenate(obs).astype(np.float32)
