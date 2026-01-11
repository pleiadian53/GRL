"""
Visualization utilities for GRL.

Create compelling visualizations of:
- Operator fields (the core visual appeal of GRL)
- Agent trajectories
- Training progress
"""

from typing import Optional, Tuple

import numpy as np
import torch


def plot_operator_field(
    operator,
    xlim: Tuple[float, float] = (-5, 5),
    ylim: Tuple[float, float] = (-5, 5),
    resolution: int = 20,
    ax=None,
    show_magnitude: bool = True,
    title: str = "Operator Field",
):
    """
    Visualize an operator as a vector field.
    
    For 2D navigation, this shows arrows indicating how the
    operator transforms each point in space.
    
    Args:
        operator: An ActionOperator (must work on 2D states)
        xlim: X-axis limits
        ylim: Y-axis limits
        resolution: Grid resolution
        ax: Matplotlib axes (created if None)
        show_magnitude: Color arrows by magnitude
        title: Plot title
        
    Returns:
        Matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid to batch of states
    states = np.stack([X.flatten(), Y.flatten()], axis=1)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Apply operator
    with torch.no_grad():
        next_states = operator(states_tensor).numpy()
    
    # Compute displacements
    U = next_states[:, 0] - states[:, 0]
    V = next_states[:, 1] - states[:, 1]
    
    # Reshape
    U = U.reshape(resolution, resolution)
    V = V.reshape(resolution, resolution)
    
    # Compute magnitude for coloring
    magnitude = np.sqrt(U**2 + V**2)
    
    # Plot quiver
    if show_magnitude:
        quiver = ax.quiver(
            X, Y, U, V, magnitude,
            cmap='coolwarm',
            alpha=0.8,
        )
        plt.colorbar(quiver, ax=ax, label='Displacement magnitude')
    else:
        ax.quiver(X, Y, U, V, alpha=0.8)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    return fig, ax


def plot_trajectory(
    trajectory: np.ndarray,
    goal: Optional[np.ndarray] = None,
    obstacles: Optional[list] = None,
    ax=None,
    show_arrows: bool = True,
    title: str = "Agent Trajectory",
):
    """
    Plot an agent's trajectory through 2D space.
    
    Args:
        trajectory: Array of shape (T, 2) with positions
        goal: Optional goal position [x, y]
        obstacles: Optional list of (position, radius) tuples
        ax: Matplotlib axes
        show_arrows: Show direction arrows
        title: Plot title
        
    Returns:
        Figure and axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    trajectory = np.array(trajectory)
    
    # Plot trajectory line
    ax.plot(
        trajectory[:, 0], trajectory[:, 1],
        'b-', linewidth=2, alpha=0.7, label='Trajectory'
    )
    
    # Plot start and end
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=15, label='End')
    
    # Plot direction arrows
    if show_arrows and len(trajectory) > 5:
        skip = max(1, len(trajectory) // 10)
        for i in range(0, len(trajectory) - 1, skip):
            dx = trajectory[i + 1, 0] - trajectory[i, 0]
            dy = trajectory[i + 1, 1] - trajectory[i, 1]
            ax.arrow(
                trajectory[i, 0], trajectory[i, 1],
                dx * 0.8, dy * 0.8,
                head_width=0.2, head_length=0.1,
                fc='blue', ec='blue', alpha=0.5
            )
    
    # Plot goal
    if goal is not None:
        goal_circle = plt.Circle(goal, 0.5, color='green', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')
    
    # Plot obstacles
    if obstacles:
        for pos, radius in obstacles:
            obstacle = plt.Circle(pos, radius, color='red', alpha=0.3)
            ax.add_patch(obstacle)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_training_curves(
    metrics: dict,
    window: int = 100,
    ax=None,
    title: str = "Training Progress",
):
    """
    Plot training curves with smoothing.
    
    Args:
        metrics: Dict mapping metric names to lists of values
        window: Smoothing window size
        ax: Matplotlib axes (or list of axes for multiple metrics)
        title: Plot title
        
    Returns:
        Figure and axes
    """
    import matplotlib.pyplot as plt
    
    n_metrics = len(metrics)
    
    if ax is None:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
    else:
        fig = ax[0].figure if isinstance(ax, list) else ax.figure
        axes = ax if isinstance(ax, list) else [ax]
    
    def smooth(values, w):
        if len(values) < w:
            return values
        return np.convolve(values, np.ones(w) / w, mode='valid')
    
    for ax, (name, values) in zip(axes, metrics.items()):
        values = np.array(values)
        
        # Plot raw values with low alpha
        ax.plot(values, alpha=0.3, color='blue')
        
        # Plot smoothed values
        smoothed = smooth(values, window)
        x_smooth = np.arange(window - 1, len(values))
        ax.plot(x_smooth, smoothed, color='blue', linewidth=2, label=f'{name} (smoothed)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig, axes


def create_field_animation(
    operator,
    trajectory: np.ndarray,
    save_path: str,
    xlim: Tuple[float, float] = (-5, 5),
    ylim: Tuple[float, float] = (-5, 5),
    fps: int = 10,
):
    """
    Create an animation of the agent moving through the operator field.
    
    Args:
        operator: The field operator
        trajectory: Agent positions over time
        save_path: Path to save the animation
        xlim, ylim: Plot limits
        fps: Frames per second
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot field once
    plot_operator_field(operator, xlim, ylim, ax=ax, show_magnitude=True)
    
    # Initialize agent marker
    point, = ax.plot([], [], 'ro', markersize=15)
    trail, = ax.plot([], [], 'r-', linewidth=2, alpha=0.5)
    
    trajectory = np.array(trajectory)
    
    def init():
        point.set_data([], [])
        trail.set_data([], [])
        return point, trail
    
    def animate(i):
        point.set_data([trajectory[i, 0]], [trajectory[i, 1]])
        trail.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
        return point, trail
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(trajectory), interval=1000 // fps, blit=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    return save_path
