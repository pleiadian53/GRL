"""
Visualization workflow for GRL.
"""

from pathlib import Path
from typing import Optional

import torch

from grl.eval.visualization import plot_operator_field, plot_trajectory
from grl.operators.field import FieldOperator
from grl.policies.policy import OperatorPolicy
from grl.utils.reproducibility import get_device


def visualize_operator(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "results/visualizations",
    xlim: tuple = (-5, 5),
    ylim: tuple = (-5, 5),
    resolution: int = 20,
):
    """
    Visualize the learned operator field.
    
    Args:
        checkpoint_path: Path to checkpoint (or None for random)
        output_dir: Directory to save visualizations
        xlim: X-axis limits
        ylim: Y-axis limits
        resolution: Grid resolution
    """
    import matplotlib.pyplot as plt
    
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create or load policy
    state_dim = 4  # Navigation env
    policy = OperatorPolicy(
        state_dim=state_dim,
        operator_class=FieldOperator,
        hidden_dims=[256, 256],
    ).to(device)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Using random (untrained) operator")
    
    # Get the operator
    operator = policy.get_operator()
    
    # Create a 2D wrapper for visualization
    class Operator2D:
        def __init__(self, full_operator, state_dim):
            self.full_operator = full_operator
            self.state_dim = state_dim
        
        def __call__(self, states_2d):
            # Pad to full state dim with zeros (goal position)
            batch_size = states_2d.shape[0]
            full_states = torch.zeros(batch_size, self.state_dim, device=states_2d.device)
            full_states[:, :2] = states_2d
            
            # Apply operator
            result = self.full_operator(full_states)
            return result[:, :2]  # Return only position
    
    op_2d = Operator2D(operator, state_dim)
    
    # Plot field
    fig, ax = plot_operator_field(
        op_2d,
        xlim=xlim,
        ylim=ylim,
        resolution=resolution,
        title="Learned Operator Field",
    )
    
    save_path = output_path / "operator_field.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved field visualization to {save_path}")
    
    return save_path


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize GRL operator")
    parser.add_argument("--checkpoint", help="Path to checkpoint file")
    parser.add_argument("--output-dir", default="results/visualizations", help="Output directory")
    parser.add_argument("--resolution", type=int, default=20, help="Grid resolution")
    
    args = parser.parse_args()
    
    visualize_operator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
