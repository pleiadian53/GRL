# GRL Scripts

Utility scripts for training, evaluation, and analysis.

## Installation Verification

```bash
# Verify installation and detect compute resources
python scripts/verify_installation.py
```

This script automatically:
- Detects CPU, CUDA GPU, or Apple MPS
- Tests PyTorch installation
- Verifies GRL package
- Runs basic operator tests
- Provides detailed hardware info

## Training

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/pendulum.yaml

# Train with overrides
python scripts/train.py training.num_episodes=5000 training.least_action_weight=0.1
```

## Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py checkpoints/final_checkpoint.pt

# Evaluate with rendering
python scripts/evaluate.py checkpoints/final_checkpoint.pt --render
```

## Analysis

```bash
# Compare multiple runs
python scripts/compare_runs.py results/run1 results/run2 results/run3

# Generate paper figures
python scripts/generate_figures.py --checkpoint checkpoints/final_checkpoint.pt
```

## Baseline Comparison

```bash
# Train SAC baseline
python scripts/train_baseline.py --algo sac --env field_navigation

# Compare GRL vs baselines
python scripts/compare_with_baseline.py
```
