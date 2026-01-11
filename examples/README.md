# GRL Examples

Standalone, runnable examples demonstrating GRL capabilities.

## Quick Start Examples

### 01_basic_operator.py
```bash
python examples/01_basic_operator.py
```
Creates and applies different operator types to visualize their behavior.

### 02_field_navigation.py
```bash
python examples/02_field_navigation.py
```
Trains a GRL agent on 2D navigation and visualizes the learned field.

### 03_pendulum_control.py
```bash
python examples/03_pendulum_control.py
```
Operator-based pendulum control with torque field visualization.

## Advanced Examples

### 04_custom_operator.py
Demonstrates how to define custom operator types for specific domains.

### 05_baseline_comparison.py
Compares GRL with SAC on the same environment, plotting:
- Learning curves
- Trajectory smoothness
- Final performance

## Running All Examples

```bash
cd GRL
mamba activate grl
python -m pytest examples/ --doctest-modules
```
