# Installation Guide

## Prerequisites

- Python 3.10 or higher (< 3.13)
- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/)

### Supported Platforms

| Platform | GPU Acceleration | Notes |
|----------|------------------|-------|
| **macOS Apple Silicon** (M1/M2/M3) | MPS | Primary development platform |
| macOS Intel | CPU only | Supported but slower |
| Linux + NVIDIA GPU | CUDA | Recommended for heavy training |
| RunPods (A40/A100) | CUDA | For expensive experiments |

**Primary development environment**: MacBook Pro M1 (2020) with 16GB RAM

---

## Local Development Setup (macOS Apple Silicon)

### Option 1: Mamba/Conda (Recommended)

```bash
# Create the environment
mamba env create -f environment.yml

# Activate
mamba activate grl

# Install the package in development mode
pip install -e .
```

### Option 2: Poetry (for development)

```bash
# Create conda environment first (for PyTorch with MPS)
mamba create -n grl python=3.11 pytorch torchvision -c pytorch
mamba activate grl

# Install dependencies with Poetry
poetry install --with dev,experiment,operators,baselines
```

### Option 3: Pip only

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch (MPS-enabled for Apple Silicon)
pip install torch torchvision

# Install the package
pip install -e ".[dev]"
```

---

## Verifying Installation

### Automated Verification (Recommended)

Run the verification script to automatically detect your hardware and test the installation:

```bash
python scripts/verify_installation.py
```

This will:
- Detect available compute resources (CPU, CUDA GPU, or Apple MPS)
- Verify PyTorch installation
- Test GRL package imports
- Run basic operator tests on your device
- Provide a comprehensive status report

### Manual Verification

Alternatively, test manually:

```python
import grl
print(grl.__version__)

# Test operator creation
from grl.operators import AffineOperator
import torch

op = AffineOperator(state_dim=4)
state = torch.randn(1, 4)
next_state = op(state)
print(f"State shape: {state.shape} -> {next_state.shape}")

# Check available accelerator
device = (
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")
```

---

## Apple Silicon (MPS) Notes

PyTorch supports Metal Performance Shaders (MPS) on Apple Silicon for GPU acceleration:

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Use MPS device
device = torch.device("mps")
tensor = torch.randn(3, 3, device=device)
```

### Known Limitations

- Some operations may fall back to CPU
- Memory management differs from CUDA
- For large-scale training, consider RunPods (see below)

---

## RunPods Setup (For Heavy Training)

For expensive training runs requiring better GPU resources (A40, A100, etc.), use RunPods.

### Quick Setup

1. Deploy a GPU pod (A40 recommended for GRL experiments)
2. Follow the standard Miniforge installation:

```bash
# On the pod
cd /tmp
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
rm Miniforge3-$(uname)-$(uname -m).sh

# Initialize shell
~/miniforge3/bin/conda init bash
source ~/.bashrc

# Clone and setup GRL
cd /workspace
git clone https://github.com/YOUR_USERNAME/GRL.git
cd GRL

# Create environment with CUDA support
mamba env create -f environment.yml
mamba activate grl

# For CUDA, ensure PyTorch has CUDA support
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -e .
```

### Verify CUDA on RunPods

```bash
nvidia-smi  # Check GPU

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Reference Documentation

For detailed RunPods setup including SSH configuration, file sync, and troubleshooting, see:
- `genai-lab/docs/runpods/project_setup_on_new_pod.md`

---

## Optional Dependencies

### Differentiable Physics (Brax/JAX)

For physics-based experiments with differentiable simulation:

```bash
# On Apple Silicon
pip install jax jax-metal

# On Linux/CUDA
pip install jax jaxlib brax
```

### Neural Operators

Already included, but for the full suite:

```bash
pip install neuraloperator
```

---

## Development Setup

For contributing or extending GRL:

```bash
# Install with all development dependencies
poetry install --with dev

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
pytest tests/

# Run linting
ruff check src/
black --check src/
```

---

## Troubleshooting

### MPS Issues (Apple Silicon)

If MPS is not working:

```python
import torch

# Check if MPS is available
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True

# If False, ensure you have the latest PyTorch
# pip install --upgrade torch torchvision
```

Some operations may not be supported on MPS yet. If you encounter errors, fall back to CPU:

```python
device = torch.device("cpu")  # Fallback
```

### CUDA Issues (Linux/RunPods)

If PyTorch doesn't detect your GPU:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
pip uninstall torch torchvision
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Gymnasium Rendering

For environment visualization:

```bash
# macOS (usually works out of the box)
pip install pygame

# Linux
sudo apt-get install python3-opengl xvfb
```

### Memory Issues on M1 (16GB)

For large experiments on limited memory:

```python
# Use smaller batch sizes
batch_size = 32  # or 16

# Enable memory-efficient settings
torch.mps.empty_cache()  # Clear MPS cache periodically
```

Or consider using RunPods for memory-intensive training.

---

## Quick Reference

| Task | Command |
|------|---------|
| Create environment | `mamba env create -f environment.yml` |
| Activate | `mamba activate grl` |
| Install package | `pip install -e .` |
| Run tests | `pytest tests/` |
| Check device | `python -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')"` |
| Train agent | `python -m grl.workflows.train` |
