# GRL Setup on RunPods

Guide for setting up the GRL environment on a RunPods GPU pod for heavy training experiments.

**When to use RunPods:**
- Training with large operator networks
- PDE control experiments requiring significant compute
- Baseline comparisons with many episodes
- Experiments that exceed M1's 16GB memory

**GPU Recommendation**: A40 (48GB VRAM) for most GRL experiments

---

## Quick Setup

### Step 1: Install Miniforge

```bash
cd /tmp
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
rm Miniforge3-$(uname)-$(uname -m).sh

# Initialize shell
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Step 2: Clone and Setup GRL

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/GRL.git
cd GRL

# Create environment
mamba env create -f environment.yml
mamba activate grl

# Add CUDA support
mamba install pytorch-cuda=12.1 -c pytorch -c nvidia

# Install GRL
pip install -e .
```

### Step 3: Verify Setup

```bash
# Check GPU
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test GRL
python -c "import grl; print(f'GRL version: {grl.__version__}')"
```

---

## Running GRL Training on GPU

```bash
# Quick training test
python -m grl.workflows.train --episodes 100 --save-dir /workspace/checkpoints

# Full training run
python -m grl.workflows.train \
    --env field_navigation \
    --episodes 10000 \
    --save-dir /workspace/checkpoints/field_nav_run1

# Evaluate checkpoint
python -m grl.workflows.evaluate /workspace/checkpoints/final_checkpoint.pt
```

---

## Syncing Results Back to Local

From your local machine:

```bash
# Get results from pod
rsync -avz --progress root@POD_IP:/workspace/GRL/checkpoints/ ./checkpoints/ -e "ssh -p PORT"
rsync -avz --progress root@POD_IP:/workspace/GRL/results/ ./results/ -e "ssh -p PORT"
```

---

## Reference

For detailed RunPods setup including SSH configuration, see:
`genai-lab/docs/runpods/project_setup_on_new_pod.md`
