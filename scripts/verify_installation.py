#!/usr/bin/env python
"""
GRL Installation Verification Script

Automatically detects available compute resources (CPU, CUDA, MPS)
and verifies that the GRL package is correctly installed.

Usage:
    python scripts/verify_installation.py
    # or simply
    python -m scripts.verify_installation
"""

import sys
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    compatible = version.major == 3 and 10 <= version.minor < 13
    
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if compatible:
        return True, f"✓ Python {version_str}"
    else:
        return False, f"✗ Python {version_str} (requires 3.10-3.12)"


def check_pytorch() -> Tuple[bool, str, Dict]:
    """Check PyTorch installation and available devices."""
    try:
        import torch
        
        info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        }
        
        # Determine primary device
        if info["cuda_available"]:
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            info["device_name"] = device_name
        elif info["mps_available"]:
            device = "mps"
            device_name = "Apple Silicon GPU"
            info["device_name"] = device_name
        else:
            device = "cpu"
            device_name = "CPU only"
            info["device_name"] = device_name
        
        info["device"] = device
        
        return True, f"✓ PyTorch {info['version']}", info
    
    except ImportError:
        return False, "✗ PyTorch not installed", {}


def check_grl_package() -> Tuple[bool, str]:
    """Check if GRL package is installed."""
    try:
        import grl
        return True, f"✓ GRL {grl.__version__}"
    except ImportError:
        return False, "✗ GRL package not found (run: pip install -e .)"


def check_grl_operators() -> Tuple[bool, str, List[str]]:
    """Check if GRL operators can be imported and created."""
    try:
        from grl.operators import (
            AffineOperator,
            FieldOperator,
            KernelOperator,
        )
        
        # Test operator creation
        operators = []
        
        try:
            op = AffineOperator(state_dim=4)
            operators.append("AffineOperator")
        except Exception as e:
            return False, f"✗ AffineOperator failed: {e}", []
        
        try:
            op = FieldOperator(state_dim=4)
            operators.append("FieldOperator")
        except Exception as e:
            return False, f"✗ FieldOperator failed: {e}", []
        
        try:
            op = KernelOperator(state_dim=4)
            operators.append("KernelOperator")
        except Exception as e:
            return False, f"✗ KernelOperator failed: {e}", []
        
        return True, f"✓ Operators: {', '.join(operators)}", operators
    
    except ImportError as e:
        return False, f"✗ Failed to import operators: {e}", []


def check_grl_environments() -> Tuple[bool, str, List[str]]:
    """Check if GRL environments can be imported."""
    try:
        from grl.envs import FieldNavigationEnv, OperatorPendulumEnv
        
        envs = ["FieldNavigationEnv", "OperatorPendulumEnv"]
        return True, f"✓ Environments: {', '.join(envs)}", envs
    
    except ImportError as e:
        return False, f"✗ Failed to import environments: {e}", []


def test_device_tensor_operations(device: str) -> Tuple[bool, str]:
    """Test basic tensor operations on the detected device."""
    try:
        import torch
        
        # Create tensors on device (with gradients enabled)
        device_obj = torch.device(device)
        x = torch.randn(10, 10, device=device_obj, requires_grad=True)
        y = torch.randn(10, 10, device=device_obj, requires_grad=True)
        
        # Test operations
        z = x @ y  # Matrix multiplication
        loss = z.mean()
        loss.backward()  # Test backprop
        
        # Verify gradients exist
        assert x.grad is not None, "Gradient computation failed"
        assert y.grad is not None, "Gradient computation failed"
        
        return True, f"✓ Tensor operations on {device.upper()}"
    
    except Exception as e:
        return False, f"✗ Tensor operations failed: {e}"


def test_operator_on_device(device: str) -> Tuple[bool, str]:
    """Test creating and running a GRL operator on the detected device."""
    try:
        import torch
        from grl.operators import FieldOperator
        
        device_obj = torch.device(device)
        
        # Create operator and move to device
        operator = FieldOperator(state_dim=4).to(device_obj)
        
        # Test forward pass
        state = torch.randn(8, 4, device=device_obj)
        next_state = operator(state)
        
        # Test energy computation
        energy = operator.energy()
        
        assert next_state.shape == (8, 4), "Output shape mismatch"
        assert next_state.device.type == device, "Device mismatch"
        
        return True, f"✓ Operator execution on {device.upper()}"
    
    except Exception as e:
        return False, f"✗ Operator test failed: {e}"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def print_device_details(info: Dict):
    """Print detailed device information."""
    print("\n📊 Compute Resources:")
    print(f"   Primary Device: {info['device'].upper()}")
    print(f"   Device Name:    {info['device_name']}")
    
    if info['cuda_available']:
        import torch
        print(f"\n   CUDA Details:")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - GPU Count:    {torch.cuda.device_count()}")
        print(f"   - GPU 0:        {torch.cuda.get_device_name(0)}")
        
        # Memory info
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)  # GB
        print(f"   - Memory:       {total_memory:.1f} GB")
    
    elif info['mps_available']:
        print(f"\n   MPS Details:")
        print(f"   - MPS Built:    {info['mps_built']}")
        print(f"   - Platform:     Apple Silicon")
    
    else:
        print(f"\n   CPU Mode:")
        print(f"   - Note: No GPU acceleration available")
        print(f"   - Consider using RunPods for heavy training")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("  GRL Installation Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Check Python version
    print_section("Python Environment")
    passed, msg = check_python_version()
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    # Check PyTorch
    print_section("PyTorch")
    passed, msg, info = check_pytorch()
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    if passed:
        print_device_details(info)
        device = info["device"]
    else:
        print("\n❌ Cannot proceed without PyTorch")
        sys.exit(1)
    
    # Check GRL package
    print_section("GRL Package")
    passed, msg = check_grl_package()
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    if not passed:
        print("\n❌ GRL package not installed")
        print("\nTo install:")
        print("  cd /Users/pleiadian53/work/GRL")
        print("  pip install -e .")
        sys.exit(1)
    
    # Check operators
    passed, msg, operators = check_grl_operators()
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    # Check environments
    passed, msg, envs = check_grl_environments()
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    # Test tensor operations
    print_section("Device Testing")
    passed, msg = test_device_tensor_operations(device)
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    # Test operator on device
    passed, msg = test_operator_on_device(device)
    print(f"  {msg}")
    all_passed = all_passed and passed
    
    # Final summary
    print_section("Summary")
    if all_passed:
        print("  ✅ All checks passed!")
        print(f"\n  Device Configuration:")
        print(f"     Primary: {device.upper()}")
        print(f"     Status:  Ready for training")
        
        print(f"\n  Quick Start:")
        print(f"     python -m grl.workflows.train --episodes 100")
    else:
        print("  ⚠️  Some checks failed")
        print("\n  Please review the errors above and:")
        print("     1. Ensure all dependencies are installed")
        print("     2. Check INSTALL.md for troubleshooting")
        print("     3. Run: pip install -e .")
    
    print("\n" + "=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
