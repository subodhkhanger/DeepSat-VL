#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) is available and working
on Mac M1/M2/M3 systems.
"""

import torch
import sys


def test_mps():
    print("=" * 60)
    print("Testing MPS (Apple Silicon GPU) Support")
    print("=" * 60)
    print()

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()

    print(f"MPS built into PyTorch: {mps_built}")
    print(f"MPS available on system: {mps_available}")
    print()

    if not mps_built:
        print("✗ MPS is not built into this PyTorch installation.")
        print("  Please install PyTorch with MPS support:")
        print("  pip install torch torchvision torchaudio")
        return False

    if not mps_available:
        print("✗ MPS is not available on this system.")
        print("  Possible reasons:")
        print("  - macOS version is older than 12.3")
        print("  - Not running on Apple Silicon (M1/M2/M3)")
        return False

    print("✓ MPS is available!")
    print()

    # Test basic operations
    print("Testing basic tensor operations on MPS...")
    try:
        device = torch.device("mps")

        # Create tensor
        x = torch.ones(5, 5, device=device)
        print(f"✓ Created tensor on MPS: shape {x.shape}")

        # Basic math
        y = x + 2
        print(f"✓ Addition: {y[0, 0].item()}")

        # Matrix multiplication
        z = torch.mm(x, x)
        print(f"✓ Matrix multiplication: shape {z.shape}")

        # Move between devices
        cpu_tensor = z.cpu()
        print(f"✓ Moved tensor to CPU: {cpu_tensor.device}")

        mps_tensor = cpu_tensor.to(device)
        print(f"✓ Moved tensor back to MPS: {mps_tensor.device}")

        print()
        print("=" * 60)
        print("All tests passed! MPS is working correctly.")
        print("=" * 60)
        print()
        print("You can now use MPS for GPU acceleration:")
        print("  device = torch.device('mps')")
        print("  model = model.to(device)")
        print("  data = data.to(device)")
        print()
        return True

    except Exception as e:
        print(f"✗ Error during MPS testing: {e}")
        print("  MPS may not be fully functional on your system.")
        return False


def print_system_info():
    """Print system and device information"""
    print("System Information:")
    print("-" * 60)

    # Available devices
    print("Available devices:")
    print(f"  - CPU: ✓")
    if torch.cuda.is_available():
        print(f"  - CUDA: ✓ ({torch.cuda.device_count()} device(s))")
    else:
        print(f"  - CUDA: ✗")
    if torch.backends.mps.is_available():
        print(f"  - MPS: ✓")
    else:
        print(f"  - MPS: ✗")
    print()


if __name__ == "__main__":
    print_system_info()
    success = test_mps()
    sys.exit(0 if success else 1)
