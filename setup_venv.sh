#!/bin/bash
# Quick setup script for Mac M1/M2 environment

set -e  # Exit on error

echo "======================================================"
echo "DeepSat-VL Mac M1/M2 Setup Script"
echo "======================================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.9 or later"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Package managers upgraded"
echo ""

# Install PyTorch
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio
echo "✓ PyTorch installed"
echo ""

# Install other requirements
echo "Installing project dependencies..."
pip install -r requirements.txt
echo "✓ All dependencies installed"
echo ""

# Test MPS availability
echo "======================================================"
echo "Testing MPS (Apple Silicon GPU) availability..."
echo "======================================================"
python3 test_mps.py

echo ""
echo "======================================================"
echo "Setup Complete!"
echo "======================================================"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Download and prepare the DOTA dataset (see SETUP_M1.md)"
echo "  2. Train the model: python train.py --config configs/dota_vitb_m1.yaml"
echo "  3. Evaluate: python eval.py --config configs/dota_vitb_m1.yaml --checkpoint outputs/dota_vitb_m1/best.pt"
echo ""
