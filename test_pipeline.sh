#!/bin/bash
# Complete pipeline test with tiny dataset
# This script tests the entire workflow without needing the full DOTA dataset

set -e  # Exit on error

echo "======================================================"
echo "DeepSat-VL Pipeline Test (Tiny Dataset)"
echo "======================================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: Virtual environment not activated"
    echo "Activating venv..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "Error: venv not found. Run ./setup_venv.sh first"
        exit 1
    fi
fi
echo ""

# Step 1: Test MPS
echo "Step 1/6: Testing MPS availability..."
python test_mps.py
echo ""

# Step 2: Create tiny dataset
echo "Step 2/6: Creating tiny synthetic dataset..."
python create_tiny_dataset.py \
    --output data/tiny_DOTA \
    --train 20 \
    --val 5 \
    --size 1024
echo ""

# Step 3: Verify dataset
echo "Step 3/6: Verifying dataset structure..."
if [ -d "data/tiny_DOTA/images" ]; then
    echo "✓ Images directory exists"
    IMAGE_COUNT=$(ls data/tiny_DOTA/images/*.jpg 2>/dev/null | wc -l)
    echo "  Found $IMAGE_COUNT images"
else
    echo "✗ Images directory not found"
    exit 1
fi

if [ -f "data/tiny_DOTA/annotations_train.json" ]; then
    echo "✓ Training annotations exist"
else
    echo "✗ Training annotations not found"
    exit 1
fi

if [ -f "data/tiny_DOTA/annotations_val.json" ]; then
    echo "✓ Validation annotations exist"
else
    echo "✗ Validation annotations not found"
    exit 1
fi
echo ""

# Step 4: Train for a few epochs
echo "Step 4/6: Training model (5 epochs on tiny dataset)..."
echo "This will take 5-15 minutes depending on your Mac..."
python train.py --config configs/tiny_test.yaml
echo ""

# Step 5: Evaluate
echo "Step 5/6: Evaluating model..."
if [ -f "outputs/tiny_test/best.pt" ]; then
    python eval.py \
        --config configs/tiny_test.yaml \
        --checkpoint outputs/tiny_test/best.pt
else
    echo "Warning: best.pt not found, using latest.pt"
    python eval.py \
        --config configs/tiny_test.yaml \
        --checkpoint outputs/tiny_test/latest.pt
fi
echo ""

# Step 6: Visualize
echo "Step 6/6: Generating visualizations..."
if [ -f "outputs/tiny_test/best.pt" ]; then
    CKPT="outputs/tiny_test/best.pt"
else
    CKPT="outputs/tiny_test/latest.pt"
fi

python scripts/visualize_orientation.py \
    --config configs/tiny_test.yaml \
    --checkpoint "$CKPT" \
    --split val \
    --limit 5 \
    --out outputs/tiny_vis

echo ""
echo "======================================================"
echo "Pipeline Test Complete!"
echo "======================================================"
echo ""
echo "Results:"
echo "  - Model checkpoint: outputs/tiny_test/"
echo "  - Visualizations: outputs/tiny_vis/"
echo ""
echo "To view visualizations:"
echo "  open outputs/tiny_vis/"
echo ""
echo "Next steps:"
echo "  1. Download the full DOTA dataset (see DATASET_GUIDE.md)"
echo "  2. Train on full data: python train.py --config configs/dota_vitb_m1.yaml"
echo ""
