# 🛰️ DeepSat-VL: Oriented Object Detection in Satellite Imagery

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black.svg)](https://www.apple.com/mac/)

A PyTorch-based transformer model for oriented object detection in satellite imagery. **Automatically supports NVIDIA CUDA GPUs, Apple Silicon (M1/M2/M3), and CPU.**

##  Key Features

- ** Oriented Object Detection**: DETR-style architecture with rotated bounding boxes
- ** Universal GPU Support**: Auto-detects CUDA → MPS → CPU
- ** Easy Setup**: Test with synthetic data before downloading 10GB+ dataset
- ** Multiple Backbones**: CLIP ViT, TIMM ViT, ResNet50
- ** 15 Object Classes**: Planes, ships, vehicles, buildings, and more
- ** Production Ready**: Training, evaluation, and visualization included

## Quick Start

### 1. Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/DeepSat-VL.git
cd DeepSat-VL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test (No Downloads!)

```bash
# Quick test with synthetic data (~10 min)
./test_pipeline.sh

# Or manually:
python create_tiny_dataset.py --output data/tiny_DOTA
python train.py --config configs/tiny_test.yaml
```

**Device will auto-select: CUDA > MPS > CPU** 

### 3.Get DOTA Dataset

- **Download**: https://captain-whu.github.io/DOTA/dataset.html
- Register and download DOTA v1.0 (train + val)

```bash
# Preprocess
python scripts/prepare_dota.py --src data/DOTA_raw/train --dst data/DOTA --split train
python scripts/prepare_dota.py --src data/DOTA_raw/val --dst data/DOTA --split val
```

### 4.Train

```bash
# Start training (device auto-detected)
python train.py --config configs/dota_vitb_baseline.yaml
```

### 5. Evaluate & Visualize

```bash
# Evaluate
python eval.py --config configs/dota_vitb_baseline.yaml --checkpoint outputs/dota_vitb_baseline/best.pt

# Visualize
python scripts/visualize_orientation.py --config configs/dota_vitb_baseline.yaml --checkpoint outputs/dota_vitb_baseline/best.pt
```

##  Architecture

```
Input Image (1024×1024)
    ↓
Vision Encoder (CLIP ViT / ResNet50)
    ↓
Transformer Encoder + Decoder
    ↓
Detection Heads:
  ├─ Classification (16 classes)
  └─ Bbox Regression (cx, cy, w, h, θ)
```

##  System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux / macOS 12.3+ / Windows 10+ | Linux / macOS 13+ |
| **Python** | 3.9+ | 3.10+ |
| **RAM** | 8GB | 16GB+ |
| **GPU** | Optional (CPU works) | NVIDIA (6GB+) or M1 Pro/Max |
| **Storage** | 30GB | 50GB+ SSD |

**GPU Support:**
-  NVIDIA CUDA GPUs (fastest, 5-10x faster than MPS)
-  Apple Silicon M1/M2/M3 (MPS backend)
-  CPU (works but slow)

## Configuration

| Config | Batch Size | Best For |
|--------|------------|----------|
| `tiny_test.yaml` | 2 | Quick testing |
| `dota_vitb_m1.yaml` | 2 | 8GB VRAM/RAM |
| `dota_vitb_baseline.yaml` | 4 | 12GB+ VRAM |

**Adjust batch size** based on your hardware in the config file.

##  Performance Benchmarks

**Training Speed (50 epochs, DOTA dataset):**

### NVIDIA GPUs (batch_size=4)

| GPU | VRAM | Time per Epoch | Total (50 epochs) |
|-----|------|----------------|-------------------|
| RTX 3060 | 12GB | ~2-3 min | ~2-3 hours |
| RTX 3080 | 10GB | ~1.5-2 min | ~1.5-2 hours |
| RTX 4090 | 24GB | ~1 min | ~1 hour |

### Apple Silicon (batch_size=4)

| Mac | Memory | Time per Epoch | Total (50 epochs) |
|-----|--------|----------------|-------------------|
| M1 | 8GB | ~10-12 min | ~8-10 hours |
| M1 Pro | 16GB | ~8-10 min | ~6-8 hours |
| M1 Max | 32GB | ~6-8 min | ~5-6 hours |

**Note:** CUDA is significantly faster for this workload.

##  Dataset: DOTA

- **Images**: 2,806 aerial images
- **Objects**: 188,282 instances
- **Classes**: 15 categories
- **Format**: Oriented bounding boxes (rotation angle)

**Classes**: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool

##  Tools & Scripts

```bash
# Training
python train.py --config <config_file>

# Evaluation
python eval.py --config <config> --checkpoint <checkpoint>

# Visualization
python scripts/visualize_orientation.py --config <config> --checkpoint <checkpoint>

# Dataset preparation
python scripts/prepare_dota.py --src <raw> --dst <processed> --split <train/val>

# Testing
python create_tiny_dataset.py --output data/tiny_DOTA
python test_mps.py  # Test GPU availability
```

##  Troubleshooting

### Device not detected

```bash
# Test GPU availability
python test_mps.py  # Mac
nvidia-smi  # NVIDIA
```

### Out of memory

Reduce batch size in config:
```yaml
train:
  batch_size: 1  # or 2
```

### Slow training

- Use smaller backbone: `backbone: "torchvision_resnet50"`
- Reduce model size: `enc_layers: 2`, `dec_layers: 2`

##  Citation

```bibtex
@software{DeepSatVL2024,
  title = {DeepSat-VL: Oriented Object Detection in Satellite Imagery},
  author = {Subodh khanger},
  year = {2024},
  url = {https://github.com/yourusername/DeepSat-VL}
}
```

**DOTA Dataset:**
```bibtex
@InProceedings{Xia_2018_CVPR,
  author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
  booktitle = {CVPR},
  year = {2018}
}
```



##  Acknowledgments

- [DOTA Dataset](https://captain-whu.github.io/DOTA/)
- [DETR](https://arxiv.org/abs/2005.12872)
- [CLIP](https://arxiv.org/abs/2103.00020)
- PyTorch & Apple for MPS backend

##  Links

- **Issues**: [GitHub Issues](https://github.com/yourusername/DeepSat-VL/issues)
- **DOTA Dataset**: https://captain-whu.github.io/DOTA/
- **PyTorch MPS**: https://pytorch.org/docs/stable/notes/mps.html


