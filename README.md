# GeoGuessrCV

Country-Level Image Geolocation with CNNs and Vision Transformers

## Overview

This project tackles the task of predicting a photograph's country of origin using only pixel information. The goal is to build accurate models that can identify the country where a photo was taken, which has applications in fact-checking, humanitarian response, and travel identification.

## Models

The project implements and compares two different architectures:

1. **ResNet-50 Baseline**: A CNN-based approach that achieves around 41.3% top-1 accuracy and 59.2% top-5 accuracy on the validation set.

2. **Vision Transformer (ViT)**: A pure-attention architecture using 16×16 patches that captures global scene cues better than ResNet, leading to improved performance.

## Dataset

The dataset consists of ~50,000 Street View images with country labels, split 80/10/10 (train/val/test) in a stratified manner to ensure every country appears in all splits.

## Usage

### Training the ViT Model

```bash
python train_geo_vit.py \
    --root /path/to/dataset/directory \
    --epochs 3 \
    --batch-size 32 \
    --lr 1e-5 \
    --weight-decay 0.05 \
    --layer-wise-decay 0.75
```

Key parameters:

- `--batch-size`: 16-64 depending on memory (default: 32)
- `--lr`: Base learning rate, typically 1e-5 to 5e-5 (default: 1e-5)
- `--layer-wise-decay`: Factor for layer-wise learning rate decay (0.65-0.8 recommended, default: 0.75)
- `--no-layer-wise-lr`: Flag to disable layer-wise learning rate decay
- `--weight-decay`: Weight decay value (default: 0.05)

### Jupyter Notebook

For a more interactive experience with visualizations, use the provided notebook:

```
vit_geoguessr.ipynb
```

This notebook demonstrates:
- Dataset exploration
- Model training
- Evaluation and visualization
- Performance analysis
- Comparison with ResNet baseline

## Model Architecture

The Vision Transformer (ViT) model uses:
- google/vit-base-patch16-224 checkpoint (ImageNet pre-trained)
- 16×16 patches
- Pure attention encoder
- Global self-attention for capturing scene-level context

## Performance Optimization

ViT models benefit from several training optimizations:
- Layer-wise learning rate decay (0.65-0.8)
- Lower base learning rate than CNNs (1e-5 to 5e-5)
- Mixed-precision training (fp16=True)
- Weight decay in the 0.01-0.1 range

## Requirements

- transformers>=4.40.0
- torch
- torchvision
- datasets
- evaluate
- matplotlib (for visualization)
- seaborn (for visualization)
- scikit-learn (for metrics)
