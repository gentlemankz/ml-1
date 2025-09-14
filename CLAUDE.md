# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **competition-grade machine learning project** for car damage detection and cleanliness assessment, built for the inDrive hackathon. The system uses advanced multi-task learning to evaluate car condition from photos, determining both cleanliness levels (0.0-1.0) and damage assessment (0.0-1.0) along with weather condition classification.

**🏆 OPTIMIZED FOR HIGH-END HARDWARE**: This project is specifically tuned for beast-mode setups like 3x RTX 5090 + 96 vCPU + 424GB RAM configurations.

## Development Environment Setup

### Dependencies Installation
```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch >= 2.1.0 with torchvision
- timm for model architectures
- albumentations for data augmentation
- wandb for experiment tracking
- accelerate for distributed training

### GPU Training & Hardware Optimization
The project features **intelligent hardware detection** and automatic configuration:
- **Multi-GPU support**: Optimized for 1-4 GPU setups with DistributedDataParallel
- **Mixed precision**: Supports both float16 and bfloat16 (RTX 5090 optimized)
- **Memory management**: Automatic batch size scaling based on available VRAM
- **CPU optimization**: Scales data loading workers based on CPU core count
- **torch.compile**: PyTorch 2.0+ optimization for 20% speedup

## Core Architecture

### Multi-task Model (`model.py`)
- **CarInspectionModel**: Main model class using TensorFlow EfficientNetV2-S backbone
- **Multi-task heads**: Separate heads for cleanliness regression, damage assessment, and weather classification
- **Custom loss function**: MultiTaskLoss combining regression and classification losses
- **Uncertainty quantification**: Built-in Monte Carlo Dropout for confidence estimation

### Dataset Pipeline (`dataset.py`)
- **CarInspectionDataset**: PyTorch dataset with albumentations transforms
- **Multi-task labels**: Handles cleanliness_score, damage_score, and weather_condition
- **Robust loading**: Fallback mechanisms for corrupted images
- **Train/val/test splits**: Automatic data splitting based on CSV split column

### Training Pipeline (`train.py`)
- **Distributed training**: Multi-GPU support with DistributedDataParallel
- **Mixed precision**: Automatic mixed precision for faster training
- **Advanced schedulers**: CosineAnnealingLR and ReduceLROnPlateau
- **Experiment tracking**: Weights & Biases integration
- **Checkpointing**: Automatic model saving and resuming

## Common Commands

### Training

#### 🚀 **Beast Mode Training** (3x RTX 5090 + High-end CPU)
```bash
# Fully automated training with optimal settings
./run_runpod.sh

# Manual high-performance training
torchrun --nproc_per_node=3 train.py \
    --model_name tf_efficientnetv2_s \
    --batch_size 96 \
    --input_size 512 \
    --num_workers 24 \
    --epochs 30 \
    --lr 2e-4 \
    --mixed_precision \
    --use_bfloat16 \
    --use_compile \
    --gradient_accumulation_steps 2
```

#### 🖥️ **Standard Training**
```bash
# Basic training
python train.py --model_name tf_efficientnetv2_s --batch_size 32

# Multi-GPU training (2-4 GPUs)
torchrun --nproc_per_node=4 train.py --batch_size 64

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pth

# Custom dataset path
python train.py --csv_file path/to/dataset.csv
```

#### ⚡ **Optimized Training Arguments**
```bash
# Memory-efficient training
python train.py \
    --model_name tf_efficientnetv2_s \
    --batch_size 32 \
    --input_size 384 \
    --mixed_precision \
    --gradient_accumulation_steps 4

# High-quality training (longer but better results)
python train.py \
    --model_name tf_efficientnetv2_s \
    --batch_size 48 \
    --input_size 512 \
    --epochs 50 \
    --lr 1e-4
```

### Data Preparation
```bash
# Convert COCO annotations to CSV format
python convert_annotations.py --input_dir datasets/coco-folders --output_file data/car_inspection_dataset.csv

# Analyze dataset statistics
python dataset.py --csv_file data/car_inspection_dataset.csv --analyze
```

### Cloud Training Setup
```bash
# RunPod setup
chmod +x run_runpod.sh
./run_runpod.sh

# Paperspace setup
python setup_paperspace.py
```

## Dataset Structure

The project expects a CSV file with columns:
- `image_path`: Full path to image file
- `cleanliness_score`: Float 0.0-1.0 (0=dirty, 1=clean)
- `damage_score`: Float 0.0-1.0 (0=damaged, 1=undamaged)
- `weather_condition`: Integer 0-3 (clear, rain, fog, snow)
- `split`: String (train/val/test)

## Model Outputs

The trained model produces three outputs:
1. **Cleanliness regression**: Continuous score 0.0-1.0
2. **Damage assessment**: Continuous score 0.0-1.0
3. **Weather classification**: 4-class probability distribution

## Key Features

- **Multi-task learning**: Jointly optimizes cleanliness, damage, and weather prediction
- **Uncertainty quantification**: Monte Carlo Dropout for prediction confidence
- **Data augmentation**: Comprehensive transforms including weather simulation
- **Distributed training**: Scales across multiple GPUs
- **Cloud ready**: Configured for RunPod, Paperspace, and Azure ML
- **Experiment tracking**: Full WandB integration with metrics, hyperparameters, and artifacts

## File Structure

- `train.py`: Main training script with distributed support
- `model.py`: Multi-task model architecture and loss functions
- `dataset.py`: Data loading and preprocessing pipeline
- `utils.py`: Utility functions for logging, checkpointing, metrics
- `convert_annotations.py`: COCO to CSV conversion utility
- `requirements.txt`: Python dependencies
- `run_runpod.sh`: RunPod training automation script

## Performance Notes & Hardware Configurations

### 🏆 **Beast Mode Setup** (3x RTX 5090 + 96 vCPU + 424GB RAM)
- **Batch size**: 96 (32 per GPU) - **effective batch size 192** with gradient accumulation
- **Input resolution**: 512x512 - **high quality** for competition
- **Training time**: ~45-60 minutes for 30 epochs
- **Memory usage**: ~25-28GB per GPU (safe margin)
- **Workers**: 24 (8 per GPU) - **maximizes 96 vCPU**
- **Precision**: bfloat16 - **RTX 5090 optimized**
- **Expected accuracy**: >94% with optimized hyperparameters

### 💪 **High-End Setup** (2x RTX 4090/5090)
- **Batch size**: 64 (32 per GPU) - effective batch size 128
- **Input resolution**: 384x384
- **Training time**: ~90-120 minutes
- **Memory usage**: ~20-24GB per GPU
- **Expected accuracy**: >92%

### 🖥️ **Standard Setup** (Single RTX 4080/4090)
- **Batch size**: 32-48 depending on VRAM
- **Input resolution**: 384x384
- **Training time**: ~2-3 hours
- **Memory usage**: ~16-20GB
- **Expected accuracy**: >90%

### 📊 **Model Architecture**
- **Backbone**: TensorFlow EfficientNetV2-S (86.5M parameters)
- **Multi-task heads**: Cleanliness regression + Damage assessment + Weather classification
- **Advanced features**: Monte Carlo Dropout, Temperature Scaling, Learnable Loss Weights
- **Optimization**: torch.compile + mixed precision + distributed training

## 🚀 Beast Mode Optimizations

### Hardware-Specific Configurations
The `run_runpod.sh` script automatically detects your hardware and applies optimal settings:

```bash
# 3x RTX 5090 + 96 vCPU + 424GB RAM Detection
- Batch size: 96 (32 per GPU)
- Input resolution: 512x512
- Workers: 24 (8 per GPU)
- Precision: bfloat16
- Effective batch size: 192 (with gradient accumulation)
- Training time: ~45-60 minutes
```

### Memory Management Strategy
- **Smart batch sizing**: Starts aggressive, falls back if OOM
- **Dynamic worker allocation**: Scales with CPU cores
- **Gradient accumulation**: Maintains large effective batch sizes safely
- **Mixed precision**: bfloat16 for RTX 5090, float16 for others

### Competition-Ready Features
- **High-resolution training**: 512x512 for better accuracy
- **Advanced augmentations**: Weather simulation, noise injection
- **Uncertainty quantification**: Monte Carlo Dropout for confidence
- **Model compilation**: torch.compile for 20% speedup
- **Learnable loss weights**: Automatic task balancing

## Troubleshooting & Common Issues

### 🚨 **Memory Issues**
- **CUDA out of memory**:
  - Reduce `--batch_size` (try 32, 24, 16)
  - Reduce `--input_size` (try 384, 320, 224)
  - Reduce `--num_workers` (try 8, 4)
  - Disable `--use_compile` if enabled
- **CPU RAM overflow**: Reduce `--num_workers` parameter
- **Triton compilation errors**: Disable `--use_compile` for older GPUs

### 🔧 **Training Issues**
- **NaN losses**:
  - Lower learning rate (`--lr 5e-5`)
  - Check data normalization in dataset.py
  - Reduce gradient accumulation steps
- **Slow convergence**:
  - Increase learning rate (`--lr 2e-4`)
  - Enable `--use_compile` for modern GPUs
  - Increase effective batch size via gradient accumulation
- **Model not improving**:
  - Check dataset balance and quality
  - Verify data augmentation isn't too aggressive
  - Try different model architectures

### 🌐 **Distributed Training Issues**
- **NCCL errors**:
  - Check GPU visibility: `nvidia-smi`
  - Verify CUDA/driver compatibility
  - Use single GPU training as fallback
- **Port conflicts**: Change distributed port in torchrun command
- **Synchronization issues**: Ensure all GPUs have same CUDA version

### 📊 **Dataset Issues**
- **Loading errors**:
  - Check image paths are absolute in CSV file
  - Verify all images exist and are readable
  - Check file permissions in dataset directories
- **Label inconsistencies**: Run `python dataset.py` for validation
- **Poor performance**: Verify train/val/test splits are balanced

### ⚡ **Performance Optimization**
- **Slow data loading**:
  - Increase `--num_workers` (match CPU cores)
  - Enable `pin_memory=True` in dataloaders
  - Use SSD storage for dataset
- **Low GPU utilization**:
  - Increase batch size until memory limit
  - Enable `--use_compile` for PyTorch 2.0+
  - Check CPU bottlenecks with `htop`

### 🔍 **Quick Diagnostics**
```bash
# Test single batch training
python train.py --epochs 1 --batch_size 8

# Test model creation
python model.py

# Test dataset loading
python dataset.py

# Check GPU status
nvidia-smi
```