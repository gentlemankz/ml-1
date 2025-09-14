# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for car damage detection and cleanliness assessment, built for the inDrive hackathon. The system uses multi-task learning to evaluate car condition from photos, determining both cleanliness levels (0.0-1.0) and damage assessment (0.0-1.0) along with weather condition classification.

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

### GPU Training
The project is optimized for GPU training with mixed precision support using torch.cuda.amp. Training scripts automatically detect available hardware and adjust batch sizes accordingly.

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
```bash
# Basic training
python train.py --model_name tf_efficientnetv2_s --batch_size 32

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 train.py --distributed

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pth

# Custom dataset path
python train.py --csv_file path/to/dataset.csv
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

## Performance Notes

- Uses TensorFlow EfficientNetV2-S for optimal accuracy/speed trade-off
- Input size: 384x384 pixels
- Batch size automatically adjusts based on available GPU memory
- Mixed precision training provides ~2x speedup on modern GPUs
- Distributed training scales linearly across GPUs

## Troubleshooting

- **CUDA out of memory**: Reduce batch_size in training args
- **Dataset loading errors**: Check image paths in CSV file
- **Distributed training issues**: Ensure NCCL backend is available
- **Missing dependencies**: Run `pip install -r requirements.txt`