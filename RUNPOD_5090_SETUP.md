# 🚀 RunPod 3x RTX 5090 Training Setup

## Quick Start Guide

### 1. Create RunPod Pod
1. Go to [RunPod.io](https://runpod.io) → **Pods** → **Deploy**
2. **Template**: Select "PyTorch 2.8 + CUDA 12.x"
3. **GPU**: Choose "3x RTX 5090 (96GB VRAM)"
4. **Storage**: Add 100GB+ persistent volume
5. **Pricing**: Select "On-Demand" for uninterrupted training
6. **Deploy** (takes 30-60 seconds)

### 2. Setup Project
```bash
# Connect via SSH or Web Terminal
git clone https://github.com/your-username/selectra-decentrathon-ai-inDrive.git
cd selectra-decentrathon-ai-inDrive/ml

# Make script executable
chmod +x run_runpod.sh
```

### 3. Start Training
```bash
# Run the automated training pipeline
./run_runpod.sh
```

## What the Script Does

The `run_runpod.sh` script automatically:

### 🔍 **GPU Detection**
- Auto-detects 3x RTX 5090 setup
- Optimizes settings for 96GB total VRAM
- Configures distributed training

### ⚡ **Optimized Settings for 3x RTX 5090**
- **Batch Size**: 192 (64 per GPU)
- **Resolution**: 512px (higher quality)
- **Workers**: 24 (8 per GPU)
- **Epochs**: 20 (faster with large batches)
- **Learning Rate**: 3e-4 (scaled for large batch)

### 🏃‍♂️ **Training Pipeline**
1. Environment setup
2. Data conversion (`convert_annotations.py`)
3. Multi-GPU distributed training
4. Model export for inference
5. Results summary

## Expected Performance

### 💨 **Speed**
- **Training Time**: 15-30 minutes (vs 2-4 hours single GPU)
- **Throughput**: ~8-10x faster than RTX 4090
- **Batch Processing**: 192 images simultaneously

### 💰 **Cost**
- **Pod Cost**: ~$15-20/hour
- **Training Duration**: 20-30 minutes
- **Total Cost**: ~$5-10 per training run

### 📊 **Quality**
- **Higher Resolution**: 512px vs 384px
- **Larger Batches**: Better gradient estimates
- **Faster Convergence**: Fewer epochs needed

## Monitoring

### 📈 **Weights & Biases**
- Real-time training metrics
- GPU utilization monitoring
- Loss curves and validation scores

### 🖥️ **RunPod Console**
- GPU memory usage
- Pod performance metrics
- Cost tracking

## Advanced Configuration

### Custom Hyperparameters
Modify these lines in `run_runpod.sh` if needed:

```bash
# For even larger batches (if you have enough data)
BATCH_SIZE=256  # Line 100

# For ultra-high resolution training
INPUT_SIZE=640  # Line 101

# For longer training
EPOCHS=30       # Line 102
```

### Memory Optimization
The script automatically:
- Enables mixed precision (50% memory savings)
- Optimizes data loading workers
- Uses gradient checkpointing for large models

## Troubleshooting

### 🚨 **Common Issues**
- **CUDA OOM**: Reduce batch size to 128 or 64
- **Data loading slow**: Increase persistent storage
- **Network timeout**: Use RunPod's fast internet regions

### 🔧 **Debug Mode**
```bash
# Run with debugging
CUDA_LAUNCH_BLOCKING=1 ./run_runpod.sh
```

## Results

After training completes, you'll find:
- **Best Model**: `models/*/best_model.pth`
- **Inference Model**: `models/*_inference.pth`
- **Training Logs**: W&B dashboard
- **Performance Stats**: Terminal output

## Next Steps

1. **Download Model**: Use RunPod's file manager
2. **Deploy**: Use the inference model for production
3. **Fine-tune**: Adjust hyperparameters for your dataset
4. **Scale**: Try multi-node training for even larger models

---
*Generated for 3x RTX 5090 RunPod setup - Expected 10x training speedup!* 🎯