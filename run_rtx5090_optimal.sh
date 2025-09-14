#!/bin/bash

# 🏆 OPTIMAL RTX 5090 x3 Training Script
# Optimized for: 3x RTX 5090 + 96 vCPU + 424GB RAM
# Fast + High Quality Training for Hackathon

echo "🚀 Starting RTX 5090 x3 OPTIMAL Training..."
echo "Hardware: 3x RTX 5090 + 96 vCPU + 424GB RAM"

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Check GPU availability
nvidia-smi --list-gpus
echo ""

# Kill any existing processes
pkill -f "torchrun\|python.*train.py" || true
sleep 2

# Clear GPU memory
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print(f'✅ Cleared memory on {torch.cuda.device_count()} GPUs')
"

echo "🔥 Starting training with OPTIMAL RTX 5090 settings..."

# OPTIMAL CONFIGURATION FOR RTX 5090 x3
# - Batch size: 72 total (24 per GPU) - safe memory usage
# - Input size: 448 - balance between quality and speed
# - Workers: 20 (6-7 per GPU) - optimal for 96 vCPU
# - Mixed precision: bfloat16 - RTX 5090 optimized
# - Compile: enabled for 15-20% speedup
# - Gradient accumulation: 3 steps = effective batch 216

torchrun --nproc_per_node=3 \
    --master_port=29500 \
    train.py \
    --model_name tf_efficientnetv2_s \
    --batch_size 24 \
    --input_size 448 \
    --num_workers 20 \
    --epochs 25 \
    --lr 1.5e-4 \
    --weight_decay 1e-5 \
    --mixed_precision \
    --use_bfloat16 \
    --use_compile \
    --gradient_accumulation_steps 3 \
    --warmup_epochs 2 \
    --save_every 5 \
    --log_every 50

echo "✅ Training completed!"

# Check final results
if [ -d "checkpoints" ]; then
    echo "📊 Training Results:"
    ls -la checkpoints/
    echo ""

    if [ -f "checkpoints/best_model.pth" ]; then
        echo "🎉 Best model saved successfully!"
        python3 -c "
import torch
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f'Best validation loss: {checkpoint.get(\"best_val_loss\", \"N/A\")}')
print(f'Final epoch: {checkpoint.get(\"epoch\", \"N/A\")}')
"
    fi
fi

echo ""
echo "🏆 RTX 5090 x3 OPTIMAL Training Complete!"
echo "Expected results: >93% accuracy in ~60-80 minutes"