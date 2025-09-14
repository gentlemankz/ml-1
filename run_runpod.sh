#!/bin/bash

# RunPod training script for Car Inspection Model
# Usage: ./run_runpod.sh

echo "🚀 Starting RunPod Car Inspection Training Pipeline"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Make sure you're in the ML directory."
    print_status "Current directory: $(pwd)"
    print_status "Available files: $(ls -la)"
    exit 1
fi

# Step 1: Setup Environment
print_step "Setting up Paperspace environment..."
python3 setup_paperspace.py

if [ $? -ne 0 ]; then
    print_error "Environment setup failed!"
    exit 1
fi

print_status "Environment setup completed!"

# Step 2: Check GPU
print_step "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Step 3: Convert Annotations
print_step "Converting COCO annotations to regression labels..."
python3 convert_annotations.py

if [ $? -ne 0 ]; then
    print_error "Annotation conversion failed!"
    exit 1
fi

print_status "Data conversion completed!"

# Step 4: Test Dataset Loading
print_step "Testing dataset loading..."
python3 dataset.py

if [ $? -ne 0 ]; then
    print_error "Dataset loading test failed!"
    exit 1
fi

print_status "Dataset loading test passed!"

# Step 5: Test Model Creation
print_step "Testing model creation..."
python3 model.py

if [ $? -ne 0 ]; then
    print_error "Model creation test failed!"
    exit 1
fi

print_status "Model creation test passed!"

# Step 6: Start Training
print_step "Starting model training..."

# Training configuration for 3x RTX 5090 setup
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
TOTAL_MEMORY=$(python3 -c "import torch; print(sum([torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]))" 2>/dev/null || echo "0")

print_status "Detected $GPU_COUNT GPUs with total VRAM: $(($TOTAL_MEMORY / 1024 / 1024 / 1024))GB"

# Optimize batch size for multi-GPU setup
if [ $GPU_COUNT -ge 3 ] && [ $TOTAL_MEMORY -gt 80000000000 ]; then
    BATCH_SIZE=384  # 128 per GPU for 3x RTX 5090 (32GB each)
    INPUT_SIZE=512  # Higher resolution for better accuracy
    EPOCHS=40       # More epochs for competition quality
    LR=2e-4         # Optimized learning rate for larger batches
    GRAD_ACCUM=2    # Gradient accumulation for effective batch size 768
    print_status "3x RTX 5090 detected! Using optimized settings: batch_size=$BATCH_SIZE, input_size=$INPUT_SIZE"
elif [ $GPU_COUNT -eq 2 ]; then
    BATCH_SIZE=192  # 96 per GPU for 2 GPUs
    INPUT_SIZE=448
    EPOCHS=35
    LR=1.5e-4
    GRAD_ACCUM=2
    print_status "Dual GPU setup detected, using batch_size=$BATCH_SIZE"
elif [ $TOTAL_MEMORY -gt 20000000000 ]; then
    BATCH_SIZE=96   # Single high-memory GPU
    INPUT_SIZE=384
    EPOCHS=40
    LR=1e-4
    GRAD_ACCUM=4
    print_status "High-memory single GPU detected, using batch_size=$BATCH_SIZE"
else
    BATCH_SIZE=32   # Conservative settings
    INPUT_SIZE=384
    EPOCHS=40
    LR=1e-4
    GRAD_ACCUM=4
    print_status "Standard setup, using batch_size=$BATCH_SIZE"
fi

# Optimize workers for multi-GPU
NUM_WORKERS=$(nproc)
if [ $GPU_COUNT -ge 3 ]; then
    NUM_WORKERS=48  # 16 workers per GPU for 96 vCPU
elif [ $GPU_COUNT -eq 2 ]; then
    NUM_WORKERS=32  # 16 workers per GPU
elif [ $NUM_WORKERS -gt 16 ]; then
    NUM_WORKERS=16  # Use more workers for single GPU
else
    NUM_WORKERS=$(($NUM_WORKERS > 8 ? $NUM_WORKERS : 8))
fi

print_status "Using $NUM_WORKERS workers for data loading"

# Training command with multi-GPU support
if [ $GPU_COUNT -gt 1 ]; then
    print_status "Starting multi-GPU training with $GPU_COUNT GPUs..."
    torchrun --nproc_per_node=$GPU_COUNT --nnodes=1 train.py \
        --model_name tf_efficientnetv2_s \
        --pretrained \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --epochs $EPOCHS \
        --lr $LR \
        --input_size $INPUT_SIZE \
        --mixed_precision \
        --learnable_weights \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --use_compile \
        --wandb_project car-inspection-runpod-5090 \
        --experiment_name "multi_gpu_${GPU_COUNT}x_$(date +%Y%m%d_%H%M%S)"
else
    print_status "Starting single GPU training..."
    python3 train.py \
        --model_name tf_efficientnetv2_s \
        --pretrained \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --epochs $EPOCHS \
        --lr $LR \
        --input_size $INPUT_SIZE \
        --mixed_precision \
        --learnable_weights \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --use_compile \
        --wandb_project car-inspection-runpod \
        --experiment_name "single_gpu_$(date +%Y%m%d_%H%M%S)"
fi

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    print_error "Training failed with exit code $TRAINING_EXIT_CODE"
    exit 1
fi

print_status "Training completed successfully!"

# Step 7: Export Model
print_step "Exporting trained model..."

# Find the best model
BEST_MODEL=$(find models -name "best_model.pth" -type f | head -1)

if [ -z "$BEST_MODEL" ]; then
    print_warning "Best model not found, using latest checkpoint..."
    BEST_MODEL=$(find models -name "checkpoint.pth" -type f | head -1)
fi

if [ ! -z "$BEST_MODEL" ]; then
    print_status "Found model: $BEST_MODEL"

    # Create inference script
    cat > export_model.py << 'EOF'
import torch
from model import CarInspectionModel
from utils import export_model_for_inference
import sys

if len(sys.argv) != 2:
    print("Usage: python export_model.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"Loading model from {model_path}")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Create model
model = CarInspectionModel()
model.load_state_dict(checkpoint['model_state_dict'])

# Export for inference
export_path = model_path.replace('.pth', '_inference.pth')
export_model_for_inference(model, export_path)

print(f"Model exported for inference: {export_path}")
EOF

    python3 export_model.py "$BEST_MODEL"
    print_status "Model export completed!"
else
    print_warning "No trained model found for export"
fi

# Step 8: Final Summary
print_step "Training pipeline completed!"
echo ""
echo "📊 RESULTS SUMMARY:"
echo "=================="

if [ -d "models" ]; then
    echo "📁 Model files:"
    ls -la models/*/
    echo ""
fi

if [ -f "data/car_inspection_dataset.csv" ]; then
    echo "📊 Dataset info:"
    python3 -c "
import pandas as pd
df = pd.read_csv('data/car_inspection_dataset.csv')
print(f'Total samples: {len(df)}')
print(f'Train/Val/Test split: {df.groupby(\"split\").size().to_dict()}')
print(f'Average cleanliness: {df[\"cleanliness_score\"].mean():.3f}')
print(f'Average damage: {df[\"damage_score\"].mean():.3f}')
"
    echo ""
fi

print_status "Check Weights & Biases dashboard for detailed training metrics"
print_status "Model files saved in the models/ directory"

echo ""
echo "🎉 HACKATHON TRAINING PIPELINE COMPLETED! 🎉"
echo "============================================="