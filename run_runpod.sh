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

# Training configuration for Paperspace
BATCH_SIZE=32
if python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory)" 2>/dev/null | head -1 | awk '{print $1 > 20000000000}' | grep -q 1; then
    BATCH_SIZE=48  # Larger batch for high-memory GPUs
    print_status "High-memory GPU detected, using batch size $BATCH_SIZE"
else
    print_status "Using batch size $BATCH_SIZE"
fi

# Get number of CPU cores for data loading
NUM_WORKERS=$(nproc)
if [ $NUM_WORKERS -gt 8 ]; then
    NUM_WORKERS=8
fi

print_status "Using $NUM_WORKERS workers for data loading"

# Training command
python3 train.py \
    --model_name efficientnetv2_l \
    --pretrained \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs 30 \
    --lr 1e-4 \
    --input_size 384 \
    --mixed_precision \
    --learnable_weights \
    --wandb_project car-inspection-runpod \
    --experiment_name "hackathon_$(date +%Y%m%d_%H%M%S)"

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