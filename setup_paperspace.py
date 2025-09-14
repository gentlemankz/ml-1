#!/usr/bin/env python3
"""
Paperspace setup script for Car Damage Detection project
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"🚀 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def main():
    print("🔧 Setting up Paperspace environment for Car Inspection ML...")

    # Update system
    run_command("sudo apt-get update -y", "Updating system packages")

    # Install system dependencies
    run_command("sudo apt-get install -y git wget unzip htop nvtop", "Installing system tools")

    # Install Python dependencies
    run_command("pip install --upgrade pip", "Upgrading pip")
    run_command("pip install -r requirements.txt", "Installing Python packages")

    # Setup data directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("✅ Environment setup completed!")

    # Check GPU
    gpu_info = run_command("nvidia-smi", "Checking GPU")
    if gpu_info:
        print("🔥 GPU detected and ready!")
    else:
        print("⚠️  No GPU detected - training will be slow")

    # Check PyTorch CUDA
    cuda_check = run_command("python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}\")'", "Testing PyTorch CUDA")

    print("\n🎯 Ready to start training!")
    print("Next steps:")
    print("1. Run: python convert_annotations.py")
    print("2. Run: python train.py")

if __name__ == "__main__":
    main()