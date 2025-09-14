#!/usr/bin/env python3
"""
Training script for Multi-task Car Inspection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

import wandb
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import CarInspectionModel, MultiTaskLoss
from dataset import create_dataloaders, analyze_dataset
from utils import setup_logging, save_checkpoint, load_checkpoint, calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train Car Inspection Model')

    # Model args
    parser.add_argument('--model_name', type=str, default='efficientnetv2_l',
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Data args
    parser.add_argument('--csv_file', type=str, default='data/car_inspection_dataset.csv',
                       help='Dataset CSV file')
    parser.add_argument('--input_size', type=int, default=384,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers per GPU')

    # Training args
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs for large batch training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps for larger effective batch size')

    # Loss args
    parser.add_argument('--learnable_weights', action='store_true', default=True,
                       help='Use learnable task weights')

    # Training settings
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--use_bfloat16', action='store_true', default=False,
                       help='Use bfloat16 instead of float16 for mixed precision (better on RTX 5090)')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--use_compile', action='store_true', default=False,
                       help='Use torch.compile for optimization (PyTorch 2.0+)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='car-inspection-hackathon',
                       help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')

    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')

    return parser.parse_args()

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_device()
        self.setup_experiment()
        self.setup_data()
        self.setup_model()
        self.setup_training()

    def setup_distributed(self):
        """Setup distributed training"""
        self.is_distributed = False
        self.local_rank = self.args.local_rank
        self.world_size = 1
        self.rank = 0

        if 'WORLD_SIZE' in os.environ:
            self.is_distributed = True
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])

            # Initialize with timeout and proper error handling
            try:
                dist.init_process_group(
                    backend='nccl',
                    timeout=torch.distributed.default_pg_timeout
                )
                torch.cuda.set_device(self.local_rank)

                if self.rank == 0:
                    print(f"🚀 Distributed training with {self.world_size} GPUs")
            except Exception as e:
                print(f"❌ Failed to initialize distributed training: {e}")
                print("🔄 Falling back to single GPU training")
                self.is_distributed = False
                self.multi_gpu = False
        elif torch.cuda.device_count() > 1:
            self.is_distributed = False
            self.multi_gpu = True
            print(f"🚀 Multi-GPU training with {torch.cuda.device_count()} GPUs using DataParallel")
        else:
            self.multi_gpu = False
            print("🚀 Single GPU training")

    def setup_device(self):
        """Setup training device"""
        if self.is_distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
        elif self.args.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.args.device)

        if self.rank == 0 or not self.is_distributed:
            print(f"🔥 Device: {self.device}")
            if torch.cuda.is_available():
                if self.is_distributed or hasattr(self, 'multi_gpu'):
                    total_gpus = torch.cuda.device_count()
                    total_memory = sum([torch.cuda.get_device_properties(i).total_memory for i in range(total_gpus)])
                    print(f"🚀 GPUs: {total_gpus}x {torch.cuda.get_device_name(0)}")
                    print(f"💾 Total GPU Memory: {total_memory / 1e9:.1f} GB")
                else:
                    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def setup_experiment(self):
        """Setup experiment tracking"""
        # Create experiment name if not provided
        if self.args.experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.args.model_name}_{timestamp}"
        else:
            self.experiment_name = self.args.experiment_name

        # Setup directories
        self.save_dir = Path(self.args.save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B only on rank 0
        if self.rank == 0 or not self.is_distributed:
            wandb.init(
                project=self.args.wandb_project,
                name=self.experiment_name,
                config=vars(self.args)
            )
            print(f"📊 Experiment: {self.experiment_name}")
        else:
            wandb.init(mode='disabled')

    def setup_data(self):
        """Setup data loaders"""
        print("📂 Setting up data loaders...")

        # Analyze dataset
        analyze_dataset(self.args.csv_file)

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.args.csv_file,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            input_size=self.args.input_size
        )

        print(f"✅ Data loaders ready")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        if self.test_loader:
            print(f"   Test batches: {len(self.test_loader)}")

    def setup_model(self):
        """Setup model and loss function"""
        print("🏗️ Setting up model...")

        # Create model
        self.model = CarInspectionModel(
            model_name=self.args.model_name,
            pretrained=self.args.pretrained,
            dropout=self.args.dropout
        ).to(self.device)

        # Setup multi-GPU training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        elif hasattr(self, 'multi_gpu') and self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        # Count parameters
        model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
        total_params = sum(p.numel() for p in model_for_params.parameters())
        trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)

        if self.rank == 0 or not self.is_distributed:
            print(f"📊 Model Parameters:")
            print(f"   Total: {total_params:,}")
            print(f"   Trainable: {trainable_params:,}")

        # Setup loss function
        self.criterion = MultiTaskLoss(
            learnable_weights=self.args.learnable_weights
        ).to(self.device)

        # Apply torch.compile for optimization (PyTorch 2.0+)
        if self.args.use_compile and hasattr(torch, 'compile'):
            print("🚀 Applying torch.compile optimization...")
            model_to_compile = self.model.module if hasattr(self.model, 'module') else self.model
            compiled_model = torch.compile(model_to_compile, mode='max-autotune')
            if hasattr(self.model, 'module'):
                self.model.module = compiled_model
            else:
                self.model = compiled_model

        # Log model to W&B (only on rank 0)
        if self.rank == 0 or not self.is_distributed:
            wandb.watch(self.model, log='all')

    def setup_training(self):
        """Setup optimizer and scheduler"""
        print("⚙️ Setting up training...")

        # Optimizer - scale learning rate for distributed training
        lr = self.args.lr
        if self.is_distributed:
            lr = self.args.lr * self.world_size  # Linear scaling rule

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-7
        )

        # Mixed precision scaler with bfloat16 support
        if self.args.mixed_precision:
            if self.args.use_bfloat16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                print("🚀 Using bfloat16 mixed precision (optimal for RTX 5090)")
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bfloat16 doesn't need gradient scaling
            else:
                print("🚀 Using float16 mixed precision")
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_patience = 10

        # Resume from checkpoint if provided
        if self.args.resume:
            self.load_checkpoint(self.args.resume)

        print("✅ Training setup complete")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total_loss': [], 'cleanliness_loss': [], 'damage_loss': [], 'weather_loss': []}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device, non_blocking=True)
            targets = {
                'cleanliness_score': batch['cleanliness_score'].to(self.device, non_blocking=True),
                'damage_score': batch['damage_score'].to(self.device, non_blocking=True),
                'weather_condition': batch['weather_condition'].to(self.device, non_blocking=True)
            }

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with gradient accumulation
            if self.args.mixed_precision:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(images, return_uncertainty=True)
                    losses = self.criterion(outputs, targets)
                    loss = losses['total_loss'] / self.args.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = self.model(images, return_uncertainty=True)
                losses = self.criterion(outputs, targets)
                loss = losses['total_loss'] / self.args.gradient_accumulation_steps
                loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.args.gradient_clipping > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
                        self.optimizer.step()
                else:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                self.optimizer.zero_grad()

            # Track losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key].append(losses[key].item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to W&B (every 50 steps, only on rank 0)
            if batch_idx % 50 == 0 and (self.rank == 0 or not self.is_distributed):
                step = self.current_epoch * len(self.train_loader) + batch_idx
                wandb.log({
                    'train/step_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': step
                })

        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            if values:
                epoch_metrics[f'train/{key}'] = np.mean(values)

        return epoch_metrics

    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        val_losses = {'total_loss': [], 'cleanliness_loss': [], 'damage_loss': [], 'weather_loss': []}

        all_outputs = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            # Move batch to device
            images = batch['image'].to(self.device, non_blocking=True)
            targets = {
                'cleanliness_score': batch['cleanliness_score'].to(self.device, non_blocking=True),
                'damage_score': batch['damage_score'].to(self.device, non_blocking=True),
                'weather_condition': batch['weather_condition'].to(self.device, non_blocking=True)
            }

            # Forward pass
            outputs = self.model(images, return_uncertainty=True)
            losses = self.criterion(outputs, targets)

            # Track losses
            for key in val_losses:
                if key in losses:
                    val_losses[key].append(losses[key].item())

            # Collect outputs and targets for metrics
            all_outputs.append({
                'cleanliness_score': outputs['cleanliness_score'].cpu(),
                'damage_score': outputs['damage_score'].cpu(),
                'weather_logits': outputs['weather_logits'].cpu(),
                'overall_confidence': outputs.get('overall_confidence', torch.zeros_like(outputs['cleanliness_score'])).cpu()
            })

            all_targets.append({
                'cleanliness_score': batch['cleanliness_score'],
                'damage_score': batch['damage_score'],
                'weather_condition': batch['weather_condition']
            })

        # Calculate metrics
        val_metrics = calculate_metrics(all_outputs, all_targets)

        # Add loss metrics
        for key, values in val_losses.items():
            if values:
                val_metrics[f'val/{key}'] = np.mean(values)

        return val_metrics

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args
        }

        # Save regular checkpoint
        checkpoint_path = self.save_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"✅ Resumed from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training for {self.args.epochs} epochs...")

        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch

            # Log to W&B (only on rank 0)
            if self.rank == 0 or not self.is_distributed:
                wandb.log(all_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.args.epochs}:")
            print(f"  Train Loss: {train_metrics.get('train/total_loss', 0):.4f}")
            print(f"  Val Loss: {val_metrics.get('val/total_loss', 0):.4f}")
            print(f"  Cleanliness MAE: {val_metrics.get('val/cleanliness_mae', 0):.4f}")
            print(f"  Damage MAE: {val_metrics.get('val/damage_mae', 0):.4f}")
            print(f"  Weather Acc: {val_metrics.get('val/weather_accuracy', 0):.4f}")

            # Check for best model
            current_val_loss = val_metrics.get('val/total_loss', float('inf'))
            is_best = current_val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = current_val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"⏰ Early stopping at epoch {epoch+1}")
                break

        print("✅ Training completed!")

        # Test evaluation if test loader exists
        if self.test_loader:
            print("🧪 Evaluating on test set...")
            # Load best model
            best_checkpoint = self.save_dir / 'best_model.pth'
            if best_checkpoint.exists():
                checkpoint = torch.load(best_checkpoint)
                self.model.load_state_dict(checkpoint['model_state_dict'])

            test_metrics = self.validate()  # Same as validation but with test data
            wandb.log({'test/' + k.replace('val/', ''): v for k, v in test_metrics.items()})

def main():
    args = parse_args()

    # Check if dataset exists
    if not Path(args.csv_file).exists():
        print(f"❌ Dataset file not found: {args.csv_file}")
        print("Run 'python convert_annotations.py' first!")
        return

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()

    print("🎉 Training pipeline completed!")

if __name__ == "__main__":
    main()