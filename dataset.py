#!/usr/bin/env python3
"""
PyTorch Dataset for Car Inspection Multi-task Learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path

class CarInspectionDataset(Dataset):
    """Multi-task dataset for car cleanliness and damage assessment"""

    def __init__(self, csv_file, transform=None, split='train'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        self.split = split

        print(f"📊 {split.upper()} Dataset: {len(self.df)} images")
        print(f"   Cleanliness range: {self.df['cleanliness_score'].min():.3f} - {self.df['cleanliness_score'].max():.3f}")
        print(f"   Damage range: {self.df['damage_score'].min():.3f} - {self.df['damage_score'].max():.3f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = row['image_path']

        try:
            # Use cv2 for better compatibility with albumentations
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image is None:
                # Fallback to PIL
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)

        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            # Create dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default to tensor conversion
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Labels
        cleanliness_score = torch.tensor(row['cleanliness_score'], dtype=torch.float32)
        damage_score = torch.tensor(row['damage_score'], dtype=torch.float32)
        weather_condition = torch.tensor(row['weather_condition'], dtype=torch.long)

        return {
            'image': image,
            'cleanliness_score': cleanliness_score,
            'damage_score': damage_score,
            'weather_condition': weather_condition,
            'image_path': image_path
        }

def get_transforms(input_size=384):
    """Get data augmentation transforms"""

    # Training transforms with heavy augmentation
    train_transform = A.Compose([
        # Resize and crop
        A.Resize(input_size + 32, input_size + 32),
        A.RandomCrop(input_size, input_size),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.Rotate(limit=15, p=0.3),

        # Color augmentations (important for dirt/cleanliness)
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),

        # Weather-like effects
        A.RandomRain(p=0.1),  # Simulate rain conditions
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.05),
        A.RandomSunFlare(p=0.05),
        A.RandomShadow(p=0.1),

        # Noise and blur
        A.GaussNoise(var_limit=(5, 20), p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),

        # Cutout for robustness
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Validation/test transforms (minimal)
    val_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform

def create_dataloaders(csv_file, batch_size=32, num_workers=4, input_size=384):
    """Create train/validation/test dataloaders"""

    train_transform, val_transform = get_transforms(input_size)

    # Create datasets
    train_dataset = CarInspectionDataset(csv_file, transform=train_transform, split='train')
    val_dataset = CarInspectionDataset(csv_file, transform=val_transform, split='valid')

    # Check if test split exists
    df = pd.read_csv(csv_file)
    test_dataset = None
    if 'test' in df['split'].unique():
        test_dataset = CarInspectionDataset(csv_file, transform=val_transform, split='test')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader

def analyze_dataset(csv_file):
    """Analyze dataset statistics"""
    df = pd.read_csv(csv_file)

    print("📊 DATASET ANALYSIS")
    print(f"Total samples: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    print("\n🎯 CLEANLINESS SCORES:")
    print(f"Mean: {df['cleanliness_score'].mean():.3f}")
    print(f"Std: {df['cleanliness_score'].std():.3f}")
    print(f"Min: {df['cleanliness_score'].min():.3f}")
    print(f"Max: {df['cleanliness_score'].max():.3f}")

    print("\n🔧 DAMAGE SCORES:")
    print(f"Mean: {df['damage_score'].mean():.3f}")
    print(f"Std: {df['damage_score'].std():.3f}")
    print(f"Min: {df['damage_score'].min():.3f}")
    print(f"Max: {df['damage_score'].max():.3f}")

    print("\n🌤️ WEATHER CONDITIONS:")
    weather_names = {0: 'Normal', 1: 'Overcast', 2: 'Snow', 3: 'Rain'}
    weather_counts = df['weather_condition'].value_counts()
    for weather_id, count in weather_counts.items():
        print(f"{weather_names.get(weather_id, f'Unknown({weather_id})')}: {count}")

if __name__ == "__main__":
    # Test dataset loading
    csv_file = "data/car_inspection_dataset.csv"

    if Path(csv_file).exists():
        analyze_dataset(csv_file)

        # Test dataloader creation
        train_loader, val_loader, test_loader = create_dataloaders(csv_file, batch_size=4)

        print(f"\n🚀 DATALOADERS READY:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        if test_loader:
            print(f"Test batches: {len(test_loader)}")

        # Test batch loading
        batch = next(iter(train_loader))
        print(f"\n📦 SAMPLE BATCH:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Cleanliness: {batch['cleanliness_score'][:3]}")
        print(f"Damage: {batch['damage_score'][:3]}")
        print(f"Weather: {batch['weather_condition'][:3]}")

    else:
        print(f"❌ Dataset file not found: {csv_file}")
        print("Run 'python convert_annotations.py' first!")