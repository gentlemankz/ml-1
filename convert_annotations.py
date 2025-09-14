#!/usr/bin/env python3
"""
Convert COCO annotations to multi-task regression labels for car inspection
"""

import json
import os
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

class COCOToRegressionConverter:
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.coco_dir = self.datasets_dir / "coco-folders"
        self.unlabeled_dir = self.datasets_dir / "no-labeled"

        # Category mappings
        self.damage_categories = {'dent', 'scratch'}
        self.cleanliness_categories = {'dirt'}

        # Weather condition mappings for unlabeled data
        self.weather_mapping = {
            'direct-sunlight': 0,
            'overcast-day': 1,
            'snow': 2,
            'after-the-rain': 3
        }

    def analyze_coco_dataset(self, coco_file):
        """Analyze COCO annotation file"""
        with open(coco_file, 'r') as f:
            data = json.load(f)

        print(f"📊 Dataset: {coco_file.parent.name}")
        print(f"   Images: {len(data.get('images', []))}")
        print(f"   Annotations: {len(data.get('annotations', []))}")

        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        print(f"   Categories: {list(categories.values())}")

        return data, categories

    def calculate_scores_from_annotations(self, image_info, annotations, categories, image_area):
        """Calculate cleanliness and damage scores from COCO annotations"""
        cleanliness_score = 1.0  # Start with clean
        damage_score = 1.0      # Start with no damage

        total_dirt_area = 0
        damage_count = 0

        for ann in annotations:
            category_name = categories.get(ann['category_id'], '').lower()
            bbox_area = ann.get('area', 0)

            if category_name in self.cleanliness_categories:
                # Dirt affects cleanliness
                dirt_ratio = bbox_area / image_area if image_area > 0 else 0
                total_dirt_area += dirt_ratio

            elif category_name in self.damage_categories:
                # Damage affects damage score
                damage_count += 1
                damage_ratio = bbox_area / image_area if image_area > 0 else 0

                # Different penalties for different damage types
                if category_name == 'dent':
                    damage_score -= min(0.4, damage_ratio * 2.0)  # Dents are worse
                elif category_name == 'scratch':
                    damage_score -= min(0.3, damage_ratio * 1.5)  # Scratches less severe

        # Apply dirt penalty
        cleanliness_score = max(0.0, cleanliness_score - min(0.8, total_dirt_area * 3.0))

        # Apply damage penalty
        damage_score = max(0.0, damage_score)

        # If no damage annotations but dirt present, assume some wear
        if damage_count == 0 and total_dirt_area > 0.1:
            damage_score = max(0.6, damage_score)  # Slight penalty for very dirty cars

        return cleanliness_score, damage_score

    def process_coco_datasets(self):
        """Process all COCO datasets"""
        all_data = []

        for dataset_dir in self.coco_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            print(f"\n🔄 Processing {dataset_dir.name}...")

            # Find annotation files
            for split in ['train', 'valid', 'test']:
                ann_file = dataset_dir / split / "_annotations.coco.json"
                if not ann_file.exists():
                    continue

                data, categories = self.analyze_coco_dataset(ann_file)

                # Create image path mappings
                images = {img['id']: img for img in data['images']}

                # Group annotations by image
                image_annotations = {}
                for ann in data['annotations']:
                    img_id = ann['image_id']
                    if img_id not in image_annotations:
                        image_annotations[img_id] = []
                    image_annotations[img_id].append(ann)

                # Process each image
                for img_id, img_info in tqdm(images.items(), desc=f"Processing {split}"):
                    image_path = dataset_dir / split / img_info['file_name']

                    if not image_path.exists():
                        continue

                    # Calculate image area
                    image_area = img_info['width'] * img_info['height']

                    # Get annotations for this image
                    anns = image_annotations.get(img_id, [])

                    # Calculate scores
                    cleanliness, damage = self.calculate_scores_from_annotations(
                        img_info, anns, categories, image_area
                    )

                    # Detect weather condition from dataset name (basic heuristic)
                    weather = 0  # default: normal
                    if 'dirt' in dataset_dir.name.lower():
                        weather = 1  # likely overcast/muddy conditions

                    all_data.append({
                        'image_path': str(image_path),
                        'cleanliness_score': round(cleanliness, 3),
                        'damage_score': round(damage, 3),
                        'weather_condition': weather,
                        'dataset': dataset_dir.name,
                        'split': split,
                        'has_annotations': len(anns) > 0
                    })

        return all_data

    def process_unlabeled_data(self):
        """Process unlabeled data with manual scoring"""
        unlabeled_data = []

        # Manual scoring based on folder structure
        scoring_rules = {
            'clean-car-in-direct-sunlight': {'cleanliness': 0.95, 'damage': 0.90, 'weather': 0},
            'clean-car-in-snow': {'cleanliness': 0.85, 'damage': 0.85, 'weather': 2},
            'clean-car-on-overcast-day': {'cleanliness': 0.90, 'damage': 0.88, 'weather': 1},
            'dirty-car-in-direct-sunlight': {'cleanliness': 0.20, 'damage': 0.70, 'weather': 0},
            'dirty-car-in-snow': {'cleanliness': 0.15, 'damage': 0.65, 'weather': 2},
            'dirty-car-on-overcast-day': {'cleanliness': 0.25, 'damage': 0.72, 'weather': 1},
            'dirty-cars-after-the-rain': {'cleanliness': 0.10, 'damage': 0.60, 'weather': 3}
        }

        for condition_dir in self.unlabeled_dir.rglob("*"):
            if not condition_dir.is_dir():
                continue

            folder_name = condition_dir.name
            if folder_name in scoring_rules:
                scores = scoring_rules[folder_name]

                # Find all images in this folder
                for img_path in condition_dir.glob("*.jpg"):
                    # Add some variance to avoid overfitting
                    cleanliness = scores['cleanliness'] + np.random.normal(0, 0.05)
                    damage = scores['damage'] + np.random.normal(0, 0.03)

                    unlabeled_data.append({
                        'image_path': str(img_path),
                        'cleanliness_score': round(np.clip(cleanliness, 0, 1), 3),
                        'damage_score': round(np.clip(damage, 0, 1), 3),
                        'weather_condition': scores['weather'],
                        'dataset': 'unlabeled',
                        'split': 'train',  # Add to training set
                        'has_annotations': False
                    })

        return unlabeled_data

    def create_dataset_csv(self):
        """Create final dataset CSV"""
        print("🚀 Converting COCO annotations to regression labels...")

        # Process COCO datasets
        coco_data = self.process_coco_datasets()
        print(f"✅ Processed {len(coco_data)} COCO images")

        # Process unlabeled data
        unlabeled_data = self.process_unlabeled_data()
        print(f"✅ Processed {len(unlabeled_data)} unlabeled images")

        # Combine all data
        all_data = coco_data + unlabeled_data

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Save to CSV
        output_file = "data/car_inspection_dataset.csv"
        df.to_csv(output_file, index=False)

        # Print statistics
        print(f"\n📊 DATASET STATISTICS:")
        print(f"Total images: {len(df)}")
        print(f"Average cleanliness: {df['cleanliness_score'].mean():.3f}")
        print(f"Average damage score: {df['damage_score'].mean():.3f}")
        print(f"\nSplit distribution:")
        print(df['split'].value_counts())
        print(f"\nWeather distribution:")
        print(df['weather_condition'].value_counts())
        print(f"\nDataset distribution:")
        print(df['dataset'].value_counts())

        print(f"\n💾 Dataset saved to: {output_file}")
        return output_file

def main():
    converter = COCOToRegressionConverter()
    dataset_file = converter.create_dataset_csv()

    print("\n🎯 Ready for training!")
    print(f"Dataset: {dataset_file}")

if __name__ == "__main__":
    main()