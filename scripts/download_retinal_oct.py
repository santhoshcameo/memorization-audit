#!/usr/bin/env python3
"""
Download and prepare Retinal OCT dataset from Kaggle.

Dataset: Retinal OCT Images (optical coherence tomography)
Source: https://www.kaggle.com/datasets/paultimothymooney/kermany2018

Classes:
- CNV: Choroidal Neovascularization (common)
- DME: Diabetic Macular Edema (rare)
- DRUSEN: Early AMD drusen deposits (moderate)
- NORMAL: Normal retina (common)

Usage:
    python scripts/download_retinal_oct.py
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_dataset():
    """Download Retinal OCT dataset from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install kagglehub")
        import kagglehub

    print("=" * 60)
    print("DOWNLOADING RETINAL OCT DATASET")
    print("=" * 60)
    print()

    # Download dataset
    print("Downloading from Kaggle (this may take a while)...")
    path = kagglehub.dataset_download("paultimothymooney/kermany2018")
    print(f"Downloaded to: {path}")

    return Path(path)


def find_image_directories(download_path):
    """Find the train/test directories with images."""
    # The dataset structure is typically:
    # kermany2018/OCT2017/train/{CNV,DME,DRUSEN,NORMAL}/*.jpeg
    # kermany2018/OCT2017/test/{CNV,DME,DRUSEN,NORMAL}/*.jpeg

    possible_paths = [
        download_path / "OCT2017",
        download_path / "oct2017",
        download_path,
    ]

    for base in possible_paths:
        train_path = base / "train"
        test_path = base / "test"
        if train_path.exists() and test_path.exists():
            return train_path, test_path

    # Search recursively
    for train_dir in download_path.rglob("train"):
        if (train_dir / "CNV").exists() or (train_dir / "NORMAL").exists():
            test_dir = train_dir.parent / "test"
            if test_dir.exists():
                return train_dir, test_dir

    raise FileNotFoundError(f"Could not find train/test directories in {download_path}")


def prepare_dataset(download_path, output_dir):
    """Prepare dataset with stratified splits."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("PREPARING RETINAL OCT DATASET")
    print("=" * 60)

    # Find image directories
    train_dir, test_dir = find_image_directories(download_path)
    print(f"Found train directory: {train_dir}")
    print(f"Found test directory: {test_dir}")

    # Class names
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]

    # Collect all images
    all_samples = []

    for split_name, split_dir in [("orig_train", train_dir), ("orig_test", test_dir)]:
        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue

            for img_path in class_dir.glob("*.jpeg"):
                all_samples.append({
                    'original_path': img_path,
                    'class_name': class_name,
                    'original_split': split_name
                })

            # Also check for .jpg extension
            for img_path in class_dir.glob("*.jpg"):
                all_samples.append({
                    'original_path': img_path,
                    'class_name': class_name,
                    'original_split': split_name
                })

    print(f"\nTotal images found: {len(all_samples)}")

    # Show class distribution
    class_counts = Counter([s['class_name'] for s in all_samples])
    print("\nOriginal class distribution:")
    for cls, count in sorted(class_counts.items()):
        pct = count / len(all_samples) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    # Create stratified splits
    # For memorization study: 70% train, 15% canary, 15% test
    np.random.seed(42)

    metadata_rows = []

    for class_name in classes:
        class_df = df[df['class_name'] == class_name].copy()
        class_df = class_df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(class_df)
        n_train = int(0.70 * n)
        n_canary = int(0.15 * n)

        # Assign splits
        class_df['split'] = 'test'
        class_df.iloc[:n_train, class_df.columns.get_loc('split')] = 'train'
        class_df.iloc[n_train:n_train+n_canary, class_df.columns.get_loc('split')] = 'canary'

        metadata_rows.append(class_df)

    # Combine all
    metadata_df = pd.concat(metadata_rows, ignore_index=True)

    # Copy images and create metadata
    print("\nCopying images and creating metadata...")
    final_metadata = []

    for idx, row in metadata_df.iterrows():
        src_path = row['original_path']
        # Create unique filename
        new_filename = f"{row['class_name']}_{idx:06d}.jpeg"
        dst_path = images_dir / new_filename

        # Copy image
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        final_metadata.append({
            'image_id': new_filename,
            'dx': row['class_name'],
            'split': row['split']
        })

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(metadata_df)} images...")

    # Save metadata
    metadata_path = output_dir / "metadata.csv"
    final_df = pd.DataFrame(final_metadata)
    final_df.to_csv(metadata_path, index=False)

    # Print split statistics
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    for split in ['train', 'canary', 'test']:
        split_df = final_df[final_df['split'] == split]
        print(f"\n{split.upper()} split: {len(split_df)} images")
        for cls in classes:
            count = len(split_df[split_df['dx'] == cls])
            pct = count / len(split_df) * 100
            rarity = ""
            if pct < 5:
                rarity = " [RARE]"
            elif pct < 15:
                rarity = " [MODERATE]"
            print(f"  {cls}: {count} ({pct:.1f}%){rarity}")

    # Save split info
    split_info = {
        'train': len(final_df[final_df['split'] == 'train']),
        'canary': len(final_df[final_df['split'] == 'canary']),
        'test': len(final_df[final_df['split'] == 'test']),
        'total': len(final_df),
        'classes': classes
    }

    import yaml
    with open(output_dir / "split_info.yaml", 'w') as f:
        yaml.dump(split_info, f)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Images saved to: {images_dir}")

    return output_dir


def update_dataset_config(output_dir):
    """Update the dataset configuration file."""
    config_path = PROJECT_ROOT / "config" / "datasets" / "retinal_oct.yaml"

    config_content = f"""# Retinal OCT Dataset Configuration
# Optical Coherence Tomography images for retinal disease classification

name: retinal_oct
description: Retinal OCT Images - Kermany et al. 2018

info:
  num_classes: 4
  modality: Optical Coherence Tomography (OCT)
  domain: Ophthalmology
  source: https://www.kaggle.com/datasets/paultimothymooney/kermany2018

paths:
  images: {output_dir}/images
  metadata: {output_dir}/metadata.csv

classes:
  names:
    - CNV      # Choroidal Neovascularization
    - DME      # Diabetic Macular Edema
    - DRUSEN   # Early AMD
    - NORMAL   # Normal retina

  descriptions:
    CNV: "Choroidal Neovascularization - abnormal blood vessel growth"
    DME: "Diabetic Macular Edema - fluid accumulation from diabetes"
    DRUSEN: "Drusen deposits - early age-related macular degeneration"
    NORMAL: "Normal healthy retina"

preprocessing:
  target_size: 224
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

augmentation:
  use_dataset_default: true
"""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    print(f"\nDataset config saved to: {config_path}")


def main():
    output_dir = PROJECT_ROOT / "data" / "retinal_oct"

    # Check if already downloaded
    if (output_dir / "metadata.csv").exists() and (output_dir / "images").exists():
        n_images = len(list((output_dir / "images").glob("*.jpeg")))
        if n_images > 50000:
            print(f"Retinal OCT dataset already exists with {n_images} images")
            print("Use --force to re-download")
            return

    # Download
    download_path = download_dataset()

    # Prepare
    prepare_dataset(download_path, output_dir)

    # Update config
    update_dataset_config(output_dir)

    print("\n" + "=" * 60)
    print("RETINAL OCT DATASET READY")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run preprocessing: python scripts/preprocess_data.py --dataset retinal_oct")
    print(f"  2. Run experiment: python run_experiment.py --experiment retinal_oct_baseline")


if __name__ == "__main__":
    main()
