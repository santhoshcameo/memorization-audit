#!/usr/bin/env python3
"""
HAM10000 Downloader - Skin Lesion Dataset

Downloads HAM10000 (Human Against Machine with 10000 training images) dataset
from Kaggle and prepares it for the memorization study.

Key features:
- Auto-downloads from Kaggle using kagglehub
- Creates stratified train/canary/test splits
- Handles the highly imbalanced class distribution
- Optimized for rare disease memorization analysis

Classes:
- akiec: Actinic keratoses (rare)
- bcc: Basal cell carcinoma
- bkl: Benign keratosis
- df: Dermatofibroma (rare)
- mel: Melanoma
- nv: Melanocytic nevi (most common)
- vasc: Vascular lesions (rare)

Usage:
    python scripts/download_ham10000.py
    python scripts/download_ham10000.py --output-dir data/ham10000_custom
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "HAM10000"


def download_dataset():
    """Download HAM10000 using kagglehub."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING HAM10000 DATASET")
    logger.info("=" * 60)

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading from Kaggle...")
    logger.info("Dataset: skin-cancer-mnist-ham10000")

    # Download dataset
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    logger.info(f"Dataset downloaded to: {path}")

    return Path(path)


def load_metadata(data_path: Path) -> pd.DataFrame:
    """Load and process the metadata file."""
    logger.info("Loading metadata...")

    # Find the metadata CSV file
    metadata_files = list(data_path.rglob("HAM10000_metadata.csv"))
    if not metadata_files:
        # Try alternate name
        metadata_files = list(data_path.rglob("*metadata*.csv"))

    if not metadata_files:
        raise FileNotFoundError(f"No metadata CSV found in {data_path}")

    metadata_path = metadata_files[0]
    logger.info(f"Found metadata: {metadata_path}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Total images in dataset: {len(df):,}")

    # Print class distribution
    class_counts = df['dx'].value_counts()
    logger.info(f"\nClass distribution ({len(class_counts)} classes):")

    class_names = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }

    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        rarity = "RARE" if pct < 5 else ""
        name = class_names.get(cls, cls)
        logger.info(f"  {cls:8s} ({name:25s}): {count:5,} ({pct:5.2f}%) {rarity}")

    return df


def copy_images(data_path: Path, df: pd.DataFrame, output_dir: Path) -> None:
    """Copy images to the output directory."""
    logger.info("=" * 60)
    logger.info("COPYING IMAGES")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image directories
    image_dirs = []
    for pattern in ["HAM10000_images_part_*", "ham10000_images_part_*", "*images*"]:
        image_dirs.extend(data_path.rglob(pattern))

    image_dirs = [d for d in image_dirs if d.is_dir()]

    if not image_dirs:
        # Images might be directly in the data path
        image_dirs = [data_path]

    logger.info(f"Found {len(image_dirs)} image directories")

    # Get list of needed images
    needed_images = set(df['image_id'].tolist())
    logger.info(f"Need to copy {len(needed_images):,} images")

    # Check existing images
    existing = set(f.stem for f in output_dir.glob("*.jpg"))
    needed_images -= existing

    if not needed_images:
        logger.info("All images already copied!")
        return

    logger.info(f"Already have {len(existing):,}, need {len(needed_images):,} more")

    # Copy images
    copied = 0
    for img_dir in image_dirs:
        if not needed_images:
            break

        for img_file in img_dir.glob("*.jpg"):
            if img_file.stem in needed_images:
                shutil.copy2(img_file, output_dir / img_file.name)
                needed_images.discard(img_file.stem)
                copied += 1

                if copied % 1000 == 0:
                    logger.info(f"  Copied {copied:,} images...")

    logger.info(f"Total copied: {copied:,} images")

    if needed_images:
        logger.warning(f"Could not find {len(needed_images)} images")


def create_splits(df: pd.DataFrame,
                  output_dir: Path,
                  train_ratio: float = 0.70,
                  canary_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> None:
    """Create stratified train/canary/test splits."""
    logger.info("=" * 60)
    logger.info("CREATING STRATIFIED SPLITS")
    logger.info("=" * 60)

    from sklearn.model_selection import train_test_split

    # First split: train+canary vs test
    train_canary_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['dx'],
        random_state=seed
    )

    # Second split: train vs canary
    canary_adjusted_ratio = canary_ratio / (train_ratio + canary_ratio)
    train_df, canary_df = train_test_split(
        train_canary_df,
        test_size=canary_adjusted_ratio,
        stratify=train_canary_df['dx'],
        random_state=seed
    )

    # Assign split labels
    train_df = train_df.copy()
    canary_df = canary_df.copy()
    test_df = test_df.copy()

    train_df['split'] = 'train'
    canary_df['split'] = 'canary'
    test_df['split'] = 'test'

    # Combine and save
    final_df = pd.concat([train_df, canary_df, test_df], ignore_index=True)

    # Save metadata
    metadata_path = output_dir / "HAM10000_metadata.csv"
    final_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save split info
    import yaml
    split_info = {
        'train': len(train_df),
        'canary': len(canary_df),
        'test': len(test_df),
        'total': len(final_df),
        'seed': seed,
        'classes': sorted(df['dx'].unique().tolist())
    }

    split_info_path = output_dir.parent.parent / "processed" / "ham10000" / "split_info.yaml"
    split_info_path.parent.mkdir(parents=True, exist_ok=True)

    with open(split_info_path, 'w') as f:
        yaml.dump(split_info, f, default_flow_style=False)

    logger.info(f"\nSplit distribution:")
    logger.info(f"  Train:  {len(train_df):,} ({len(train_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Canary: {len(canary_df):,} ({len(canary_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Test:   {len(test_df):,} ({len(test_df)/len(final_df)*100:.1f}%)")

    # Per-class canary distribution
    logger.info(f"\nCanary samples per class:")
    canary_counts = canary_df['dx'].value_counts()
    for cls, count in canary_counts.items():
        logger.info(f"  {cls:8s}: {count:,}")


def update_config(output_dir: Path) -> None:
    """Update the HAM10000 config file with correct paths."""
    import yaml

    config_path = PROJECT_ROOT / "config" / "datasets" / "ham10000.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return

    # Read existing config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update paths to be relative to PROJECT_ROOT
    config['paths']['images'] = str(output_dir)
    config['paths']['metadata'] = str(output_dir / "HAM10000_metadata.csv")

    # Backup old config
    backup_path = config_path.with_suffix('.yaml.backup')
    shutil.copy2(config_path, backup_path)

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download HAM10000 skin lesion dataset'
    )
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/raw/HAM10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, use existing data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to existing downloaded data')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HAM10000 DOWNLOAD & PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Step 1: Download or locate data
    if args.skip_download and args.data_path:
        data_path = Path(args.data_path)
        logger.info(f"Using existing data at: {data_path}")
    else:
        data_path = download_dataset()

    # Step 2: Load metadata
    df = load_metadata(data_path)

    # Step 3: Copy images
    copy_images(data_path, df, output_dir)

    # Step 4: Create splits
    create_splits(df, output_dir, seed=args.seed)

    # Step 5: Update config
    update_config(output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Metadata: {output_dir / 'HAM10000_metadata.csv'}")
    logger.info("")
    logger.info("Next step: Run training with ./run_all.sh")


if __name__ == "__main__":
    main()
