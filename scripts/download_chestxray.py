#!/usr/bin/env python3
"""
NIH ChestX-ray14 Downloader with Stratified Sampling

Downloads NIH ChestX-ray14 dataset and creates a stratified sample
optimized for rare disease memorization analysis.

Key features:
- Downloads only what's needed (metadata first, then selective images)
- Stratified sampling to ensure rare diseases are well-represented
- Minimum samples per class for statistical validity
- Progress tracking and resume capability

Usage:
    python scripts/download_chestxray.py --target-samples 50000
    python scripts/download_chestxray.py --target-samples 40000 --min-per-class 500
"""

import os
import sys
import argparse
import shutil
import tarfile
import logging
from pathlib import Path
from collections import Counter
import random

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
OUTPUT_DIR = PROJECT_ROOT / "data" / "chestxray_50k"


def download_dataset():
    """Download NIH ChestX-ray14 using kagglehub."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING NIH CHESTX-RAY14 DATASET")
    logger.info("=" * 60)

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading from Kaggle (this may take a while)...")
    logger.info("Note: Full dataset is ~45GB but we'll only use what we need")

    # Download dataset
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    logger.info(f"Dataset downloaded to: {path}")

    return Path(path)


def load_metadata(data_path: Path) -> pd.DataFrame:
    """Load and process the metadata file."""
    logger.info("Loading metadata...")

    # Find the Data_Entry CSV file
    metadata_files = list(data_path.rglob("Data_Entry*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No Data_Entry*.csv found in {data_path}")

    metadata_path = metadata_files[0]
    logger.info(f"Found metadata: {metadata_path}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Total images in dataset: {len(df):,}")

    # Rename columns for consistency
    df = df.rename(columns={
        'Image Index': 'image_id',
        'Finding Labels': 'finding_labels'
    })

    # Extract primary finding (first label, excluding "No Finding")
    def get_primary_finding(labels):
        findings = labels.split('|')
        for f in findings:
            if f != 'No Finding':
                return f
        return 'No Finding'

    df['dx'] = df['finding_labels'].apply(get_primary_finding)

    # Remove "No Finding" samples (we want disease samples)
    df_diseases = df[df['dx'] != 'No Finding'].copy()
    logger.info(f"Samples with diseases: {len(df_diseases):,}")

    return df, df_diseases


def compute_stratified_sample(df: pd.DataFrame,
                              target_samples: int,
                              min_per_class: int,
                              seed: int = 42) -> pd.DataFrame:
    """
    Create stratified sample ensuring rare diseases are well-represented.

    Strategy:
    1. Ensure minimum samples per class (critical for rare diseases)
    2. Fill remaining quota proportionally
    3. Oversample rare classes if needed
    """
    logger.info("=" * 60)
    logger.info("COMPUTING STRATIFIED SAMPLE")
    logger.info("=" * 60)
    logger.info(f"Target total samples: {target_samples:,}")
    logger.info(f"Minimum per class: {min_per_class}")

    np.random.seed(seed)
    random.seed(seed)

    # Get class distribution
    class_counts = df['dx'].value_counts()
    classes = class_counts.index.tolist()
    n_classes = len(classes)

    logger.info(f"\nOriginal class distribution ({n_classes} classes):")
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        rarity = "RARE" if pct < 2 else "MODERATE" if pct < 5 else ""
        logger.info(f"  {cls:25s}: {count:6,} ({pct:5.2f}%) {rarity}")

    # Calculate target samples per class
    # Strategy: Ensure min_per_class, then distribute remaining proportionally

    # Step 1: Allocate minimum to each class
    allocated = {cls: min(min_per_class, class_counts[cls]) for cls in classes}
    total_allocated = sum(allocated.values())

    # Step 2: Distribute remaining samples proportionally
    remaining = target_samples - total_allocated
    if remaining > 0:
        # Calculate proportional allocation for remainder
        available = {cls: class_counts[cls] - allocated[cls] for cls in classes}
        total_available = sum(available.values())

        for cls in classes:
            if total_available > 0:
                extra = int(remaining * (available[cls] / total_available))
                extra = min(extra, available[cls])  # Don't exceed available
                allocated[cls] += extra

    # Step 3: Sample from each class
    sampled_dfs = []
    for cls in classes:
        n_samples = allocated[cls]
        cls_df = df[df['dx'] == cls]

        if n_samples >= len(cls_df):
            # Take all samples
            sampled_dfs.append(cls_df)
        else:
            # Random sample
            sampled_dfs.append(cls_df.sample(n=n_samples, random_state=seed))

    result_df = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle the result
    result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Log final distribution
    final_counts = result_df['dx'].value_counts()
    logger.info(f"\nFinal stratified sample ({len(result_df):,} samples):")
    for cls in classes:
        count = final_counts.get(cls, 0)
        pct = count / len(result_df) * 100
        orig_pct = class_counts[cls] / len(df) * 100
        logger.info(f"  {cls:25s}: {count:6,} ({pct:5.2f}%) [orig: {orig_pct:.2f}%]")

    return result_df


def extract_images(data_path: Path,
                   sample_df: pd.DataFrame,
                   output_dir: Path) -> None:
    """
    Extract only the images we need from the tar.gz files.
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING SELECTED IMAGES")
    logger.info("=" * 60)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Get list of needed images
    needed_images = set(sample_df['image_id'].tolist())
    logger.info(f"Need to extract {len(needed_images):,} images")

    # Check for already extracted images
    existing = set(f.name for f in images_dir.glob("*.png"))
    needed_images -= existing

    if not needed_images:
        logger.info("All images already extracted!")
        return

    logger.info(f"Already have {len(existing):,}, need {len(needed_images):,} more")

    # Find tar.gz files
    tar_files = sorted(data_path.rglob("images_*.tar.gz"))
    if not tar_files:
        # Try looking for extracted folders
        image_folders = sorted(data_path.rglob("images_*"))
        image_folders = [f for f in image_folders if f.is_dir()]

        if image_folders:
            logger.info(f"Found {len(image_folders)} extracted image folders")
            extracted = 0
            for folder in image_folders:
                for img_file in folder.glob("*.png"):
                    if img_file.name in needed_images:
                        shutil.copy2(img_file, images_dir / img_file.name)
                        extracted += 1
                        needed_images.discard(img_file.name)

                        if extracted % 1000 == 0:
                            logger.info(f"  Copied {extracted:,} images...")

            logger.info(f"Copied {extracted:,} images from extracted folders")
            return

    # Extract from tar files
    logger.info(f"Found {len(tar_files)} tar.gz files to search")

    extracted = 0
    for tar_path in tar_files:
        if not needed_images:
            break

        logger.info(f"Searching {tar_path.name}...")

        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if not needed_images:
                        break

                    img_name = Path(member.name).name
                    if img_name in needed_images:
                        # Extract to output directory
                        member.name = img_name  # Flatten path
                        tar.extract(member, images_dir)
                        needed_images.discard(img_name)
                        extracted += 1

                        if extracted % 500 == 0:
                            logger.info(f"  Extracted {extracted:,} images...")
        except Exception as e:
            logger.warning(f"Error reading {tar_path}: {e}")
            continue

    logger.info(f"Total extracted: {extracted:,} images")

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
    metadata_path = output_dir / "metadata.csv"
    final_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save split info
    split_info = {
        'train': len(train_df),
        'canary': len(canary_df),
        'test': len(test_df),
        'total': len(final_df),
        'seed': seed,
        'classes': sorted(df['dx'].unique().tolist())
    }

    import yaml
    with open(output_dir / "split_info.yaml", 'w') as f:
        yaml.dump(split_info, f, default_flow_style=False)

    logger.info(f"\nSplit distribution:")
    logger.info(f"  Train:  {len(train_df):,} ({len(train_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Canary: {len(canary_df):,} ({len(canary_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Test:   {len(test_df):,} ({len(test_df)/len(final_df)*100:.1f}%)")

    # Per-class canary distribution
    logger.info(f"\nCanary samples per class:")
    canary_counts = canary_df['dx'].value_counts()
    for cls, count in canary_counts.items():
        logger.info(f"  {cls:25s}: {count:,}")


def update_config(output_dir: Path, n_samples: int) -> None:
    """Update the chestxray config file."""
    import yaml

    config_path = PROJECT_ROOT / "config" / "datasets" / "chestxray.yaml"

    # Read existing config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update paths
    config['name'] = 'chestxray'
    config['description'] = f'NIH ChestXray - {n_samples//1000}K Stratified Sample'
    config['paths']['images'] = str(output_dir / "images")
    config['paths']['metadata'] = str(output_dir / "metadata.csv")

    # Update split counts (will be set by preprocessing)
    # These are approximate, actual counts set after split creation

    # Backup old config
    backup_path = config_path.with_suffix('.yaml.backup')
    shutil.copy2(config_path, backup_path)

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated config: {config_path}")
    logger.info(f"Backup saved: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download NIH ChestX-ray14 with stratified sampling'
    )
    parser.add_argument('--target-samples', type=int, default=50000,
                        help='Target number of samples (default: 50000)')
    parser.add_argument('--min-per-class', type=int, default=500,
                        help='Minimum samples per disease class (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/chestxray_50k)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, use existing data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to existing downloaded data')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NIH CHESTX-RAY14 DOWNLOAD & STRATIFIED SAMPLING")
    logger.info("=" * 60)
    logger.info(f"Target samples: {args.target_samples:,}")
    logger.info(f"Min per class: {args.min_per_class}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Step 1: Download or locate data
    if args.skip_download and args.data_path:
        data_path = Path(args.data_path)
        logger.info(f"Using existing data at: {data_path}")
    else:
        data_path = download_dataset()

    # Step 2: Load and process metadata
    full_df, disease_df = load_metadata(data_path)

    # Step 3: Create stratified sample
    sample_df = compute_stratified_sample(
        disease_df,
        target_samples=args.target_samples,
        min_per_class=args.min_per_class,
        seed=args.seed
    )

    # Step 4: Extract images
    extract_images(data_path, sample_df, output_dir)

    # Step 5: Create splits
    create_splits(sample_df, output_dir, seed=args.seed)

    # Step 6: Update config
    update_config(output_dir, args.target_samples)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {len(sample_df):,}")
    logger.info(f"Images directory: {output_dir / 'images'}")
    logger.info(f"Metadata: {output_dir / 'metadata.csv'}")
    logger.info("")
    logger.info("Next step: Run training with ./run_all.sh")


if __name__ == "__main__":
    main()
