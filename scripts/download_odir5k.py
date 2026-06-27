#!/usr/bin/env python3
"""
ODIR-5K (Ocular Disease Intelligent Recognition) Downloader

Downloads ODIR-5K dataset and creates stratified splits for memorization study.

Dataset contains fundus photographs with 8 disease classes:
- N: Normal
- D: Diabetes
- G: Glaucoma
- C: Cataract
- A: Age-related Macular Degeneration (AMD)
- H: Hypertension
- M: Pathological Myopia
- O: Other diseases/abnormalities

Usage:
    python scripts/download_odir5k.py
    python scripts/download_odir5k.py --min-per-class 50
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "odir5k"

# Disease class mapping
DISEASE_CLASSES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'AMD',
    'H': 'Hypertension',
    'M': 'Myopia',
    'O': 'Other'
}


def download_dataset():
    """Download ODIR-5K using kagglehub."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING ODIR-5K DATASET")
    logger.info("=" * 60)

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading from Kaggle...")
    path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")
    logger.info(f"Dataset downloaded to: {path}")

    return Path(path)


def load_and_process_metadata(data_path: Path) -> pd.DataFrame:
    """Load and process ODIR-5K metadata."""
    logger.info("Loading metadata...")

    # Find the labels CSV file
    csv_files = list(data_path.rglob("*.csv"))

    # Look for the main data file
    data_file = None
    for f in csv_files:
        if 'full_df' in f.name.lower() or 'data' in f.name.lower() or 'odir' in f.name.lower():
            data_file = f
            break

    if not data_file and csv_files:
        data_file = csv_files[0]

    if not data_file:
        raise FileNotFoundError(f"No CSV file found in {data_path}")

    logger.info(f"Found metadata: {data_file}")
    df = pd.read_csv(data_file)

    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Total rows: {len(df)}")

    # ODIR-5K has left and right eye images with diagnostic keywords
    # We need to process this into a usable format

    # Check for expected columns
    if 'Left-Diagnostic Keywords' in df.columns:
        # Standard ODIR format
        processed_rows = []

        for idx, row in df.iterrows():
            patient_id = row.get('ID', idx)

            # Process left eye
            left_img = row.get('Left-Fundus', None)
            left_diag = row.get('Left-Diagnostic Keywords', '')
            if pd.notna(left_img) and left_img and str(left_img).strip():
                dx = extract_primary_diagnosis(left_diag)
                if dx:
                    processed_rows.append({
                        'image_id': str(left_img),
                        'patient_id': patient_id,
                        'eye': 'left',
                        'dx': dx,
                        'dx_full': left_diag
                    })

            # Process right eye
            right_img = row.get('Right-Fundus', None)
            right_diag = row.get('Right-Diagnostic Keywords', '')
            if pd.notna(right_img) and right_img and str(right_img).strip():
                dx = extract_primary_diagnosis(right_diag)
                if dx:
                    processed_rows.append({
                        'image_id': str(right_img),
                        'patient_id': patient_id,
                        'eye': 'right',
                        'dx': dx,
                        'dx_full': right_diag
                    })

        df = pd.DataFrame(processed_rows)

    elif 'target' in df.columns or 'label' in df.columns or 'labels' in df.columns:
        # Alternative format with direct labels
        label_col = 'target' if 'target' in df.columns else ('label' if 'label' in df.columns else 'labels')
        img_col = [c for c in df.columns if 'image' in c.lower() or 'file' in c.lower() or 'fundus' in c.lower()]

        if img_col:
            df = df.rename(columns={img_col[0]: 'image_id', label_col: 'dx'})

    logger.info(f"Processed {len(df)} images")

    # Show class distribution
    if 'dx' in df.columns:
        logger.info("\nClass distribution:")
        for dx, count in df['dx'].value_counts().items():
            pct = count / len(df) * 100
            full_name = DISEASE_CLASSES.get(dx, dx)
            rarity = "RARE" if pct < 5 else ""
            logger.info(f"  {dx} ({full_name:15s}): {count:5d} ({pct:5.2f}%) {rarity}")

    return df


def extract_primary_diagnosis(diag_str: str) -> str:
    """Extract primary diagnosis from diagnostic keywords string."""
    if pd.isna(diag_str) or not diag_str:
        return None

    diag_str = str(diag_str).strip().lower()

    # Priority order for classification
    if 'normal' in diag_str:
        return 'Normal'
    elif 'diabetes' in diag_str or 'diabetic' in diag_str:
        return 'Diabetes'
    elif 'glaucoma' in diag_str:
        return 'Glaucoma'
    elif 'cataract' in diag_str:
        return 'Cataract'
    elif 'macular' in diag_str or 'amd' in diag_str or 'age-related' in diag_str:
        return 'AMD'
    elif 'hypertension' in diag_str or 'hypertensive' in diag_str:
        return 'Hypertension'
    elif 'myopia' in diag_str or 'pathological myopia' in diag_str:
        return 'Myopia'
    else:
        return 'Other'


def copy_images(data_path: Path, df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Copy images to output directory and update paths."""
    logger.info("=" * 60)
    logger.info("COPYING IMAGES")
    logger.info("=" * 60)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find image directories
    image_dirs = []
    for pattern in ['**/ODIR-5K_Training_Images', '**/Training Images', '**/images', '**/preprocessed_images']:
        image_dirs.extend(data_path.glob(pattern))

    if not image_dirs:
        # Try to find any directory with images
        for d in data_path.rglob('*'):
            if d.is_dir() and any(d.glob('*.jpg')) or any(d.glob('*.png')):
                image_dirs.append(d)

    logger.info(f"Found {len(image_dirs)} image directories")

    # Build image lookup
    all_images = {}
    for img_dir in image_dirs:
        for img_file in img_dir.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_images[img_file.name] = img_file

    logger.info(f"Found {len(all_images)} total images")

    # Copy images and track which ones we found
    valid_rows = []
    copied = 0

    for idx, row in df.iterrows():
        img_name = row['image_id']

        # Try to find the image
        src_path = all_images.get(img_name)

        if not src_path:
            # Try with different extensions
            base_name = Path(img_name).stem
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_name = base_name + ext
                if alt_name in all_images:
                    src_path = all_images[alt_name]
                    break

        if src_path and src_path.exists():
            dst_path = images_dir / src_path.name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

            row_copy = row.copy()
            row_copy['image_id'] = src_path.name
            valid_rows.append(row_copy)

            if copied % 500 == 0:
                logger.info(f"  Copied {copied} images...")

    logger.info(f"Copied {copied} new images")
    logger.info(f"Valid samples: {len(valid_rows)}")

    return pd.DataFrame(valid_rows)


def create_stratified_splits(df: pd.DataFrame,
                             output_dir: Path,
                             min_per_class: int = 30,
                             train_ratio: float = 0.70,
                             canary_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             seed: int = 42) -> None:
    """Create stratified train/canary/test splits."""
    logger.info("=" * 60)
    logger.info("CREATING STRATIFIED SPLITS")
    logger.info("=" * 60)

    from sklearn.model_selection import train_test_split

    np.random.seed(seed)

    # Filter classes with minimum samples
    class_counts = df['dx'].value_counts()
    valid_classes = class_counts[class_counts >= min_per_class].index.tolist()

    logger.info(f"Classes with >= {min_per_class} samples: {valid_classes}")

    df_filtered = df[df['dx'].isin(valid_classes)].copy()
    logger.info(f"Samples after filtering: {len(df_filtered)}")

    # Stratified split
    try:
        train_canary_df, test_df = train_test_split(
            df_filtered,
            test_size=test_ratio,
            stratify=df_filtered['dx'],
            random_state=seed
        )

        canary_adjusted_ratio = canary_ratio / (train_ratio + canary_ratio)
        train_df, canary_df = train_test_split(
            train_canary_df,
            test_size=canary_adjusted_ratio,
            stratify=train_canary_df['dx'],
            random_state=seed
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}")
        logger.warning("Falling back to random split")

        train_canary_df, test_df = train_test_split(
            df_filtered,
            test_size=test_ratio,
            random_state=seed
        )

        canary_adjusted_ratio = canary_ratio / (train_ratio + canary_ratio)
        train_df, canary_df = train_test_split(
            train_canary_df,
            test_size=canary_adjusted_ratio,
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
    import yaml
    split_info = {
        'train': int(len(train_df)),
        'canary': int(len(canary_df)),
        'test': int(len(test_df)),
        'total': int(len(final_df)),
        'seed': seed,
        'classes': sorted(df_filtered['dx'].unique().tolist()),
        'num_classes': len(valid_classes)
    }

    with open(output_dir / "split_info.yaml", 'w') as f:
        yaml.dump(split_info, f, default_flow_style=False)

    logger.info(f"\nSplit distribution:")
    logger.info(f"  Train:  {len(train_df):,} ({len(train_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Canary: {len(canary_df):,} ({len(canary_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Test:   {len(test_df):,} ({len(test_df)/len(final_df)*100:.1f}%)")

    # Per-class canary distribution
    logger.info(f"\nCanary samples per class:")
    canary_counts = canary_df['dx'].value_counts()
    for cls in sorted(canary_counts.index):
        count = canary_counts[cls]
        total = class_counts.get(cls, 0)
        logger.info(f"  {cls:15s}: {count:4d} canaries (of {total} total)")


def create_dataset_config(output_dir: Path, num_classes: int) -> None:
    """Create the dataset configuration file."""
    import yaml

    config = {
        'name': 'odir5k',
        'description': 'ODIR-5K Ocular Disease Recognition Dataset',
        'info': {
            'num_classes': num_classes,
            'modality': 'Fundus Photography',
            'domain': 'Ophthalmology'
        },
        'paths': {
            'images': str(output_dir / "images"),
            'metadata': str(output_dir / "metadata.csv")
        },
        'classes': {
            'names': ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
        },
        'preprocessing': {
            'target_size': 224,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'augmentation': {
            'use_dataset_default': True
        }
    }

    config_path = PROJECT_ROOT / "config" / "datasets" / "odir5k.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download ODIR-5K dataset for memorization study'
    )
    parser.add_argument('--min-per-class', type=int, default=30,
                        help='Minimum samples per class (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/odir5k)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, use existing data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to existing downloaded data')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ODIR-5K DATASET DOWNLOAD & PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Step 1: Download or locate data
    if args.skip_download and args.data_path:
        data_path = Path(args.data_path)
        logger.info(f"Using existing data at: {data_path}")
    else:
        data_path = download_dataset()

    # Step 2: Load and process metadata
    df = load_and_process_metadata(data_path)

    # Step 3: Copy images
    df = copy_images(data_path, df, output_dir)

    # Step 4: Create splits
    create_stratified_splits(df, output_dir, min_per_class=args.min_per_class, seed=args.seed)

    # Step 5: Create config
    num_classes = df['dx'].nunique()
    create_dataset_config(output_dir, num_classes)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {len(df):,}")
    logger.info("")
    logger.info("Next: Run ./run_all.sh --only odir5k")


if __name__ == "__main__":
    main()
