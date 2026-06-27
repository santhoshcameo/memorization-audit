#!/usr/bin/env python3
"""
Download Kvasir-Capsule Dataset

Downloads and extracts the Kvasir-Capsule labeled images dataset.

Dataset: https://datasets.simula.no/kvasir-capsule/
Paper: https://www.nature.com/articles/s41597-021-00920-z

Total: 47,238 labeled images across 14 classes
Extreme class imbalance (3434:1 ratio):
  - Normal clean mucosa: 34,338 (72.7%)  - DOMINANT
  - Ampulla of vater: 10 (0.02%)         - EXTREMELY RARE
  - Blood - hematin: 12 (0.03%)          - EXTREMELY RARE
  - Polyp: 55 (0.12%)                    - VERY RARE
  - Erythema: 159 (0.34%)                - VERY RARE
  - Blood - fresh: 446 (0.94%)           - RARE
  - Erosion: 506 (1.07%)                 - RARE
  - Lymphangiectasia: 592 (1.25%)        - RARE
  - Foreign body: 776 (1.64%)            - RARE
  - Angiectasia: 866 (1.83%)             - RARE
  - Ulcer: 854 (1.81%)                   - RARE
  - Pylorus: 1,529 (3.24%)               - MODERATE
  - Reduced mucosal view: 2,906 (6.15%)  - MODERATE
  - Ileocecal valve: 4,189 (8.87%)       - COMMON

Usage:
    python scripts/download_kvasir_capsule.py
    python scripts/download_kvasir_capsule.py --output-dir data/kvasir_capsule
"""

import argparse
import os
import sys
import zipfile
import tarfile
import shutil
import subprocess
from pathlib import Path
import urllib.request
import ssl
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DOWNLOAD_URL = "https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-labeled-images.zip"
EXPECTED_CLASSES = 14
MIN_EXPECTED_IMAGES = 45000


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar, handling SSL issues"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try wget first with full path (most reliable for large files)
    wget_paths = ['/usr/bin/wget', '/usr/local/bin/wget', shutil.which('wget')]
    for wget_path in wget_paths:
        if wget_path and os.path.exists(wget_path):
            print(f"Using wget ({wget_path}) for download...")
            try:
                result = subprocess.run(
                    [wget_path, '--no-check-certificate', '-O', str(output_path), url],
                    check=True
                )
                print("Download complete via wget")
                return
            except subprocess.CalledProcessError as e:
                print(f"wget failed: {e}, trying alternative methods...")
                break

    # Try curl with full path
    curl_paths = ['/usr/bin/curl', '/usr/local/bin/curl', shutil.which('curl')]
    for curl_path in curl_paths:
        if curl_path and os.path.exists(curl_path):
            print(f"Using curl ({curl_path}) for download...")
            try:
                result = subprocess.run(
                    [curl_path, '-k', '-L', '-o', str(output_path), url],
                    check=True
                )
                print("Download complete via curl")
                return
            except subprocess.CalledProcessError as e:
                print(f"curl failed: {e}, trying urllib with SSL workaround...")
                break

    # Fallback: urllib with SSL context that doesn't verify
    print("Using urllib with SSL workaround...")

    # Create unverified SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

    # Install the SSL context
    https_handler = urllib.request.HTTPSHandler(context=ssl_context)
    opener = urllib.request.build_opener(https_handler)
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print()  # Newline after progress


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip file with progress"""
    print(f"Extracting: {zip_path}")
    print(f"To: {extract_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        total = len(members)
        for i, member in enumerate(members):
            if (i + 1) % 1000 == 0 or i == total - 1:
                print(f"\r  Extracting: {i+1}/{total}", end='', flush=True)
            zip_ref.extract(member, extract_dir)
    print()


def organize_images(extract_dir: Path, output_dir: Path):
    """
    Organize Kvasir-Capsule images into flat structure.

    The zip contains: labelled_images/<class_name>.tar.gz
    Each tar.gz contains: <class_name>/<image>.jpg

    We extract all tar.gz files and flatten to: images/<class>_<image>.jpg
    """
    print("Organizing images...")

    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find the labelled_images directory
    labeled_dir = None
    for possible_path in [
        extract_dir / 'labelled_images',
        extract_dir / 'labeled_images',
        extract_dir / 'kvasir-capsule-labeled-images' / 'labelled_images',
        extract_dir / 'kvasir-capsule-labeled-images' / 'labeled_images',
    ]:
        if possible_path.exists():
            labeled_dir = possible_path
            break

    if labeled_dir is None:
        # Search for it
        for root, dirs, files in os.walk(extract_dir):
            for d in dirs:
                if 'label' in d.lower() and 'image' in d.lower():
                    labeled_dir = Path(root) / d
                    break
            if labeled_dir:
                break

    if labeled_dir is None:
        print(f"ERROR: Could not find labeled_images directory in {extract_dir}")
        print("Contents:")
        for item in extract_dir.rglob('*'):
            print(f"  {item}")
        sys.exit(1)

    print(f"Found labeled images at: {labeled_dir}")

    # Check if we have tar.gz files or directories
    tar_files = list(labeled_dir.glob('*.tar.gz'))
    class_dirs = [d for d in labeled_dir.iterdir() if d.is_dir()]

    records = []
    class_name_map = {}

    # If we have tar.gz files, extract them first
    if tar_files:
        print(f"Found {len(tar_files)} tar.gz files, extracting...")
        temp_extract_dir = output_dir / 'temp_extract'
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        for tar_file in sorted(tar_files):
            class_name = tar_file.stem.replace('.tar', '')  # Remove .tar from .tar.gz
            normalized_name = class_name.lower().replace(' ', '_').replace('-', '_')
            class_name_map[class_name] = normalized_name

            print(f"  Extracting {tar_file.name}...")
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(temp_extract_dir)
            except Exception as e:
                print(f"    Error extracting {tar_file}: {e}")
                continue

            # Find extracted images - check all subdirectories (case-insensitive)
            image_files = []
            for subdir in temp_extract_dir.iterdir():
                if subdir.is_dir():
                    image_files.extend(list(subdir.glob('*.jpg')))
                    image_files.extend(list(subdir.glob('*.jpeg')))
                    image_files.extend(list(subdir.glob('*.png')))

            # Also check root of temp_extract_dir
            image_files.extend(list(temp_extract_dir.glob('*.jpg')))
            image_files.extend(list(temp_extract_dir.glob('*.jpeg')))
            image_files.extend(list(temp_extract_dir.glob('*.png')))

            for img_file in image_files:
                image_id = f"{normalized_name}_{img_file.stem}"
                new_name = f"{image_id}{img_file.suffix}"
                dest_path = images_dir / new_name

                shutil.copy2(img_file, dest_path)

                records.append({
                    'image_id': new_name,
                    'dx': normalized_name,
                    'original_class': class_name,
                    'original_file': img_file.name
                })

            # Clean up temp directory after each extraction to save space
            for item in temp_extract_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # Remove temp directory
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)

    # If we have directories directly (already extracted)
    elif class_dirs:
        print(f"Found {len(class_dirs)} class directories")
        for class_dir in sorted(class_dirs):
            class_name = class_dir.name.strip()
            normalized_name = class_name.lower().replace(' ', '_').replace('-', '_')
            class_name_map[class_name] = normalized_name

            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))

            for img_file in image_files:
                image_id = f"{normalized_name}_{img_file.stem}"
                new_name = f"{image_id}{img_file.suffix}"
                dest_path = images_dir / new_name

                shutil.copy2(img_file, dest_path)

                records.append({
                    'image_id': new_name,
                    'dx': normalized_name,
                    'original_class': class_name,
                    'original_file': img_file.name
                })

    print(f"Organized {len(records)} images across {len(class_name_map)} classes")

    return pd.DataFrame(records), class_name_map


def create_splits(df: pd.DataFrame, output_dir: Path, seed: int = 42):
    """Create stratified train/canary/test splits"""
    print("\nCreating stratified splits...")

    np.random.seed(seed)

    # For extremely rare classes (n < 10), we need special handling
    min_samples_per_class = df.groupby('dx').size().min()
    print(f"Smallest class has {min_samples_per_class} samples")

    # Split ratios: 70% train, 15% canary, 15% test
    # For tiny classes, ensure at least 1 sample in each split

    # First, separate tiny classes (n < 15) for special handling
    class_counts = df.groupby('dx').size()
    tiny_classes = class_counts[class_counts < 15].index.tolist()
    normal_classes = class_counts[class_counts >= 15].index.tolist()

    all_train = []
    all_canary = []
    all_test = []

    # Handle tiny classes: 1 canary, 1 test, rest train
    for cls in tiny_classes:
        cls_df = df[df['dx'] == cls].copy()
        n = len(cls_df)

        # Shuffle
        cls_df = cls_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # At minimum: 1 test, 1 canary, rest train
        if n >= 3:
            all_test.append(cls_df.iloc[:1])
            all_canary.append(cls_df.iloc[1:2])
            all_train.append(cls_df.iloc[2:])
        elif n == 2:
            all_test.append(cls_df.iloc[:1])
            all_canary.append(cls_df.iloc[1:2])
        else:  # n == 1
            all_canary.append(cls_df.iloc[:1])  # Put in canary for memorization test

        print(f"  {cls}: {n} total -> train={max(0,n-2)}, canary=1, test={min(1,n-1)}")

    # Handle normal classes with stratified split
    if normal_classes:
        normal_df = df[df['dx'].isin(normal_classes)].copy()

        # First split: train+canary (85%) vs test (15%)
        train_canary, test = train_test_split(
            normal_df,
            test_size=0.15,
            stratify=normal_df['dx'],
            random_state=seed
        )

        # Second split: train (70/85 = 82.35%) vs canary (15/85 = 17.65%)
        train, canary = train_test_split(
            train_canary,
            test_size=0.15/0.85,
            stratify=train_canary['dx'],
            random_state=seed
        )

        all_train.append(train)
        all_canary.append(canary)
        all_test.append(test)

    # Combine
    train_df = pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame()
    canary_df = pd.concat(all_canary, ignore_index=True) if all_canary else pd.DataFrame()
    test_df = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()

    # Add split column
    train_df['split'] = 'train'
    canary_df['split'] = 'canary'
    test_df['split'] = 'test'

    # Combine all
    final_df = pd.concat([train_df, canary_df, test_df], ignore_index=True)

    # Report
    print(f"\nSplit summary:")
    print(f"  Train:  {len(train_df):,} samples")
    print(f"  Canary: {len(canary_df):,} samples")
    print(f"  Test:   {len(test_df):,} samples")
    print(f"  Total:  {len(final_df):,} samples")

    # Per-class distribution
    print("\nPer-class distribution:")
    print("-" * 80)
    print(f"{'Class':<25s} {'Total':>8s} {'Train':>8s} {'Canary':>8s} {'Test':>8s} {'%':>8s} {'Tier':<12s}")
    print("-" * 80)

    total = len(final_df)
    for cls in sorted(final_df['dx'].unique()):
        cls_total = len(final_df[final_df['dx'] == cls])
        cls_train = len(train_df[train_df['dx'] == cls])
        cls_canary = len(canary_df[canary_df['dx'] == cls])
        cls_test = len(test_df[test_df['dx'] == cls])
        pct = cls_total / total * 100

        # Determine rarity tier
        if pct < 0.5:
            tier = "VERY_RARE"
        elif pct < 2:
            tier = "RARE"
        elif pct < 5:
            tier = "MODERATE"
        else:
            tier = "COMMON"

        print(f"{cls:<25s} {cls_total:8d} {cls_train:8d} {cls_canary:8d} {cls_test:8d} {pct:7.2f}% {tier:<12s}")

    print("-" * 80)

    # Save metadata
    metadata_path = output_dir / 'kvasir_capsule_metadata.csv'
    final_df.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata: {metadata_path}")

    # Save split info
    split_info = {
        'dataset': 'kvasir_capsule',
        'seed': seed,
        'total_samples': len(final_df),
        'train': len(train_df),
        'canary': len(canary_df),
        'test': len(test_df),
        'n_classes': len(final_df['dx'].unique()),
        'classes': sorted(final_df['dx'].unique().tolist()),
    }

    import yaml
    split_info_path = output_dir / 'split_info.yaml'
    with open(split_info_path, 'w') as f:
        yaml.dump(split_info, f, default_flow_style=False)
    print(f"Saved split info: {split_info_path}")

    return final_df


def main():
    parser = argparse.ArgumentParser(
        description='Download Kvasir-Capsule dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MANUAL DOWNLOAD OPTION:
If automatic download fails due to SSL issues, you can:
1. Download the file manually from your browser:
   https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-labeled-images.zip

2. Place it in data/kvasir_capsule/ directory

3. Run this script with --from-zip:
   python scripts/download_kvasir_capsule.py --from-zip data/kvasir_capsule/kvasir-capsule-labeled-images.zip
        """
    )
    parser.add_argument('--output-dir', '-o', type=str,
                       default='data/kvasir_capsule',
                       help='Output directory (default: data/kvasir_capsule)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for splits (default: 42)')
    parser.add_argument('--keep-zip', action='store_true',
                       help='Keep downloaded zip file')
    parser.add_argument('--from-zip', type=str, default=None,
                       help='Path to pre-downloaded zip file (skip download)')
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 80)
    print("KVASIR-CAPSULE DATASET DOWNLOAD")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()

    # Check if already processed
    metadata_path = output_dir / 'kvasir_capsule_metadata.csv'
    images_dir = output_dir / 'images'

    if metadata_path.exists() and images_dir.exists():
        df = pd.read_csv(metadata_path)
        if 'split' in df.columns and len(df) > MIN_EXPECTED_IMAGES:
            print(f"Dataset already processed: {len(df)} images")
            print("Use --force flag to re-download (not implemented)")
            return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine zip path
    if args.from_zip:
        # Use pre-downloaded zip
        zip_path = Path(args.from_zip)
        if not zip_path.exists():
            print(f"ERROR: Specified zip file not found: {zip_path}")
            sys.exit(1)
        print(f"Using pre-downloaded zip: {zip_path}")
    else:
        # Download
        zip_path = output_dir / 'kvasir-capsule-labeled-images.zip'
        if not zip_path.exists():
            download_with_progress(DOWNLOAD_URL, zip_path)
        else:
            print(f"Using existing zip: {zip_path}")

    # Extract
    extract_dir = output_dir / 'extracted'
    if not extract_dir.exists():
        extract_zip(zip_path, extract_dir)
    else:
        print(f"Using existing extraction: {extract_dir}")

    # Organize
    df, class_map = organize_images(extract_dir, output_dir)

    # Verify classes
    n_classes = len(df['dx'].unique())
    if n_classes != EXPECTED_CLASSES:
        print(f"WARNING: Expected {EXPECTED_CLASSES} classes, found {n_classes}")

    # Create splits
    final_df = create_splits(df, output_dir, args.seed)

    # Cleanup
    if not args.keep_zip and zip_path.exists():
        print(f"\nRemoving zip file: {zip_path}")
        zip_path.unlink()

    if extract_dir.exists():
        print(f"Removing extraction directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nDataset ready at: {output_dir}")
    print(f"Images: {output_dir / 'images'}")
    print(f"Metadata: {metadata_path}")
    print(f"\nNext steps:")
    print(f"  1. Verify data: python -c \"import pandas as pd; print(pd.read_csv('{metadata_path}').groupby('dx').size())\"")
    print(f"  2. Run experiment: python run_experiment.py --experiment kvasir_baseline")


if __name__ == '__main__':
    main()
