#!/usr/bin/env python3
"""
Data Preprocessing Script
Creates train/canary/test splits with stratification

This MUST be run before training to ensure:
1. Reproducible splits across all experiments
2. Balanced class distribution
3. No data leakage

Usage:
    python scripts/preprocess_data.py --dataset ham10000
    python scripts/preprocess_data.py --dataset chestxray
    python scripts/preprocess_data.py --dataset ham10000 --seed 42 --split-ratio 0.7,0.15,0.15
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DataPreprocessor:
    """Preprocesses dataset and creates stratified splits"""
    
    def __init__(self, dataset_name: str, seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        
        # Load dataset config
        config_path = PROJECT_ROOT / 'config' / 'datasets' / f'{dataset_name}.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed
        np.random.seed(seed)
        
        print("="*80)
        print(f"DATA PREPROCESSING: {dataset_name.upper()}")
        print("="*80)
        print(f"Seed: {seed}")
        print()
    
    def load_raw_metadata(self) -> pd.DataFrame:
        """Load raw metadata"""
        metadata_path = Path(self.config['paths']['metadata'])
        
        print(f"Loading metadata from: {metadata_path}")
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        print(f"✅ Loaded {len(df)} samples")
        
        return df
    
    def validate_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate metadata has required columns"""
        
        # Dataset-specific validation and preprocessing
        if self.dataset_name == 'ham10000':
            required_cols = ['image_id', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")
        
        elif self.dataset_name == 'chestxray':
            # Check if already processed
            if 'dx' in df.columns and 'split' in df.columns:
                print("  Already processed, skipping transformation")
                return df

            required_cols = ['Image Index', 'Finding Labels']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")

            # Rename columns to standard format
            df = df.rename(columns={
                'Image Index': 'image_id',
                'Finding Labels': 'dx'
            })

            # Take first finding only (some images have multiple labels)
            df['dx'] = df['dx'].apply(lambda x: x.split('|')[0])
            print(f"  Extracted primary findings")

            # Remove "No Finding" samples
            before_count = len(df)
            df = df[df['dx'] != 'No Finding'].copy()
            removed_count = before_count - len(df)
            print(f"  Removed {removed_count} 'No Finding' samples")

            # Show class distribution
            print(f"\n  Class distribution:")
            for cls, count in df['dx'].value_counts().items():
                print(f"    {cls:<30s}: {count:4d}")
            print()

        elif self.dataset_name == 'odir5k':
            # ODIR-5K: Check if already processed by download script
            if 'dx' in df.columns and 'split' in df.columns:
                print("  Already processed (splits created by download script)")
                return df

            # If not processed, check required columns
            required_cols = ['image_id', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")

            # Show class distribution
            print(f"\n  Class distribution:")
            for cls, count in df['dx'].value_counts().items():
                pct = count / len(df) * 100
                rarity = "RARE" if pct < 5 else ""
                print(f"    {cls:<20s}: {count:4d} ({pct:5.2f}%) {rarity}")
            print()

        elif self.dataset_name == 'retinal_oct':
            # Retinal OCT: Check if already processed by download script
            if 'dx' in df.columns and 'split' in df.columns:
                print("  Already processed (splits created by download script)")
                return df

            # If not processed, check required columns
            required_cols = ['image_id', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")

            # Show class distribution
            print(f"\n  Class distribution:")
            for cls, count in df['dx'].value_counts().items():
                pct = count / len(df) * 100
                rarity = "RARE" if pct < 5 else ""
                print(f"    {cls:<20s}: {count:4d} ({pct:5.2f}%) {rarity}")
            print()

        elif self.dataset_name == 'kvasir_capsule':
            # Kvasir-Capsule: Check if already processed by download script
            if 'dx' in df.columns and 'split' in df.columns:
                print("  Already processed (splits created by download script)")
                return df

            # If not processed, check required columns
            required_cols = ['image_id', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")

            # Show class distribution with rarity tiers
            print(f"\n  Class distribution (14 classes, 3434:1 imbalance):")
            for cls, count in df['dx'].value_counts().items():
                pct = count / len(df) * 100
                if pct < 0.5:
                    tier = "VERY_RARE"
                elif pct < 2:
                    tier = "RARE"
                elif pct < 5:
                    tier = "MODERATE"
                else:
                    tier = "COMMON"
                print(f"    {cls:<25s}: {count:5d} ({pct:6.2f}%) {tier}")
            print()

        else:
            # Generic validation
            required_cols = ['image_id', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")
        
        print(f"✅ Metadata validated")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Samples: {len(df)}")
        print()
        
        return df
    
    def check_image_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check that image files exist"""
        images_dir = Path(self.config['paths']['images'])
        
        print("Checking image files...")
        
        missing_images = []
        for image_id in df['image_id']:
            # Try .jpg first, then .png
            image_path = images_dir / f"{image_id}"
            if not image_path.exists():
                image_path = images_dir / f"{image_id.replace('.png', '.jpg').replace('.jpg', '')}.jpg"
            if not image_path.exists():
                image_path = images_dir / f"{image_id.replace('.png', '').replace('.jpg', '')}.png"
            if not image_path.exists():
                missing_images.append(image_id)
        
        if missing_images:
            print(f"⚠️ Warning: {len(missing_images)} images not found")
            print(f"   Examples: {missing_images[:5]}")
            
            # Remove missing images
            df = df[~df['image_id'].isin(missing_images)].copy()
            print(f"   Removed missing images. Remaining: {len(df)}")
        else:
            print(f"✅ All {len(df)} images found")
        
        print()
        return df
    
    def create_splits(self, df: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     canary_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> pd.DataFrame:
        """Create stratified train/canary/test splits"""
        
        print("Creating splits...")
        print(f"  Train:  {train_ratio*100:.0f}%")
        print(f"  Canary: {canary_ratio*100:.0f}%")
        print(f"  Test:   {test_ratio*100:.0f}%")
        print()
        
        # Verify ratios sum to 1
        if not np.isclose(train_ratio + canary_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        # First split: separate test set
        train_canary_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df['dx'],
            random_state=self.seed
        )
        
        # Second split: separate train and canary
        adjusted_canary_ratio = canary_ratio / (train_ratio + canary_ratio)
        train_df, canary_df = train_test_split(
            train_canary_df,
            test_size=adjusted_canary_ratio,
            stratify=train_canary_df['dx'],
            random_state=self.seed
        )
        
        # Add split column
        train_df['split'] = 'train'
        canary_df['split'] = 'canary'
        test_df['split'] = 'test'
        
        # Combine
        df_with_splits = pd.concat([train_df, canary_df, test_df], ignore_index=True)
        
        # Report
        print("Split sizes:")
        for split in ['train', 'canary', 'test']:
            split_df = df_with_splits[df_with_splits['split'] == split]
            print(f"  {split:8s}: {len(split_df):4d} samples")
        print()
        
        # Report per-class distribution
        print("Per-class distribution:")
        print("-"*80)
        print(f"{'Class':<30s} {'Train':>8s} {'Canary':>8s} {'Test':>8s} {'Total':>8s}")
        print("-"*80)
        
        for cls in sorted(df_with_splits['dx'].unique()):
            train_count = len(df_with_splits[(df_with_splits['dx']==cls) & (df_with_splits['split']=='train')])
            canary_count = len(df_with_splits[(df_with_splits['dx']==cls) & (df_with_splits['split']=='canary')])
            test_count = len(df_with_splits[(df_with_splits['dx']==cls) & (df_with_splits['split']=='test')])
            total = train_count + canary_count + test_count
            
            print(f"{cls:<30s} {train_count:8d} {canary_count:8d} {test_count:8d} {total:8d}")
        
        print("-"*80)
        print()
        
        return df_with_splits
    
    def save_processed_metadata(self, df: pd.DataFrame):
        """Save processed metadata with splits"""
        output_path = Path(self.config['paths']['metadata'])
        backup_path = output_path.with_suffix('.backup.csv')
        
        # Backup original if exists
        if output_path.exists():
            print(f"Creating backup: {backup_path.name}")
            df_original = pd.read_csv(output_path)
            df_original.to_csv(backup_path, index=False)
        
        # Save with splits
        print(f"Saving processed metadata: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} samples")
        
        # Save split info
        split_info = {
            'dataset': self.dataset_name,
            'seed': self.seed,
            'total_samples': len(df),
            'train': int((df['split'] == 'train').sum()),
            'canary': int((df['split'] == 'canary').sum()),
            'test': int((df['split'] == 'test').sum()),
            'classes': sorted(df['dx'].unique().tolist()),
        }
        
        info_path = output_path.parent / 'split_info.yaml'
        with open(info_path, 'w') as f:
            yaml.dump(split_info, f, default_flow_style=False)
        
        print(f"✅ Saved split info: {info_path.name}")
        print()
    
    def run(self, train_ratio: float = 0.7, 
            canary_ratio: float = 0.15,
            test_ratio: float = 0.15):
        """Run preprocessing pipeline"""
        
        # Load
        df = self.load_raw_metadata()
        
        # Validate (returns modified df)
        df = self.validate_metadata(df)
        
        # Check images
        df = self.check_image_files(df)
        
        # Create splits
        df_with_splits = self.create_splits(df, train_ratio, canary_ratio, test_ratio)
        
        # Save
        self.save_processed_metadata(df_with_splits)
        
        print("="*80)
        print("✅ PREPROCESSING COMPLETE")
        print("="*80)
        print()
        print("Next steps:")
        print(f"  1. Verify splits: Check {Path(self.config['paths']['metadata']).parent / 'split_info.yaml'}")
        print(f"  2. Run experiment: python run_experiment.py --experiment baseline")
        print()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Preprocess dataset and create splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default splits (70/15/15)
  python scripts/preprocess_data.py --dataset ham10000
  python scripts/preprocess_data.py --dataset chestxray
  
  # Custom splits
  python scripts/preprocess_data.py --dataset ham10000 --split-ratio 0.8,0.1,0.1
  
  # Different seed
  python scripts/preprocess_data.py --dataset ham10000 --seed 123
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset name (e.g., ham10000, chestxray)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=str,
        default='0.7,0.15,0.15',
        help='Train,canary,test ratios (default: 0.7,0.15,0.15)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing splits'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Parse split ratios
    ratios = [float(x) for x in args.split_ratio.split(',')]
    if len(ratios) != 3:
        print("Error: --split-ratio must have 3 values (train,canary,test)")
        sys.exit(1)
    
    train_ratio, canary_ratio, test_ratio = ratios
    
    # Check if splits already exist
    config_path = PROJECT_ROOT / 'config' / 'datasets' / f'{args.dataset}.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    metadata_path = Path(config['paths']['metadata'])
    
    if metadata_path.exists() and not args.force:
        df = pd.read_csv(metadata_path)
        if 'split' in df.columns:
            print(f"⚠️ Splits already exist in {metadata_path}")
            print(f"   Use --force to overwrite")
            sys.exit(1)
    
    # Run preprocessing
    preprocessor = DataPreprocessor(args.dataset, args.seed)
    preprocessor.run(train_ratio, canary_ratio, test_ratio)


if __name__ == '__main__':
    main()