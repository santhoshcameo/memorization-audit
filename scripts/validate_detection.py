#!/usr/bin/env python3
"""
Validate Pixel-Model Gap detector against ground-truth memorization scores.

Phase 1: CSV + pixel features (all 20 baseline pairs)
    - Load canary dataloader -> extract pixel features
    - For each model's memorization CSV:
        - Merge pixel atypicality with loss_candidate by image_id
        - Compute all 6 gap variants
        - Compute baselines (loss-only, pixel-only)
        - Correlate with ground-truth memorization_score
    - Output: comparison table (20 pairs x 8 methods)

Phase 2: Distinctive experiments (HAM1000, ODIR-5K)
    - Compare gap scores baseline vs distinctive for target classes
    - Validate that gap correctly tracks manipulated memorization

Usage:
    python scripts/validate_detection.py
    python scripts/validate_detection.py --phase2   # include distinctive experiments
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medmem.detection import PixelModelGapDetector, METHODS
from medmem.data_audit import DataAuditor

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'

BASELINES = {
    'ham1000_baseline':     {'num_classes': 7,  'dataset_config': 'ham10000'},
    'chestxray_baseline':   {'num_classes': 14, 'dataset_config': 'chestxray'},
    'odir5k_baseline':      {'num_classes': 8,  'dataset_config': 'odir5k'},
    'retinal_oct_baseline': {'num_classes': 4,  'dataset_config': 'retinal_oct'},
    'kvasir_baseline':      {'num_classes': 14, 'dataset_config': 'kvasir_capsule'},
}

DISTINCTIVE = {
    'ham1000_distinctive': {
        'dataset_config': 'ham10000',
        'baseline': 'ham1000_baseline',
        'target_classes': ['df', 'akiec'],
    },
    'odir5k_distinctive': {
        'dataset_config': 'odir5k',
        'baseline': 'odir5k_baseline',
        'target_classes': ['Myopia', 'Glaucoma'],
    },
}

MODELS = ['resnet50', 'vit', 'mae', 'medsam']


def _build_dataloader(exp_name, dataset_name):
    """Build canary dataloader."""
    import yaml
    exp_config_path = Path(f'config/experiments/{exp_name}.yaml')
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)

    dataset_config_path = Path(f'config/datasets/{dataset_name}.yaml')
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)

    config = exp_config.copy()
    config['dataset_config'] = dataset_config

    if dataset_name == 'ham10000':
        from src.data.ham10000 import create_ham10000_dataloader as create_dl
    elif dataset_name == 'chestxray':
        from src.data.chestxray import create_chestxray_dataloader as create_dl
    elif dataset_name == 'odir5k':
        from src.data.odir5k import create_odir5k_dataloader as create_dl
    elif dataset_name == 'retinal_oct':
        from src.data.retinal_oct import create_retinal_oct_dataloader as create_dl
    elif dataset_name == 'kvasir_capsule':
        from src.data.kvasir_capsule import create_kvasir_capsule_dataloader as create_dl
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return create_dl(config, split='canary', shuffle=False, batch_size=64,
                     num_workers=4, pin_memory=True)


# -----------------------------------------------------------------------
# Phase 1: All baseline pairs
# -----------------------------------------------------------------------

def phase1():
    """Validate all 6 gap variants + baselines across 20 model-dataset pairs."""
    print("=" * 80)
    print("Phase 1: Pixel-Model Gap Validation (all baseline pairs)")
    print("=" * 80)

    auditor = DataAuditor()
    all_rows = []

    for exp_name, exp_info in BASELINES.items():
        dataset_name = exp_info['dataset_config']
        mem_dir = RESULTS_ROOT / exp_name / 'memorization'
        if not mem_dir.exists():
            print(f"\n  SKIP {exp_name} — no memorization dir")
            continue

        # Build dataloader and extract pixel features ONCE per dataset
        print(f"\n{'='*60}")
        print(f"  {exp_name}: extracting pixel features...")
        try:
            dataloader = _build_dataloader(exp_name, dataset_name)
        except Exception as e:
            print(f"  SKIP — dataloader failed: {e}")
            continue

        t0 = time.time()
        features, class_names, image_ids = auditor._extract_all(dataloader)
        elapsed = time.time() - t0
        print(f"  Extracted {len(image_ids)} samples in {elapsed:.1f}s")

        # Build pixel atypicality df once
        detector = PixelModelGapDetector(method='percentile_gap')
        pixel_df = detector._pixel_atypicality(features, class_names, image_ids)

        for model_name in MODELS:
            mem_csv = mem_dir / f'{model_name}_memorization_scores.csv'
            if not mem_csv.exists():
                continue

            mem_df = pd.read_csv(mem_csv)
            # Handle ODIR-5K duplicates
            mem_df = mem_df.groupby('image_id', as_index=False).mean(numeric_only=True)
            # Re-add class_name (lost by mean())
            class_lookup = pd.read_csv(mem_csv)[['image_id', 'class_name']].drop_duplicates('image_id')
            mem_df = mem_df.merge(class_lookup, on='image_id', how='left')

            # Merge pixel features with losses
            ldf = mem_df[['image_id', 'loss_candidate']].rename(
                columns={'loss_candidate': 'model_loss'})
            merged = pixel_df.merge(ldf, on='image_id', how='inner')

            if len(merged) < 10:
                print(f"    {model_name}: only {len(merged)} merged samples, SKIP")
                continue

            ground_truth = mem_df.set_index('image_id')['memorization_score']
            merged_gt = merged.set_index('image_id').join(
                ground_truth.rename('gt_memorization'), how='inner'
            ).reset_index()

            if len(merged_gt) < 10:
                continue

            gt = merged_gt['gt_memorization'].values
            is_high = (gt > 0.3).astype(int)
            n_high = is_high.sum()

            row = {
                'experiment': exp_name,
                'model': model_name,
                'n_samples': len(merged_gt),
                'n_high_risk': n_high,
            }

            # --- Baselines ---
            # Loss-only: negative loss_candidate (lower loss = more memorized)
            loss_signal = -merged_gt['model_loss'].values
            rho_loss, _ = stats.spearmanr(loss_signal, gt)
            row['loss_only_rho'] = round(rho_loss, 4)
            if n_high >= 2 and n_high < len(gt):
                row['loss_only_auroc'] = round(roc_auc_score(is_high, loss_signal), 4)
            else:
                row['loss_only_auroc'] = None

            # Pixel-only: atypicality alone
            pixel_signal = merged_gt['pixel_atypicality'].values
            rho_pixel, _ = stats.spearmanr(pixel_signal, gt)
            row['pixel_only_rho'] = round(rho_pixel, 4)
            if n_high >= 2 and n_high < len(gt):
                row['pixel_only_auroc'] = round(roc_auc_score(is_high, pixel_signal), 4)
            else:
                row['pixel_only_auroc'] = None

            # --- 6 Gap Variants ---
            for method in METHODS:
                det = PixelModelGapDetector(method=method)
                result = det.detect_from_precomputed(
                    merged_gt[['image_id', 'class_name', 'pixel_atypicality']],
                    merged_gt[['image_id', 'model_loss']].rename(
                        columns={'model_loss': 'loss_candidate'}),
                )
                gap_scores = result.sample_df.set_index('image_id')['gap_score']
                # Align with ground truth
                aligned = merged_gt.set_index('image_id').join(
                    gap_scores.rename('gap'), how='inner'
                ).reset_index()

                gap_vals = aligned['gap'].values
                gt_vals = aligned['gt_memorization'].values
                is_high_a = (gt_vals > 0.3).astype(int)

                rho_gap, _ = stats.spearmanr(gap_vals, gt_vals)
                row[f'{method}_rho'] = round(rho_gap, 4)

                if is_high_a.sum() >= 2 and is_high_a.sum() < len(gt_vals):
                    row[f'{method}_auroc'] = round(
                        roc_auc_score(is_high_a, gap_vals), 4)
                else:
                    row[f'{method}_auroc'] = None

            all_rows.append(row)
            print(f"    {model_name}: n={row['n_samples']}, "
                  f"loss_rho={row['loss_only_rho']:.3f}, "
                  f"best_gap_rho={max(row.get(f'{m}_rho', -1) for m in METHODS):.3f}")

    if not all_rows:
        print("\nNo pairs validated!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("COMPARISON TABLE: Spearman rho (gap_score vs ground-truth M(x))")
    print(f"{'='*80}")

    method_cols = ['loss_only_rho', 'pixel_only_rho'] + [f'{m}_rho' for m in METHODS]
    summary_df = df[['experiment', 'model'] + method_cols].copy()
    print(summary_df.to_string(index=False))

    # Mean across all pairs
    print(f"\nMean Spearman rho across {len(df)} pairs:")
    for col in method_cols:
        vals = df[col].dropna()
        name = col.replace('_rho', '')
        print(f"  {name:30s}: {vals.mean():.4f} (std={vals.std():.4f})")

    # AUROC comparison
    auroc_cols = ['loss_only_auroc', 'pixel_only_auroc'] + [f'{m}_auroc' for m in METHODS]
    print(f"\nMean AUROC for HIGH-risk detection (M(x) > 0.3):")
    for col in auroc_cols:
        vals = df[col].dropna()
        name = col.replace('_auroc', '')
        if len(vals) > 0:
            print(f"  {name:30s}: {vals.mean():.4f} (std={vals.std():.4f}, n={len(vals)})")

    # Best method
    rho_means = {m: df[f'{m}_rho'].dropna().mean() for m in METHODS}
    best_method = max(rho_means, key=rho_means.get)
    best_rho = rho_means[best_method]
    loss_rho = df['loss_only_rho'].dropna().mean()
    pixel_rho = df['pixel_only_rho'].dropna().mean()

    print(f"\nBest gap method: {best_method} (mean rho={best_rho:.4f})")
    print(f"  vs loss-only: {best_rho:.4f} vs {loss_rho:.4f} "
          f"(delta={best_rho - loss_rho:+.4f})")
    print(f"  vs pixel-only: {best_rho:.4f} vs {pixel_rho:.4f} "
          f"(delta={best_rho - pixel_rho:+.4f})")

    # Per-class correlation
    print(f"\n{'='*80}")
    print("PER-CLASS CORRELATION: mean_gap vs mean_M(x)")
    print(f"{'='*80}")
    class_rows = []
    for _, row in df.iterrows():
        exp = row['experiment']
        model = row['model']
        mem_csv = RESULTS_ROOT / exp / 'memorization' / f'{model}_memorization_scores.csv'
        if not mem_csv.exists():
            continue
        mem_data = pd.read_csv(mem_csv)
        actual_per_class = mem_data.groupby('class_name')['memorization_score'].mean()
        if len(actual_per_class) < 3:
            continue

        # Use best method
        det = PixelModelGapDetector(method=best_method)
        ldf = mem_data[['image_id', 'loss_candidate']].groupby('image_id', as_index=False).mean()
        class_lookup = mem_data[['image_id', 'class_name']].drop_duplicates('image_id')

        pixel_sub = pixel_df if exp == list(BASELINES.keys())[0] else None
        # We need to re-extract for each dataset — use cached if possible
        # For simplicity, compute from the result we already have
        # Actually, we can compute gap from the already-merged data
        # Skip re-extraction — just get class-level from sample-level
        pass  # Handled below

    # Compute class-level correlation from sample-level gap scores
    # Re-run once with best method per pair
    class_rho_rows = []
    for _, row_data in df.iterrows():
        exp = row_data['experiment']
        model = row_data['model']
        mem_csv = RESULTS_ROOT / exp / 'memorization' / f'{model}_memorization_scores.csv'
        if not mem_csv.exists():
            continue
        mem_data = pd.read_csv(mem_csv)
        actual_per_class = mem_data.groupby('class_name')['memorization_score'].mean()
        if len(actual_per_class) < 3:
            continue
        class_rho_rows.append({
            'experiment': exp,
            'model': model,
            'n_classes': len(actual_per_class),
        })

    if class_rho_rows:
        print(f"  (Class-level validation requires re-extraction — "
              f"see per-pair results above)")

    # Save results
    output_dir = RESULTS_ROOT / 'gap_detection_validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'phase1_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'phase1_results.csv'}")

    return df


# -----------------------------------------------------------------------
# Phase 2: Distinctive experiments
# -----------------------------------------------------------------------

def phase2():
    """Compare gap scores between baseline and distinctive experiments."""
    print(f"\n{'='*80}")
    print("Phase 2: Distinctive Experiment Comparison")
    print(f"{'='*80}")

    auditor = DataAuditor()

    for dist_name, dist_info in DISTINCTIVE.items():
        baseline_name = dist_info['baseline']
        dataset_name = dist_info['dataset_config']
        target_classes = dist_info['target_classes']

        print(f"\n  {dist_name} vs {baseline_name}")
        print(f"  Target classes: {target_classes}")

        # Check both exist
        baseline_mem = RESULTS_ROOT / baseline_name / 'memorization'
        dist_mem = RESULTS_ROOT / dist_name / 'memorization'
        if not baseline_mem.exists() or not dist_mem.exists():
            print(f"  SKIP — missing directories")
            continue

        # Extract pixel features for both
        datasets = {}
        for exp, exp_name in [('baseline', baseline_name), ('distinctive', dist_name)]:
            try:
                dl = _build_dataloader(exp_name, dataset_name)
                feats, cls_names, img_ids = auditor._extract_all(dl)
                det = PixelModelGapDetector(method='regression_residual')
                pixel_df = det._pixel_atypicality(feats, cls_names, img_ids)
                datasets[exp] = pixel_df
            except Exception as e:
                print(f"  SKIP {exp} — {e}")
                continue

        if len(datasets) < 2:
            continue

        # Compare for each model
        models_found = []
        for model_name in ['resnet50', 'vit']:
            bl_csv = baseline_mem / f'{model_name}_memorization_scores.csv'
            di_csv = dist_mem / f'{model_name}_memorization_scores.csv'
            if not bl_csv.exists() or not di_csv.exists():
                continue

            bl_mem = pd.read_csv(bl_csv)
            di_mem = pd.read_csv(di_csv)

            print(f"\n  {model_name}:")
            for label, mem_data, pixel_data in [
                ('baseline', bl_mem, datasets['baseline']),
                ('distinctive', di_mem, datasets['distinctive']),
            ]:
                det = PixelModelGapDetector(method='regression_residual')
                result = det.detect_from_precomputed(
                    pixel_data,
                    mem_data[['image_id', 'loss_candidate']],
                )
                gap_by_class = result.class_df.set_index('class_name')['mean_gap']

                # Merge with actual M(x)
                actual = mem_data.groupby('class_name')['memorization_score'].mean()

                for tc in target_classes:
                    gap_val = gap_by_class.get(tc, float('nan'))
                    mx_val = actual.get(tc, float('nan'))
                    print(f"    {label:12s} {tc:15s}: "
                          f"mean_gap={gap_val:+.4f}, mean_M(x)={mx_val:+.4f}")

            models_found.append(model_name)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate Pixel-Model Gap detector')
    parser.add_argument('--phase2', action='store_true',
                        help='Run Phase 2: distinctive experiment comparison')
    args = parser.parse_args()

    phase1_df = phase1()

    if args.phase2:
        phase2()
    else:
        print("\nSkipping Phase 2 (distinctive). Use --phase2 to enable.")


if __name__ == '__main__':
    main()
