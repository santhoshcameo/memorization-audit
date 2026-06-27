#!/usr/bin/env python3
"""
Validate ModelFeatureDetector (v2) against ground-truth memorization scores.

Loads checkpoints across 10 pairs (5 baselines × resnet50/vit), runs all 6
model-feature methods + 6 pixel-gap methods, computes Spearman rho, AUROC
(M(x)>0.3), and precision@15% against ground-truth differential M(x).

Usage:
    python scripts/validate_detection_v2.py                  # all 10 pairs
    python scripts/validate_detection_v2.py --pairs 3        # first 3 pairs
    python scripts/validate_detection_v2.py --aug 5           # 5 aug passes
    python scripts/validate_detection_v2.py --skip-pixel      # skip v1 methods
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import medmem
from medmem.detection import MODEL_FEATURE_METHODS, METHODS as PIXEL_METHODS

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'
OUTPUT_DIR = RESULTS_ROOT / 'detection_v2_validation'

BASELINES = {
    'ham1000_baseline':     {'num_classes': 7,  'dataset_config': 'ham10000'},
    'chestxray_baseline':   {'num_classes': 14, 'dataset_config': 'chestxray'},
    'odir5k_baseline':      {'num_classes': 8,  'dataset_config': 'odir5k'},
    'retinal_oct_baseline': {'num_classes': 4,  'dataset_config': 'retinal_oct'},
    'kvasir_baseline':      {'num_classes': 14, 'dataset_config': 'kvasir_capsule'},
}

MODELS = ['resnet50', 'vit']

HIGH_RISK_THRESHOLD = 0.3  # M(x) > 0.3 = HIGH


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _build_dataloader(exp_name, dataset_name, model_name):
    """Build a DataLoader for canary split."""
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

    return create_dl(
        config, split='canary', shuffle=False,
        batch_size=64, num_workers=4, pin_memory=True,
    )


def _load_model(exp_name, model_name, num_classes, device):
    """Load a trained model from checkpoint."""
    import yaml
    from src.models.base_model import ModelFactory

    exp_dir = RESULTS_ROOT / exp_name
    ckpt_path = (exp_dir / model_name / 'candidate' / 'checkpoints'
                 / f'{model_name}_candidate_best.pth')

    if not ckpt_path.exists():
        return None

    model_config_path = Path(f'config/models/{model_name}.yaml')
    if model_config_path.exists():
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
    else:
        model_config = {}

    model = ModelFactory.create(model_name, model_config, num_classes)
    model.load_checkpoint(str(ckpt_path))
    model.to(device).eval()
    return model


def _load_ground_truth(exp_name, model_name):
    """Load ground-truth memorization scores from CSV."""
    mem_csv = (RESULTS_ROOT / exp_name / 'memorization'
               / f'{model_name}_memorization_scores.csv')
    if not mem_csv.exists():
        return None
    df = pd.read_csv(mem_csv)
    # Handle ODIR-5K duplicates
    df = df.groupby('image_id', as_index=False).agg({
        'class_name': 'first',
        'memorization_score': 'mean',
        'loss_candidate': 'mean',
    })
    return df


def _compute_metrics(gap_scores, true_scores):
    """Compute Spearman rho, AUROC, and precision@15% from gap vs true M(x)."""
    from sklearn.metrics import roc_auc_score

    # Spearman correlation
    rho, p_val = stats.spearmanr(gap_scores, true_scores)

    # AUROC: binary label = M(x) > threshold
    labels = (true_scores > HIGH_RISK_THRESHOLD).astype(int)
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        auroc = float('nan')
    else:
        auroc = roc_auc_score(labels, gap_scores)

    # Precision@15%: of top 15% flagged, what fraction are truly HIGH?
    k = max(1, int(0.15 * len(gap_scores)))
    top_k_idx = np.argsort(gap_scores)[::-1][:k]
    prec_at_15 = labels[top_k_idx].mean()

    return {
        'rho': float(rho),
        'p_value': float(p_val),
        'auroc': float(auroc),
        'precision_at_15pct': float(prec_at_15),
        'n_high_risk': int(n_pos),
        'n_samples': len(gap_scores),
    }


# -----------------------------------------------------------------------
# Main validation
# -----------------------------------------------------------------------

def validate_pair(exp_name, model_name, num_classes, device,
                  num_augmentations, skip_pixel):
    """Run all detection methods on one (experiment, model) pair."""
    import torch

    print(f"\n{'='*60}")
    print(f"  {exp_name} / {model_name}")
    print(f"{'='*60}")

    # Load ground truth
    gt_df = _load_ground_truth(exp_name, model_name)
    if gt_df is None:
        print("  SKIP — no memorization CSV")
        return None

    # Load model
    try:
        model = _load_model(exp_name, model_name, num_classes, device)
    except Exception as e:
        print(f"  SKIP — model load failed: {e}")
        return None
    if model is None:
        print("  SKIP — no checkpoint")
        return None

    # Build dataloader
    dataset_name = BASELINES[exp_name]['dataset_config']
    try:
        dataloader = _build_dataloader(exp_name, dataset_name, model_name)
    except Exception as e:
        print(f"  SKIP — dataloader failed: {e}")
        del model
        torch.cuda.empty_cache()
        return None

    results = []

    # --- V2: ModelFeatureDetector (all 6 methods) ---
    # Run once with default method to extract signals, then reuse
    print(f"\n  Running ModelFeatureDetector ({num_augmentations} aug passes)...")
    t0 = time.time()

    try:
        # Use first method to get the full signals DataFrame
        first_method = MODEL_FEATURE_METHODS[0]
        detector = medmem.ModelFeatureDetector(
            method=first_method,
            num_augmentations=num_augmentations,
            feature_layer=None,
        )
        first_result = detector.detect(model, dataloader, device=device)

        # Extract signals from the sample_df for reuse
        signals_df = first_result.sample_df[[
            'image_id', 'class_name', 'feature_atypicality',
            'model_loss', 'prediction_entropy', 'aug_sensitivity',
        ]].copy()

        elapsed = time.time() - t0
        print(f"  Signal extraction: {elapsed:.1f}s")

        # Merge with ground truth
        merged = signals_df.merge(
            gt_df[['image_id', 'memorization_score']],
            on='image_id', how='inner',
        )
        if len(merged) < 10:
            print(f"  SKIP — only {len(merged)} samples matched GT")
            del model
            torch.cuda.empty_cache()
            return None

        true_scores = merged['memorization_score'].values

        # Run all 6 methods via detect_from_precomputed
        for method in MODEL_FEATURE_METHODS:
            det = medmem.ModelFeatureDetector(method=method)
            result = det.detect_from_precomputed(signals_df)

            # Align gap scores with merged order
            gap_df = result.sample_df[['image_id', 'gap_score']]
            m = merged[['image_id']].merge(gap_df, on='image_id', how='inner')
            gap_scores = m['gap_score'].values

            metrics = _compute_metrics(gap_scores, true_scores)
            metrics['experiment'] = exp_name
            metrics['model'] = model_name
            metrics['method'] = f'v2:{method}'
            metrics['detector'] = 'ModelFeatureDetector'
            results.append(metrics)

            print(f"    {method:30s}  rho={metrics['rho']:+.3f}  "
                  f"AUROC={metrics['auroc']:.3f}  P@15%={metrics['precision_at_15pct']:.3f}")

    except Exception as e:
        print(f"  ERROR (ModelFeatureDetector): {e}")
        import traceback
        traceback.print_exc()

    # --- V1: PixelModelGapDetector (all 6 methods) ---
    if not skip_pixel:
        print(f"\n  Running PixelModelGapDetector (v1 pixel methods)...")
        t0 = time.time()

        try:
            for method in PIXEL_METHODS:
                det = medmem.PixelModelGapDetector(method=method)
                result = det.detect(model, dataloader, device=device)

                gap_df = result.sample_df[['image_id', 'gap_score']]
                m = merged[['image_id']].merge(
                    gap_df, on='image_id', how='inner')
                gap_scores = m['gap_score'].values

                metrics = _compute_metrics(gap_scores, true_scores)
                metrics['experiment'] = exp_name
                metrics['model'] = model_name
                metrics['method'] = f'v1:{method}'
                metrics['detector'] = 'PixelModelGapDetector'
                results.append(metrics)

                print(f"    {method:30s}  rho={metrics['rho']:+.3f}  "
                      f"AUROC={metrics['auroc']:.3f}  P@15%={metrics['precision_at_15pct']:.3f}")

            elapsed = time.time() - t0
            print(f"  Pixel methods: {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR (PixelModelGapDetector): {e}")
            import traceback
            traceback.print_exc()

    # --- Also compute loss-only baseline ---
    if 'merged' in dir() and merged is not None and len(merged) > 0:
        # Loss-only: just use negative model_loss as gap_score
        # (lower loss → more likely memorized → higher score)
        loss_scores = -merged['model_loss'].values if 'model_loss' in merged.columns else -signals_df.merge(merged[['image_id']], on='image_id')['model_loss'].values
        metrics = _compute_metrics(loss_scores, true_scores)
        metrics['experiment'] = exp_name
        metrics['model'] = model_name
        metrics['method'] = 'baseline:loss_only'
        metrics['detector'] = 'loss_only'
        results.append(metrics)
        print(f"\n    {'loss_only (baseline)':30s}  rho={metrics['rho']:+.3f}  "
              f"AUROC={metrics['auroc']:.3f}  P@15%={metrics['precision_at_15pct']:.3f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def main():
    import torch

    parser = argparse.ArgumentParser(
        description='Validate ModelFeatureDetector v2')
    parser.add_argument('--pairs', type=int, default=0,
                        help='Limit to first N pairs (0 = all)')
    parser.add_argument('--aug', type=int, default=10,
                        help='Number of augmentation passes (default: 10)')
    parser.add_argument('--skip-pixel', action='store_true',
                        help='Skip v1 pixel-gap methods')
    parser.add_argument('--models', type=str, default='resnet50,vit',
                        help='Comma-separated model list')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = args.models.split(',')
    print(f"medmem version: {medmem.__version__}")
    print(f"Device: {device}")
    print(f"Augmentation passes: {args.aug}")
    print(f"Models: {models}")

    all_results = []
    pair_count = 0

    for exp_name, exp_info in BASELINES.items():
        for model_name in models:
            if args.pairs > 0 and pair_count >= args.pairs:
                break
            pair_count += 1

            pair_results = validate_pair(
                exp_name, model_name, exp_info['num_classes'],
                device, args.aug, args.skip_pixel)

            if pair_results:
                all_results.extend(pair_results)

        if args.pairs > 0 and pair_count >= args.pairs:
            break

    if not all_results:
        print("\nNo results collected!")
        return

    # Build comparison table
    df = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print("FULL COMPARISON TABLE")
    print(f"{'='*80}")

    # Summary by method
    summary = (
        df.groupby('method')
        .agg(
            mean_rho=('rho', 'mean'),
            std_rho=('rho', 'std'),
            mean_auroc=('auroc', 'mean'),
            std_auroc=('auroc', 'std'),
            mean_prec=('precision_at_15pct', 'mean'),
            n_pairs=('rho', 'count'),
        )
        .reset_index()
        .sort_values('mean_rho', ascending=False)
    )

    print("\nMethod summary (sorted by mean rho):\n")
    print(summary.to_string(index=False, float_format='%.3f'))

    # V2 vs V1 comparison
    v2_df = df[df['detector'] == 'ModelFeatureDetector']
    v1_df = df[df['detector'] == 'PixelModelGapDetector']
    loss_df = df[df['detector'] == 'loss_only']

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if len(v2_df) > 0:
        best_v2 = summary[summary['method'].str.startswith('v2:')].iloc[0]
        print(f"\nBest v2 method: {best_v2['method']}")
        print(f"  Mean rho:   {best_v2['mean_rho']:.3f} +/- {best_v2['std_rho']:.3f}")
        print(f"  Mean AUROC: {best_v2['mean_auroc']:.3f} +/- {best_v2['std_auroc']:.3f}")
        print(f"  Mean P@15%: {best_v2['mean_prec']:.3f}")

    if len(v1_df) > 0:
        best_v1 = summary[summary['method'].str.startswith('v1:')].iloc[0]
        print(f"\nBest v1 method: {best_v1['method']}")
        print(f"  Mean rho:   {best_v1['mean_rho']:.3f} +/- {best_v1['std_rho']:.3f}")
        print(f"  Mean AUROC: {best_v1['mean_auroc']:.3f} +/- {best_v1['std_auroc']:.3f}")
        print(f"  Mean P@15%: {best_v1['mean_prec']:.3f}")

    if len(loss_df) > 0:
        loss_row = summary[summary['method'] == 'baseline:loss_only']
        if len(loss_row) > 0:
            loss_row = loss_row.iloc[0]
            print(f"\nLoss-only baseline:")
            print(f"  Mean rho:   {loss_row['mean_rho']:.3f}")
            print(f"  Mean AUROC: {loss_row['mean_auroc']:.3f}")

    # Targets
    if len(v2_df) > 0:
        best_rho = v2_df.groupby('method')['rho'].mean().max()
        best_auroc = v2_df.groupby('method')['auroc'].mean().max()
        print(f"\n{'='*60}")
        print("TARGET CHECK")
        print(f"  Best v2 rho:   {best_rho:.3f}  (target > 0.5: "
              f"{'MET' if best_rho > 0.5 else 'NOT MET'})")
        print(f"  Best v2 AUROC: {best_auroc:.3f}  (target > 0.7: "
              f"{'MET' if best_auroc > 0.7 else 'NOT MET'})")

        # Beat loss-only check
        if len(loss_df) > 0:
            loss_rhos = loss_df.set_index(
                ['experiment', 'model'])['rho'].to_dict()
            for method in MODEL_FEATURE_METHODS:
                method_key = f'v2:{method}'
                method_df = v2_df[v2_df['method'] == method_key]
                if len(method_df) == 0:
                    continue
                beats = 0
                total = 0
                for _, row in method_df.iterrows():
                    key = (row['experiment'], row['model'])
                    if key in loss_rhos:
                        total += 1
                        if row['rho'] > loss_rhos[key]:
                            beats += 1
                if total > 0:
                    print(f"  {method}: beats loss-only on {beats}/{total} pairs")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / 'detection_v2_results.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'detection_v2_summary.csv', index=False)

    summary_dict = {
        'n_pairs': pair_count,
        'n_results': len(df),
        'methods_tested': sorted(df['method'].unique().tolist()),
        'method_summary': summary.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / 'detection_v2_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2, default=_json_default)

    print(f"\nResults saved to {OUTPUT_DIR}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    main()
