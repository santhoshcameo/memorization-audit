#!/usr/bin/env python3
"""
Validate ReferenceModelDetector against ground-truth M(x).

Loads trained checkpoints and canary dataloaders, runs reference detection
with all three methods, and compares against gold-standard memorization
scores (Spearman rho, AUROC at M(x)>0.3, precision@15%).

Quick test (default): 2 pairs (ham1000_baseline x resnet50,vit)
Full test (--full):   10 pairs (5 baselines x resnet50,vit)

Usage:
    python scripts/validate_reference_detector.py              # quick (2 pairs)
    python scripts/validate_reference_detector.py --full       # all 10 pairs
    python scripts/validate_reference_detector.py --method head_retrain  # single method
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

import medmem
from medmem.reference_detector import ReferenceModelDetector
from src.models.base_model import ModelFactory


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'
OUTPUT_DIR = RESULTS_ROOT / 'reference_detection_validation'

BASELINES = {
    'ham1000_baseline':     {'num_classes': 7,  'dataset_config': 'ham10000'},
    'chestxray_baseline':   {'num_classes': 14, 'dataset_config': 'chestxray'},
    'odir5k_baseline':      {'num_classes': 8,  'dataset_config': 'odir5k'},
    'retinal_oct_baseline': {'num_classes': 4,  'dataset_config': 'retinal_oct'},
    'kvasir_baseline':      {'num_classes': 14, 'dataset_config': 'kvasir_capsule'},
}

QUICK_PAIRS = [
    ('ham1000_baseline', 'resnet50'),
    ('ham1000_baseline', 'vit'),
]

FULL_MODELS = ['resnet50', 'vit']


def _build_dataloader(exp_name, dataset_name):
    """Build canary DataLoader using project factory functions."""
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
    """Load a trained model checkpoint."""
    exp_dir = RESULTS_ROOT / exp_name
    ckpt_path = (exp_dir / model_name / 'candidate' / 'checkpoints' /
                 f'{model_name}_candidate_best.pth')
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
    """Load ground-truth memorization scores."""
    mem_csv = (RESULTS_ROOT / exp_name / 'memorization' /
               f'{model_name}_memorization_scores.csv')
    if not mem_csv.exists():
        return None
    df = pd.read_csv(mem_csv)
    # Handle duplicates (ODIR-5K)
    if df['image_id'].duplicated().any():
        df = df.groupby('image_id', as_index=False).agg({
            'class_name': 'first',
            'memorization_score': 'mean',
        })
    return df


def precision_at_k(y_true, y_scores, k_pct=0.15):
    """Precision at top k% — fraction of top-k scored samples that are truly memorized."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / k


def evaluate_method(result, gt_df, method_name):
    """Compare reference detection result against ground-truth M(x)."""
    sample_df = result.sample_df.copy()

    # Merge on image_id
    merged = sample_df[['image_id', 'gap_score']].merge(
        gt_df[['image_id', 'memorization_score']],
        on='image_id', how='inner',
    )

    if len(merged) < 10:
        return None

    gap = merged['gap_score'].values
    gt = merged['memorization_score'].values

    # Spearman correlation
    rho, p_rho = stats.spearmanr(gap, gt)

    # AUROC: predict M(x) > 0.3
    high_mem = (gt > 0.3).astype(int)
    if high_mem.sum() > 0 and high_mem.sum() < len(high_mem):
        auroc = roc_auc_score(high_mem, gap)
    else:
        auroc = None

    # Precision@15%
    prec15 = precision_at_k(high_mem, gap, k_pct=0.15)

    # Per-class correlation
    class_merged = merged.copy()
    class_merged['class_name'] = sample_df.set_index('image_id').loc[
        merged['image_id'], 'class_name'].values
    per_class_gap = class_merged.groupby('class_name')['gap_score'].mean()
    per_class_gt = class_merged.groupby('class_name')['memorization_score'].mean()
    common = per_class_gap.index.intersection(per_class_gt.index)
    if len(common) >= 3:
        class_rho, class_p = stats.spearmanr(
            per_class_gap[common], per_class_gt[common])
    else:
        class_rho, class_p = None, None

    return {
        'method': method_name,
        'n_samples': len(merged),
        'n_high_mem': int(high_mem.sum()),
        'spearman_rho': round(rho, 4),
        'spearman_p': float(p_rho),
        'auroc': round(auroc, 4) if auroc is not None else None,
        'precision_at_15pct': round(prec15, 4),
        'class_rho': round(class_rho, 4) if class_rho is not None else None,
        'class_p': float(class_p) if class_p is not None else None,
    }


def run_pair(exp_name, model_name, methods, device):
    """Run reference detection for one (experiment, model) pair."""
    exp_info = BASELINES[exp_name]
    num_classes = exp_info['num_classes']
    dataset_name = exp_info['dataset_config']

    print(f"\n{'='*60}")
    print(f"  {exp_name} / {model_name}")
    print(f"{'='*60}")

    # Load ground truth
    gt_df = _load_ground_truth(exp_name, model_name)
    if gt_df is None:
        print("  SKIP: no ground-truth memorization CSV")
        return []

    # Load model
    try:
        model = _load_model(exp_name, model_name, num_classes, device)
    except Exception as e:
        print(f"  SKIP: model load failed: {e}")
        return []
    if model is None:
        print("  SKIP: checkpoint not found")
        return []

    # Build dataloader
    try:
        dataloader = _build_dataloader(exp_name, dataset_name)
    except Exception as e:
        print(f"  SKIP: dataloader failed: {e}")
        return []

    pair_results = []

    for method in methods:
        print(f"\n  --- Method: {method} ---")
        t0 = time.time()

        try:
            detector = ReferenceModelDetector(method=method, n_folds=5)
            result = detector.detect(model, dataloader, device=device)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        elapsed = time.time() - t0

        metrics = evaluate_method(result, gt_df, method)
        if metrics is None:
            print("  SKIP: too few matched samples")
            continue

        metrics['experiment'] = exp_name
        metrics['model'] = model_name
        metrics['time_seconds'] = round(elapsed, 1)
        pair_results.append(metrics)

        print(f"  Spearman rho: {metrics['spearman_rho']:.4f} "
              f"(p={metrics['spearman_p']:.2e})")
        print(f"  AUROC (M>0.3): {metrics['auroc']}")
        print(f"  Precision@15%: {metrics['precision_at_15pct']:.4f}")
        print(f"  Class-level rho: {metrics['class_rho']}")
        print(f"  Time: {elapsed:.1f}s")

        # Save per-pair result
        pair_dir = OUTPUT_DIR / exp_name / model_name / method
        pair_dir.mkdir(parents=True, exist_ok=True)
        result.save(str(pair_dir))

    # Free GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return pair_results


def main():
    parser = argparse.ArgumentParser(
        description='Validate ReferenceModelDetector against ground-truth M(x)')
    parser.add_argument('--full', action='store_true',
                        help='Run on all 10 pairs (5 baselines x resnet50,vit)')
    parser.add_argument('--method', type=str, default=None,
                        choices=['head_retrain', 'full_finetune', 'ensemble'],
                        help='Run only one method (default: all three)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"medmem version: {medmem.__version__}")

    methods = [args.method] if args.method else ['head_retrain', 'full_finetune', 'ensemble']

    if args.full:
        pairs = [(exp, m) for exp in BASELINES for m in FULL_MODELS]
    else:
        pairs = QUICK_PAIRS

    print(f"\nPairs to validate: {len(pairs)}")
    print(f"Methods: {methods}")

    all_results = []
    for exp_name, model_name in pairs:
        results = run_pair(exp_name, model_name, methods, device)
        all_results.extend(results)

    if not all_results:
        print("\nNo results produced!")
        return

    # Summary table
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    cols = ['experiment', 'model', 'method', 'spearman_rho', 'auroc',
            'precision_at_15pct', 'class_rho', 'time_seconds']
    print(df[cols].to_string(index=False))

    # Per-method averages
    print(f"\nPer-method averages:")
    for method in df['method'].unique():
        mdf = df[df['method'] == method]
        print(f"  {method}:")
        print(f"    Spearman rho: {mdf['spearman_rho'].mean():.4f} "
              f"(range [{mdf['spearman_rho'].min():.4f}, "
              f"{mdf['spearman_rho'].max():.4f}])")
        valid_auroc = mdf['auroc'].dropna()
        if len(valid_auroc) > 0:
            print(f"    AUROC:        {valid_auroc.mean():.4f} "
                  f"(range [{valid_auroc.min():.4f}, "
                  f"{valid_auroc.max():.4f}])")
        print(f"    Precision@15%: {mdf['precision_at_15pct'].mean():.4f}")
        print(f"    Time:         {mdf['time_seconds'].mean():.1f}s mean")

    # Targets check
    print(f"\nTargets:")
    for method in df['method'].unique():
        mdf = df[df['method'] == method]
        rho_target = mdf['spearman_rho'].mean() > 0.5
        valid_auroc = mdf['auroc'].dropna()
        auroc_target = valid_auroc.mean() > 0.7 if len(valid_auroc) > 0 else False
        print(f"  {method}: rho>{0.5} {'MET' if rho_target else 'NOT MET'}, "
              f"AUROC>{0.7} {'MET' if auroc_target else 'NOT MET'}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / 'validation_results.csv', index=False)
    with open(OUTPUT_DIR / 'validation_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
