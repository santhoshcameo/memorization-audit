#!/usr/bin/env python3
"""
Validate MiniAuditor against ground-truth M(x).

Loads trained candidate checkpoints, builds canary/train/test dataloaders,
runs MiniAuditor.audit(), and compares memorization_score against gold-standard
scores from results/*/memorization/*.csv.

Metrics: Spearman rho, AUROC at M(x)>0.3, Precision@15%, class-level rho, MIA AUC.

Quick test (default): 2 pairs (ham1000_baseline x resnet50, vit)
Full test (--full):   15 pairs (5 baselines x resnet50, vit, medsam)

Usage:
    python scripts/validate_mini_audit.py              # quick (2 pairs)
    python scripts/validate_mini_audit.py --full       # all 15 pairs
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

import medmem
from medmem.mini_audit import MiniAuditor
from src.models.base_model import ModelFactory


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'
OUTPUT_DIR = RESULTS_ROOT / 'mini_audit_validation'

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

# MAE has no pretrained weights — skip it
FULL_MODELS = ['resnet50', 'vit', 'medsam']


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _get_dataloader_factory(dataset_name):
    """Return the create_*_dataloader function for a dataset."""
    if dataset_name == 'ham10000':
        from src.data.ham10000 import create_ham10000_dataloader as fn
    elif dataset_name == 'chestxray':
        from src.data.chestxray import create_chestxray_dataloader as fn
    elif dataset_name == 'odir5k':
        from src.data.odir5k import create_odir5k_dataloader as fn
    elif dataset_name == 'retinal_oct':
        from src.data.retinal_oct import create_retinal_oct_dataloader as fn
    elif dataset_name == 'kvasir_capsule':
        from src.data.kvasir_capsule import create_kvasir_capsule_dataloader as fn
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return fn


def _build_config(exp_name, dataset_name):
    """Build config dict with dataset_config key."""
    exp_config_path = Path(f'config/experiments/{exp_name}.yaml')
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)

    dataset_config_path = Path(f'config/datasets/{dataset_name}.yaml')
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)

    config = exp_config.copy()
    config['dataset_config'] = dataset_config
    return config


def _build_dataloaders(exp_name, dataset_name):
    """Build canary, train, and test DataLoaders."""
    config = _build_config(exp_name, dataset_name)
    create_dl = _get_dataloader_factory(dataset_name)

    dl_kwargs = dict(shuffle=False, batch_size=64, num_workers=4, pin_memory=True)

    canary_loader = create_dl(config, split='canary', **dl_kwargs)
    train_loader = create_dl(config, split='train', **dl_kwargs)
    test_loader = create_dl(config, split='test', **dl_kwargs)

    return canary_loader, train_loader, test_loader


def _load_model(exp_name, model_name, num_classes, device):
    """Load a trained candidate model checkpoint."""
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


def _load_ground_truth_csv(exp_name, model_name):
    """Load ground-truth memorization scores from canary CSV (augmentation-averaged)."""
    mem_csv = (RESULTS_ROOT / exp_name / 'memorization' /
               f'{model_name}_memorization_scores.csv')
    if not mem_csv.exists():
        return None
    df = pd.read_csv(mem_csv)
    if df['image_id'].duplicated().any():
        df = df.groupby('image_id', as_index=False).agg({
            'class_name': 'first',
            'memorization_score': 'mean',
        })
    return df


def _compute_fresh_gold_standard(exp_name, model_name, num_classes,
                                  canary_loader, device):
    """Compute gold-standard M(x) using both checkpoints with medmem's scorer.

    This gives an apples-to-apples comparison: same scorer, same transforms,
    same eval mode — the only difference is how the independent was created.
    """
    from medmem.scorer import MemorizationScorer as MedMemScorer

    ind_ckpt = (RESULTS_ROOT / exp_name / model_name / 'independent' /
                'checkpoints' / f'{model_name}_independent_best.pth')
    if not ind_ckpt.exists():
        return None

    ind_model = _load_model_from_ckpt(str(ind_ckpt), model_name, num_classes, device)
    if ind_model is None:
        return None

    cand_ckpt = (RESULTS_ROOT / exp_name / model_name / 'candidate' /
                 'checkpoints' / f'{model_name}_candidate_best.pth')
    cand_model = _load_model_from_ckpt(str(cand_ckpt), model_name, num_classes, device)

    print("  [Gold Standard] Scoring with both checkpoints (same scorer)...")
    scorer = MedMemScorer(cand_model, ind_model, device)
    gt_df = scorer.score(canary_loader)

    del ind_model, cand_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return gt_df


def _load_model_from_ckpt(ckpt_path, model_name, num_classes, device):
    """Load a model from checkpoint path."""
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        return None
    model_config_path = Path(f'config/models/{model_name}.yaml')
    if model_config_path.exists():
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
    else:
        model_config = {}
    model = ModelFactory.create(model_name, model_config, num_classes)
    model.load_checkpoint(str(ckpt))
    return model.to(device).eval()


def _load_ground_truth_mia(exp_name, model_name):
    """Load ground-truth MIA results."""
    mia_json = (RESULTS_ROOT / exp_name / model_name / 'mia_validation' /
                'mia_validation_results.json')
    if not mia_json.exists():
        return None
    with open(mia_json) as f:
        return json.load(f)


def precision_at_k(y_true, y_scores, k_pct=0.15):
    """Precision at top k%."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / k


def evaluate_mini_audit(audit_result, gt_df, gt_mia=None):
    """Compare mini audit result against ground-truth M(x)."""
    mini_df = audit_result.memorization_df.copy()

    # Merge on image_id
    merged = mini_df[['image_id', 'class_name', 'memorization_score']].merge(
        gt_df[['image_id', 'memorization_score']],
        on='image_id', how='inner',
        suffixes=('_mini', '_gt'),
    )

    if len(merged) < 10:
        return None

    mini_scores = merged['memorization_score_mini'].values
    gt_scores = merged['memorization_score_gt'].values

    # Spearman correlation (sample-level)
    rho, p_rho = stats.spearmanr(mini_scores, gt_scores)

    # AUROC: predict M(x)_gt > 0.3
    high_mem = (gt_scores > 0.3).astype(int)
    if 0 < high_mem.sum() < len(high_mem):
        auroc = roc_auc_score(high_mem, mini_scores)
    else:
        auroc = None

    # Precision@15%
    prec15 = precision_at_k(high_mem, mini_scores, k_pct=0.15)

    # Per-class correlation
    per_class_mini = merged.groupby('class_name')['memorization_score_mini'].mean()
    per_class_gt = merged.groupby('class_name')['memorization_score_gt'].mean()
    common = per_class_mini.index.intersection(per_class_gt.index)
    if len(common) >= 3:
        class_rho, class_p = stats.spearmanr(
            per_class_mini[common], per_class_gt[common])
    else:
        class_rho, class_p = None, None

    # MIA AUC from mini audit
    mini_mia_auc = None
    if audit_result.mia is not None:
        try:
            mia_results = audit_result.mia.run_all_attacks()
            best = max(mia_results.values(), key=lambda r: r.auc)
            mini_mia_auc = best.auc
        except Exception:
            pass

    # Ground-truth MIA AUC for comparison
    gt_mia_auc = None
    if gt_mia is not None:
        try:
            gt_mia_auc = max(
                v.get('auc', 0) if isinstance(v, dict) else 0
                for v in gt_mia.values()
            )
        except Exception:
            pass

    return {
        'n_samples': len(merged),
        'n_high_mem': int(high_mem.sum()),
        'spearman_rho': round(rho, 4),
        'spearman_p': float(p_rho),
        'auroc': round(auroc, 4) if auroc is not None else None,
        'precision_at_15pct': round(prec15, 4),
        'class_rho': round(class_rho, 4) if class_rho is not None else None,
        'class_p': float(class_p) if class_p is not None else None,
        'mini_mia_auc': round(mini_mia_auc, 4) if mini_mia_auc is not None else None,
        'gt_mia_auc': round(gt_mia_auc, 4) if gt_mia_auc is not None else None,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_pair(exp_name, model_name, device):
    """Run mini audit validation for one (experiment, model) pair."""
    exp_info = BASELINES[exp_name]
    num_classes = exp_info['num_classes']
    dataset_name = exp_info['dataset_config']

    print(f"\n{'='*60}")
    print(f"  {exp_name} / {model_name}")
    print(f"{'='*60}")

    gt_mia = _load_ground_truth_mia(exp_name, model_name)

    # Load candidate model
    try:
        model = _load_model(exp_name, model_name, num_classes, device)
    except Exception as e:
        print(f"  SKIP: model load failed: {e}")
        return None
    if model is None:
        print("  SKIP: checkpoint not found")
        return None

    # Build dataloaders
    try:
        canary_loader, train_loader, test_loader = _build_dataloaders(
            exp_name, dataset_name)
    except Exception as e:
        print(f"  SKIP: dataloader failed: {e}")
        return None

    # Compute fresh gold standard (same scorer, same transforms)
    gt_df = _compute_fresh_gold_standard(
        exp_name, model_name, num_classes, canary_loader, device)
    if gt_df is None:
        # Fall back to CSV-based ground truth
        gt_df = _load_ground_truth_csv(exp_name, model_name)
    if gt_df is None:
        print("  SKIP: no ground truth available")
        return None

    # Run mini audit
    t0 = time.time()
    try:
        auditor = MiniAuditor(head_epochs=10, head_lr=1e-3, run_mia=True)
        audit_result = auditor.audit(
            model, canary_loader, train_loader,
            test_loader=test_loader, device=device,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - t0

    # Evaluate against ground truth
    metrics = evaluate_mini_audit(audit_result, gt_df, gt_mia)
    if metrics is None:
        print("  SKIP: too few matched samples")
        return None

    metrics['experiment'] = exp_name
    metrics['model'] = model_name
    metrics['time_seconds'] = round(elapsed, 1)

    print(f"  Spearman rho: {metrics['spearman_rho']:.4f} "
          f"(p={metrics['spearman_p']:.2e})")
    print(f"  AUROC (M>0.3): {metrics['auroc']}")
    print(f"  Precision@15%: {metrics['precision_at_15pct']:.4f}")
    print(f"  Class-level rho: {metrics['class_rho']}")
    print(f"  Mini MIA AUC: {metrics['mini_mia_auc']}")
    print(f"  GT MIA AUC:   {metrics['gt_mia_auc']}")
    print(f"  Time: {elapsed:.1f}s")

    # Save per-pair results
    pair_dir = OUTPUT_DIR / exp_name / model_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    audit_result.save_report(str(pair_dir))

    # Free GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Validate MiniAuditor against ground-truth M(x)')
    parser.add_argument('--full', action='store_true',
                        help='Run all 15 pairs (5 baselines x resnet50,vit,medsam)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"medmem version: {medmem.__version__}")

    if args.full:
        pairs = [(exp, m) for exp in BASELINES for m in FULL_MODELS]
    else:
        pairs = QUICK_PAIRS

    print(f"\nPairs to validate: {len(pairs)}")

    all_results = []
    for exp_name, model_name in pairs:
        result = run_pair(exp_name, model_name, device)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nNo results produced!")
        return

    # Summary table
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("MINI AUDIT VALIDATION SUMMARY")
    print(f"{'='*70}")
    cols = ['experiment', 'model', 'spearman_rho', 'auroc',
            'precision_at_15pct', 'class_rho', 'mini_mia_auc',
            'gt_mia_auc', 'time_seconds']
    print(df[cols].to_string(index=False))

    # Averages
    print(f"\nAverages across {len(df)} pairs:")
    print(f"  Spearman rho:  {df['spearman_rho'].mean():.4f} "
          f"(range [{df['spearman_rho'].min():.4f}, "
          f"{df['spearman_rho'].max():.4f}])")
    valid_auroc = df['auroc'].dropna()
    if len(valid_auroc) > 0:
        print(f"  AUROC:         {valid_auroc.mean():.4f} "
              f"(range [{valid_auroc.min():.4f}, "
              f"{valid_auroc.max():.4f}])")
    valid_class_rho = df['class_rho'].dropna()
    if len(valid_class_rho) > 0:
        print(f"  Class rho:     {valid_class_rho.mean():.4f} "
              f"(range [{valid_class_rho.min():.4f}, "
              f"{valid_class_rho.max():.4f}])")
    print(f"  Precision@15%: {df['precision_at_15pct'].mean():.4f}")
    print(f"  Time:          {df['time_seconds'].mean():.1f}s mean")

    # Targets check
    print(f"\nTargets (rho>0.2, AUROC>0.6, class_rho>0.3):")
    rho_ok = df['spearman_rho'].mean() > 0.2
    auroc_ok = valid_auroc.mean() > 0.6 if len(valid_auroc) > 0 else False
    class_ok = valid_class_rho.mean() > 0.3 if len(valid_class_rho) > 0 else False
    print(f"  rho > 0.2:       {'MET' if rho_ok else 'NOT MET'} "
          f"({df['spearman_rho'].mean():.4f})")
    if len(valid_auroc) > 0:
        print(f"  AUROC > 0.6:     {'MET' if auroc_ok else 'NOT MET'} "
              f"({valid_auroc.mean():.4f})")
    if len(valid_class_rho) > 0:
        print(f"  class rho > 0.3: {'MET' if class_ok else 'NOT MET'} "
              f"({valid_class_rho.mean():.4f})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / 'validation_results.csv', index=False)
    with open(OUTPUT_DIR / 'validation_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
