#!/usr/bin/env python3
"""
Validate medmem library against existing experimental results.

Part 1 (CSV-based, no GPU):
  - Load baseline memorization CSVs via VulnerabilityAudit.from_csv()
  - Run MIA + rarity analysis, compare against stored results
  - Produce reproducibility table

Part 2 (GPU-required, optional --run-distinctiveness):
  - Load trained models + data for each baseline
  - Run DistinctivenessEstimator
  - Correlate estimated risk_score with actual mean M(x) per class
  - Target: Spearman rho > 0.5 across baselines

Part 3 (CPU-only, optional --run-data-audit):
  - Load canary dataloaders for all 5 baselines
  - Run data_audit() (pixel-level risk, no model)
  - Correlate risk_score with actual mean M(x) per class
  - Compare pixel-feature JSD (new) vs model-feature cosine (old, rho=0.04)

Usage:
    python scripts/validate_medmem.py                     # Part 1 only
    python scripts/validate_medmem.py --run-distinctiveness  # Part 1 + 2
    python scripts/validate_medmem.py --run-data-audit       # Part 1 + 3
"""

import argparse
import json
import sys
from pathlib import Path
from scipy import stats

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import medmem

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'

BASELINES = {
    'ham1000_baseline':       {'num_classes': 7,  'dataset_config': 'ham10000'},
    'chestxray_baseline':     {'num_classes': 14, 'dataset_config': 'chestxray'},
    'odir5k_baseline':        {'num_classes': 8,  'dataset_config': 'odir5k'},
    'retinal_oct_baseline':   {'num_classes': 4,  'dataset_config': 'retinal_oct'},
    'kvasir_baseline':        {'num_classes': 14, 'dataset_config': 'kvasir_capsule'},
}

MODELS = ['resnet50', 'vit', 'mae', 'medsam']


# -----------------------------------------------------------------------
# Part 1: CSV-based validation
# -----------------------------------------------------------------------

def validate_csv_mode():
    """Load existing CSVs, run medmem analysis, verify consistency."""
    print("=" * 70)
    print("Part 1: CSV-Based Validation (medmem full-mode from_csv)")
    print("=" * 70)

    all_rows = []

    for exp_name, exp_info in BASELINES.items():
        exp_dir = RESULTS_ROOT / exp_name / 'memorization'
        if not exp_dir.exists():
            print(f"  SKIP {exp_name} — no memorization dir")
            continue

        for model_name in MODELS:
            mem_csv = exp_dir / f'{model_name}_memorization_scores.csv'
            test_csv = exp_dir / f'{model_name}_test_scores.csv'

            if not mem_csv.exists():
                continue

            has_test = test_csv.exists()
            try:
                audit = medmem.VulnerabilityAudit.from_csv(
                    str(mem_csv),
                    str(test_csv) if has_test else None,
                )
            except Exception as e:
                print(f"  ERROR {exp_name}/{model_name}: {e}")
                continue

            summary = audit.summary()
            n = summary['n_samples']
            mean_m = summary['memorization']['mean']
            rarity_rho = summary.get('rarity', {}).get('spearman_rho', None)
            rarity_dir = summary.get('rarity', {}).get('direction', '')
            mia_auc = summary.get('mia', {}).get('best_auc', None)

            row = {
                'experiment': exp_name,
                'model': model_name,
                'n_samples': n,
                'mean_M(x)': round(mean_m, 4),
                'rarity_rho': round(rarity_rho, 3) if rarity_rho else None,
                'rarity_dir': rarity_dir,
                'mia_auc': round(mia_auc, 3) if mia_auc else None,
            }
            all_rows.append(row)

    if not all_rows:
        print("No baseline results found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    print(f"\nValidated {len(df)} model-dataset pairs:\n")
    print(df.to_string(index=False))

    # Summary stats
    valid_mia = df['mia_auc'].dropna()
    valid_rho = df['rarity_rho'].dropna()
    print(f"\nMIA AUC: mean={valid_mia.mean():.3f}, "
          f"range=[{valid_mia.min():.3f}, {valid_mia.max():.3f}]")
    print(f"Rarity rho: mean={valid_rho.mean():.3f}, "
          f"range=[{valid_rho.min():.3f}, {valid_rho.max():.3f}]")

    return df


# -----------------------------------------------------------------------
# Part 2: Distinctiveness validation
# -----------------------------------------------------------------------

def validate_distinctiveness():
    """Load models, run DistinctivenessEstimator, correlate with actual M(x)."""
    print("\n" + "=" * 70)
    print("Part 2: Distinctiveness Estimator Validation")
    print("=" * 70)

    import torch
    import yaml
    from src.models.base_model import ModelFactory
    from src.data.transforms import TransformFactory

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Collect (estimated_risk, actual_mean_Mx) per class across all pairs
    all_risk_scores = []
    all_actual_mx = []
    pair_results = []

    for exp_name, exp_info in BASELINES.items():
        exp_dir = RESULTS_ROOT / exp_name
        mem_dir = exp_dir / 'memorization'
        if not mem_dir.exists():
            continue

        # Load dataset config for num_classes
        num_classes = exp_info['num_classes']
        dataset_name = exp_info['dataset_config']

        for model_name in MODELS:
            ckpt_path = (exp_dir / model_name / 'candidate' / 'checkpoints' /
                         f'{model_name}_candidate_best.pth')
            mem_csv = mem_dir / f'{model_name}_memorization_scores.csv'

            if not ckpt_path.exists() or not mem_csv.exists():
                continue

            print(f"\n  {exp_name} / {model_name}")

            # Load actual memorization scores
            mem_df = pd.read_csv(mem_csv)
            actual_per_class = (
                mem_df.groupby('class_name')['memorization_score']
                .mean()
                .reset_index()
            )

            # Load model
            try:
                model_config_path = Path(f'config/models/{model_name}.yaml')
                if model_config_path.exists():
                    with open(model_config_path) as f:
                        model_config = yaml.safe_load(f)
                else:
                    model_config = {}

                model = ModelFactory.create(model_name, model_config, num_classes)
                model.load_checkpoint(str(ckpt_path))
                model.to(device).eval()
            except Exception as e:
                print(f"    SKIP model load failed: {e}")
                continue

            # Build a simple dataloader from the memorization CSV images
            # We use the canary split data that was used for memorization scoring
            try:
                dataloader = _build_dataloader(exp_name, dataset_name, model_name)
            except Exception as e:
                print(f"    SKIP dataloader failed: {e}")
                continue

            # Run distinctiveness estimator
            try:
                estimator = medmem.DistinctivenessEstimator(model, device=device)
                class_risk = estimator.estimate(dataloader)
            except Exception as e:
                print(f"    SKIP estimator failed: {e}")
                continue

            # Merge with actual M(x)
            merged = class_risk.merge(
                actual_per_class,
                on='class_name',
                how='inner',
            )

            if len(merged) < 3:
                print(f"    SKIP only {len(merged)} classes matched")
                continue

            # Spearman correlation: risk_score vs actual mean M(x)
            rho, p = stats.spearmanr(merged['risk_score'], merged['memorization_score'])
            print(f"    Classes: {len(merged)}")
            print(f"    Spearman(risk_score, mean_M(x)): rho={rho:.3f}, p={p:.4f}")

            all_risk_scores.extend(merged['risk_score'].tolist())
            all_actual_mx.extend(merged['memorization_score'].tolist())
            pair_results.append({
                'experiment': exp_name,
                'model': model_name,
                'n_classes': len(merged),
                'rho': round(rho, 3),
                'p_value': round(p, 4),
                'significant': p < 0.05,
            })

            # Print class-level detail
            detail = merged[['class_name', 'frequency', 'risk_score', 'memorization_score']]
            detail = detail.sort_values('risk_score', ascending=False)
            print(detail.to_string(index=False))

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

    if not pair_results:
        print("\nNo model-dataset pairs could be validated!")
        return

    # Aggregate results
    results_df = pd.DataFrame(pair_results)
    print(f"\n{'='*70}")
    print("Distinctiveness Validation Summary")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))

    # Overall correlation across all classes from all pairs
    if len(all_risk_scores) > 3:
        overall_rho, overall_p = stats.spearmanr(all_risk_scores, all_actual_mx)
        print(f"\nOverall (pooled across all {len(all_risk_scores)} classes):")
        print(f"  Spearman rho = {overall_rho:.3f}, p = {overall_p:.4g}")
        target_met = overall_rho > 0.5
        print(f"  Target (rho > 0.5): {'MET' if target_met else 'NOT MET'}")

    sig_pairs = results_df['significant'].sum()
    print(f"\nSignificant pairs: {sig_pairs}/{len(results_df)}")
    mean_rho = results_df['rho'].mean()
    print(f"Mean per-pair rho: {mean_rho:.3f}")


def _build_dataloader(exp_name, dataset_name, model_name):
    """Build a DataLoader for canary split using the project's existing factory functions."""
    import yaml

    # Build config dict matching what the create_*_dataloader functions expect
    exp_config_path = Path(f'config/experiments/{exp_name}.yaml')
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)

    dataset_config_path = Path(f'config/datasets/{dataset_name}.yaml')
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)

    config = exp_config.copy()
    config['dataset_config'] = dataset_config

    # Use each dataset's own create_*_dataloader (handles paths, transforms, etc.)
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
        config,
        split='canary',
        shuffle=False,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )


# -----------------------------------------------------------------------
# Part 3: Data-only audit validation
# -----------------------------------------------------------------------

def validate_data_audit():
    """Run data_audit() on all baselines, correlate with actual M(x)."""
    print("\n" + "=" * 70)
    print("Part 3: Data-Only Audit Validation (pixel features, no model)")
    print("=" * 70)

    import time

    all_risk_scores = []
    all_actual_mx = []
    pair_results = []

    for exp_name, exp_info in BASELINES.items():
        exp_dir = RESULTS_ROOT / exp_name
        mem_dir = exp_dir / 'memorization'
        if not mem_dir.exists():
            print(f"  SKIP {exp_name} — no memorization dir")
            continue

        dataset_name = exp_info['dataset_config']

        # Build dataloader once per dataset
        try:
            dataloader = _build_dataloader(exp_name, dataset_name, 'resnet50')
        except Exception as e:
            print(f"  SKIP {exp_name} — dataloader failed: {e}")
            continue

        # Run data audit
        print(f"\n  {exp_name}: running data_audit()...")
        t0 = time.time()
        result = medmem.data_audit(dataloader)
        elapsed = time.time() - t0
        print(f"    Extracted {len(result.sample_risk_df)} samples in {elapsed:.1f}s")

        # Print class risk table
        print(f"    Class risk ({len(result.class_risk_df)} classes):")
        print(result.class_risk_df[
            ['class_name', 'n_samples', 'frequency', 'rarity_tier',
             'color_distinctiveness', 'texture_distinctiveness',
             'overall_distinctiveness', 'risk_score']
        ].to_string(index=False))

        # Sample risk distribution
        risk_dist = result.sample_risk_df['risk_category'].value_counts()
        print(f"    Sample risk distribution: {dict(risk_dist)}")

        # Correlate with actual M(x) for each model
        for model_name in MODELS:
            mem_csv = mem_dir / f'{model_name}_memorization_scores.csv'
            if not mem_csv.exists():
                continue

            mem_df = pd.read_csv(mem_csv)
            actual_per_class = (
                mem_df.groupby('class_name')['memorization_score']
                .mean()
                .reset_index()
            )

            merged = result.class_risk_df.merge(
                actual_per_class, on='class_name', how='inner',
            )

            if len(merged) < 3:
                continue

            rho, p = stats.spearmanr(merged['risk_score'], merged['memorization_score'])
            print(f"\n    {model_name}: Spearman(risk_score, mean_M(x)) "
                  f"rho={rho:.3f}, p={p:.4f}")

            all_risk_scores.extend(merged['risk_score'].tolist())
            all_actual_mx.extend(merged['memorization_score'].tolist())
            pair_results.append({
                'experiment': exp_name,
                'model': model_name,
                'n_classes': len(merged),
                'rho': round(rho, 3),
                'p_value': round(p, 4),
                'significant': p < 0.05,
            })

    if not pair_results:
        print("\nNo model-dataset pairs could be validated!")
        return

    # Aggregate results
    results_df = pd.DataFrame(pair_results)
    print(f"\n{'='*70}")
    print("Data-Only Audit Validation Summary")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))

    # Overall pooled correlation
    if len(all_risk_scores) > 3:
        overall_rho, overall_p = stats.spearmanr(all_risk_scores, all_actual_mx)
        print(f"\nOverall (pooled across all {len(all_risk_scores)} classes):")
        print(f"  Spearman rho = {overall_rho:.3f}, p = {overall_p:.4g}")
        target_met = overall_rho > 0.3
        print(f"  Target (rho > 0.3): {'MET' if target_met else 'NOT MET'}")
        print(f"  (Old model-feature approach was rho ~ 0.04)")

    sig_pairs = results_df['significant'].sum()
    print(f"\nSignificant pairs: {sig_pairs}/{len(results_df)}")
    mean_rho = results_df['rho'].mean()
    print(f"Mean per-pair rho: {mean_rho:.3f}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Validate medmem library')
    parser.add_argument('--run-distinctiveness', action='store_true',
                        help='Run Part 2: distinctiveness estimator validation (requires GPU)')
    parser.add_argument('--run-data-audit', action='store_true',
                        help='Run Part 3: data-only audit validation (CPU only)')
    args = parser.parse_args()

    print(f"medmem version: {medmem.__version__}")
    print()

    csv_df = validate_csv_mode()

    if args.run_distinctiveness:
        validate_distinctiveness()
    else:
        print("\nSkipping Part 2 (distinctiveness). "
              "Use --run-distinctiveness to enable.")

    if args.run_data_audit:
        validate_data_audit()
    else:
        print("\nSkipping Part 3 (data audit). "
              "Use --run-data-audit to enable.")


if __name__ == '__main__':
    main()
