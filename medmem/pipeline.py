"""
Full audit pipeline: data in → risk report out.

Usage:
    import medmem
    report = medmem.audit("/path/to/images", name="my_dataset")
    report = medmem.audit("/path/to/images", csv="labels.csv", name="my_data", quick=True)
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .data import (ingest_folder, ingest_csv, stratified_split,
                   ImageClassificationDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM)
from .engine import (create_model, train_model, compute_memorization,
                     run_mia, compute_statistics, SUPPORTED_MODELS)


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    canary_df: pd.DataFrame
    test_df: pd.DataFrame
    mia: Dict
    statistics: Dict
    time_seconds: float


@dataclass
class AuditReport:
    """Complete audit report across all models."""
    dataset_name: str
    classes: List[str]
    n_samples: int
    models: Dict[str, ModelResult] = field(default_factory=dict)
    output_dir: Optional[Path] = None

    def summary(self):
        """Print summary to console."""
        print(self._build_text())

    def save(self, path=None):
        """Save report to output directory."""
        out = Path(path) if path else self.output_dir
        if out is None:
            raise ValueError("No output path specified")
        out.mkdir(parents=True, exist_ok=True)

        with open(out / 'audit_report.txt', 'w') as f:
            f.write(self._build_text())

        for name, res in self.models.items():
            model_dir = out / name
            model_dir.mkdir(parents=True, exist_ok=True)
            res.canary_df.to_csv(model_dir / 'memorization_scores.csv', index=False)
            res.test_df.to_csv(model_dir / 'test_scores.csv', index=False)
            with open(model_dir / 'mia_results.json', 'w') as f:
                json.dump(res.mia, f, indent=2, default=str)
            with open(model_dir / 'statistics.json', 'w') as f:
                json.dump(res.statistics, f, indent=2, default=str)

        print(f"[medmem] Report saved to {out}")

    def most_vulnerable_classes(self, top_k=5):
        """Return classes with highest average memorization across models."""
        all_classes = {}
        for res in self.models.values():
            for _, row in res.statistics['per_class'].iterrows():
                cls = row['class_name']
                if cls not in all_classes:
                    all_classes[cls] = {'freq': row['frequency'], 'scores': []}
                all_classes[cls]['scores'].append(row['mean_mx'])

        ranked = sorted(all_classes.items(),
                        key=lambda x: np.mean(x[1]['scores']), reverse=True)
        return [(cls, info['freq'], np.mean(info['scores']))
                for cls, info in ranked[:top_k]]

    def _build_text(self):
        lines = []
        lines.append("=" * 70)
        lines.append(f"  MEMORIZATION AUDIT REPORT: {self.dataset_name}")
        lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Samples: {self.n_samples} | Classes: {len(self.classes)}")
        lines.append(f"  Models: {', '.join(self.models.keys())}")
        lines.append("=" * 70)

        for name, res in self.models.items():
            ov = res.statistics['overall']
            mia = res.mia
            sp = res.statistics['spearman']
            mw = res.statistics.get('mann_whitney', {})
            pc = res.statistics['per_class']

            lines.append(f"\n{'='*70}")
            lines.append(f"  MODEL: {name.upper()}")
            lines.append(f"{'='*70}")

            lines.append(f"\n  Memorization:")
            lines.append(f"    Mean M(x):     {ov['mean_mx']:+.4f}")
            lines.append(f"    High-risk:     {ov['high_risk_pct']:.1f}% (M(x) > 0.3)")

            lines.append(f"\n  Membership Inference Attack:")
            lines.append(f"    Best AUC:      {mia['best_auc']:.3f} ({mia['best_method']})")
            vuln = "VULNERABLE" if mia['best_auc'] > 0.6 else "Low risk"
            lines.append(f"    Status:        {vuln}")

            if mia['per_class_auc']:
                lines.append(f"\n  Per-Class MIA AUC (top vulnerable):")
                sorted_mia = sorted(
                    ((c, a) for c, a in mia['per_class_auc'].items() if a is not None),
                    key=lambda x: x[1], reverse=True)
                for cls, auc in sorted_mia[:7]:
                    flag = " *** VULNERABLE" if auc > 0.6 else ""
                    lines.append(f"    {cls:<25} AUC={auc:.3f}{flag}")

            lines.append(f"\n  Rarity-Memorization (Spearman):")
            lines.append(f"    rho={sp['rho']:+.4f}, p={sp['p_value']:.2e}, "
                         f"{sp['direction']} {'(significant)' if sp['significant'] else ''}")

            if mw and 'error' not in mw:
                lines.append(f"\n  Rare vs Common (Mann-Whitney U):")
                lines.append(f"    Rare mean:  {mw['rare_mean']:+.4f}")
                lines.append(f"    Common mean: {mw['common_mean']:+.4f}")
                lines.append(f"    Cohen's d:  {mw['cohens_d']:+.3f}, "
                             f"p={mw['p_value']:.2e} "
                             f"{'(significant)' if mw['significant'] else ''}")

            lines.append(f"\n  Per-Class Memorization:")
            lines.append(f"    {'Class':<25} {'Freq%':>6} {'N':>5} {'M(x)':>8} {'High%':>7}")
            lines.append(f"    {'-'*55}")
            for _, row in pc.iterrows():
                flag = " ***" if row['mean_mx'] > 0.3 else ""
                lines.append(
                    f"    {row['class_name']:<25} {row['frequency']*100:>5.1f}% "
                    f"{row['n']:>5} {row['mean_mx']:>+8.3f} "
                    f"{row['high_risk_pct']:>6.1f}%{flag}")

        # Cross-model summary
        lines.append(f"\n{'='*70}")
        lines.append(f"  HIGHEST-RISK CLASSES (averaged across models)")
        lines.append(f"{'='*70}")
        for cls, freq, avg_mx in self.most_vulnerable_classes():
            lines.append(f"    {cls:<25} freq={freq*100:.1f}%, avg M(x)={avg_mx:+.3f}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def audit(data, name="dataset", csv=None, models=None, epochs=30,
          num_augmentations=5, seed=42, output=None, quick=False):
    """
    Run a full memorization audit on an image classification dataset.

    Args:
        data: Path to image directory (folder-per-class or flat with csv).
        name: Dataset name (used for output directory).
        csv: Optional path to CSV with image_id and label columns.
        models: List of model names (default: all 4).
            Supported: 'resnet50', 'vit', 'mae', 'medsam'.
        epochs: Training epochs per model (default: 30).
        num_augmentations: Augmentation passes for M(x) (default: 5).
        seed: Random seed for reproducibility.
        output: Output directory (default: results/audit_<name>).
        quick: Quick mode — 5 epochs, 1 augmentation, resnet50+vit only.

    Returns:
        AuditReport with per-model results, statistical tests, and MIA.

    Example:
        import medmem
        report = medmem.audit("/path/to/images", name="skin_lesions")
        report.summary()
        report.save("results/my_audit")
    """
    data_dir = Path(data)

    if quick:
        epochs = 5
        num_augmentations = 1
        models = models or ['resnet50', 'vit']

    if models is None:
        models = ['resnet50', 'vit', 'mae', 'medsam']

    output_dir = Path(output) if output else Path(f'results/audit_{name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[medmem] GPU: {gpu} ({mem:.0f}GB)")

    print(f"\n[medmem] Memorization Audit: {name}")
    print(f"[medmem] Models: {', '.join(models)} | Epochs: {epochs}")

    # Ingest
    if csv:
        samples, classes = ingest_csv(data_dir, Path(csv))
    else:
        samples, classes = ingest_folder(data_dir)

    num_classes = len(classes)

    # Split
    train_samples, canary_samples, test_samples = stratified_split(samples, seed=seed)

    # Dataloaders
    batch_size = 64 if device == 'cuda' and torch.cuda.get_device_properties(0).total_memory > 30e9 else 32
    candidate_ds = ConcatDataset([
        ImageClassificationDataset(train_samples, TRAIN_TRANSFORM),
        ImageClassificationDataset(canary_samples, TRAIN_TRANSFORM),
    ])
    train_ds = ImageClassificationDataset(train_samples, TRAIN_TRANSFORM)
    canary_ds = ImageClassificationDataset(canary_samples, EVAL_TRANSFORM)
    test_ds = ImageClassificationDataset(test_samples, EVAL_TRANSFORM)

    dl_kwargs = dict(num_workers=4, pin_memory=True)
    cand_loader = DataLoader(candidate_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    ind_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    canary_loader = DataLoader(canary_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)

    # Run each model
    report = AuditReport(
        dataset_name=name, classes=classes,
        n_samples=len(samples), output_dir=output_dir,
    )

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            # Candidate
            print(f"  Training candidate (train+canary)...")
            candidate = create_model(model_name, num_classes)
            if candidate is None:
                continue
            candidate = train_model(candidate, cand_loader, epochs, device,
                                    f"{model_name}-cand")

            # Independent
            print(f"  Training independent (train only)...")
            independent = create_model(model_name, num_classes)
            if independent is None:
                continue
            independent = train_model(independent, ind_loader, epochs, device,
                                      f"{model_name}-ind")

            # Score
            print(f"  Computing M(x) ({num_augmentations} augmentations)...")
            canary_df = compute_memorization(candidate, independent, canary_loader,
                                             device, num_augmentations)
            test_df = compute_memorization(candidate, independent, test_loader,
                                           device, 1)

            # MIA + stats
            print(f"  Running MIA + statistical tests...")
            mia = run_mia(canary_df, test_df)
            statistics = compute_statistics(canary_df)

            elapsed = time.time() - t0
            report.models[model_name] = ModelResult(
                model_name=model_name, canary_df=canary_df, test_df=test_df,
                mia=mia, statistics=statistics, time_seconds=elapsed,
            )
            print(f"  Done in {elapsed:.0f}s | Mean M(x)={statistics['overall']['mean_mx']:+.3f} | "
                  f"MIA AUC={mia['best_auc']:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Save and print
    report.save()
    report.summary()
    return report
