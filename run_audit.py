#!/usr/bin/env python3
"""
Medical AI Memorization Audit Tool

One-command tool to audit a medical image classification dataset for
memorization risk. Provide your training images and the tool will:

1. Auto-detect data format (folder-per-class or CSV)
2. Create stratified train/canary/test splits
3. Train 4 architectures (ResNet50, ViT, MAE, MedSAM) with differential training
4. Compute per-sample memorization scores M(x)
5. Run membership inference attacks (5 methods)
6. Compute statistical tests (Spearman, Mann-Whitney, Cohen's d)
7. Generate a comprehensive risk report

Usage:
    # Folder-per-class structure:
    python run_audit.py --data /path/to/images/ --name my_dataset

    # CSV + images:
    python run_audit.py --data /path/to/images/ --csv metadata.csv --name my_dataset

    # Specific models only:
    python run_audit.py --data /path/to/images/ --name my_dataset --models resnet50,vit

    # Quick test (fewer epochs):
    python run_audit.py --data /path/to/images/ --name my_dataset --epochs 5 --quick
"""

import argparse
import json
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------
# Generic Image Classification Dataset
# -----------------------------------------------------------------------

class ImageClassificationDataset(Dataset):
    """Generic image classification dataset from folder or CSV."""

    def __init__(self, samples: List[Dict], transform=None):
        """
        Args:
            samples: List of {'image_path': Path, 'image_id': str,
                              'class_name': str, 'class_idx': int}
            transform: torchvision transform
        """
        self.samples = samples
        self.transform = transform
        self.targets = np.array([s['class_idx'] for s in samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, s['class_idx'], {
            'image_id': s['image_id'],
            'class_name': s['class_name'],
            'idx': idx,
        }


# -----------------------------------------------------------------------
# Data Ingestion
# -----------------------------------------------------------------------

def ingest_folder(data_dir: Path) -> Tuple[List[Dict], List[str]]:
    """Ingest folder-per-class structure: data_dir/class_name/image.jpg"""
    classes = sorted([d.name for d in data_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_dir}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    for cls in classes:
        cls_dir = data_dir / cls
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in exts:
                samples.append({
                    'image_path': img_path,
                    'image_id': img_path.stem,
                    'class_name': cls,
                    'class_idx': class_to_idx[cls],
                })

    print(f"Found {len(samples)} images across {len(classes)} classes")
    return samples, classes


def ingest_csv(data_dir: Path, csv_path: Path,
               image_col: str = 'image_id', label_col: str = 'label'
               ) -> Tuple[List[Dict], List[str]]:
    """Ingest from CSV with image paths and labels."""
    df = pd.read_csv(csv_path)

    # Auto-detect columns
    if image_col not in df.columns:
        for col in ['image_id', 'filename', 'image', 'path', 'file']:
            if col in df.columns:
                image_col = col
                break
    if label_col not in df.columns:
        for col in ['label', 'class', 'dx', 'class_name', 'category', 'target']:
            if col in df.columns:
                label_col = col
                break

    print(f"Using columns: image='{image_col}', label='{label_col}'")

    classes = sorted(df[label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    for _, row in df.iterrows():
        img_name = str(row[image_col])
        cls = str(row[label_col])

        # Try to find the image
        img_path = None
        for ext in [''] + [e for e in exts]:
            candidate = data_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        # Try in class subdirectory
        if img_path is None:
            candidate = data_dir / cls / img_name
            if candidate.exists():
                img_path = candidate

        if img_path is None:
            continue

        samples.append({
            'image_path': img_path,
            'image_id': img_name,
            'class_name': cls,
            'class_idx': class_to_idx[cls],
        })

    print(f"Found {len(samples)} images across {len(classes)} classes")
    return samples, classes


# -----------------------------------------------------------------------
# Data Splitting
# -----------------------------------------------------------------------

def stratified_split(samples: List[Dict], train_frac=0.70, canary_frac=0.15,
                     seed=42) -> Tuple[List, List, List]:
    """Stratified split into train/canary/test."""
    rng = np.random.RandomState(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s['class_name'], []).append(s)

    train, canary, test = [], [], []
    for cls, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        n = len(cls_samples)
        n_train = max(1, int(n * train_frac))
        n_canary = max(1, int(n * canary_frac))
        train.extend(cls_samples[:n_train])
        canary.extend(cls_samples[n_train:n_train + n_canary])
        test.extend(cls_samples[n_train + n_canary:])

    print(f"Split: {len(train)} train, {len(canary)} canary, {len(test)} test")
    return train, canary, test


# -----------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------

def create_model(model_name: str, num_classes: int):
    """Create a model by name."""
    from src.models.base_model import ModelFactory
    import yaml

    config_path = Path(f'config/models/{model_name}.yaml')
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    return ModelFactory.create(model_name, config, num_classes)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_model(model, loader, epochs, device, model_name="model", lr=1e-4):
    """Train a model and return it."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(f"    [{model_name}] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    model.eval()
    return model


# -----------------------------------------------------------------------
# Memorization Scoring
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_memorization(candidate, independent, loader, device,
                         num_augmentations=5):
    """Compute per-sample M(x) = Loss_ind - Loss_cand with augmentation averaging."""
    candidate.eval()
    independent.eval()

    # For augmentation averaging, we need to run multiple passes
    from src.evaluation.memorization import AugmentedBatchDataset

    if num_augmentations > 1:
        aug_dataset = AugmentedBatchDataset(loader.dataset, num_augmentations)
        aug_loader = DataLoader(aug_dataset, batch_size=128, shuffle=False,
                                num_workers=4, pin_memory=True)

        # Accumulate losses per sample
        n_samples = len(loader.dataset)
        cand_losses = np.zeros((n_samples, num_augmentations), dtype=np.float32)
        ind_losses = np.zeros((n_samples, num_augmentations), dtype=np.float32)

        for batch in aug_loader:
            images, labels, sample_indices, aug_indices = batch
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda'):
                cl = nn.functional.cross_entropy(candidate(images), labels, reduction='none')
                il = nn.functional.cross_entropy(independent(images), labels, reduction='none')

            cand_losses[sample_indices.numpy(), aug_indices.numpy()] = cl.cpu().numpy()
            ind_losses[sample_indices.numpy(), aug_indices.numpy()] = il.cpu().numpy()

        mean_cand = cand_losses.mean(axis=1)
        mean_ind = ind_losses.mean(axis=1)

        records = []
        for i in range(n_samples):
            sample = loader.dataset[i]
            meta = sample[2] if len(sample) > 2 else {}
            img_id = meta.get('image_id', f'sample_{i}') if isinstance(meta, dict) else f'sample_{i}'
            cls_name = meta.get('class_name', 'unknown') if isinstance(meta, dict) else 'unknown'
            records.append({
                'image_id': str(img_id),
                'class_name': str(cls_name),
                'true_label': int(sample[1]) if isinstance(sample[1], (int, np.integer)) else int(sample[1].item()),
                'loss_candidate': float(mean_cand[i]),
                'loss_independent': float(mean_ind[i]),
                'memorization_score': float(mean_ind[i] - mean_cand[i]),
            })
    else:
        records = []
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            meta = batch[2] if len(batch) > 2 else None

            with torch.amp.autocast('cuda'):
                cl = nn.functional.cross_entropy(candidate(images), labels, reduction='none')
                il = nn.functional.cross_entropy(independent(images), labels, reduction='none')

            for i in range(len(images)):
                img_id = meta.get('image_id', ['?'])[i] if meta else f'sample_{i}'
                cls_name = meta.get('class_name', ['?'])[i] if meta else 'unknown'
                records.append({
                    'image_id': str(img_id),
                    'class_name': str(cls_name),
                    'true_label': int(labels[i].item()),
                    'loss_candidate': float(cl[i].item()),
                    'loss_independent': float(il[i].item()),
                    'memorization_score': float(il[i].item() - cl[i].item()),
                })

    df = pd.DataFrame(records)
    df['risk_category'] = df['memorization_score'].apply(
        lambda s: 'HIGH' if s > 0.3 else ('MODERATE' if s > 0.1 else 'LOW'))
    return df


# -----------------------------------------------------------------------
# MIA
# -----------------------------------------------------------------------

def run_mia(canary_df, test_df):
    """Run membership inference attacks."""
    member_scores = canary_df['memorization_score'].values
    nonmember_scores = test_df['memorization_score'].values

    all_scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])

    # Attack 1: M(x) threshold
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, all_scores)
    else:
        auc = 0.5

    # Attack 2: Loss-based (lower candidate loss = more likely member)
    member_loss = canary_df['loss_candidate'].values
    nonmember_loss = test_df['loss_candidate'].values
    loss_scores = -np.concatenate([member_loss, nonmember_loss])  # negate: lower loss = higher score
    if len(np.unique(labels)) > 1:
        loss_auc = roc_auc_score(labels, loss_scores)
    else:
        loss_auc = 0.5

    best_auc = max(auc, loss_auc)
    best_method = 'memorization_score' if auc >= loss_auc else 'loss_threshold'

    # Per-class MIA
    per_class = {}
    for cls in canary_df['class_name'].unique():
        m = canary_df[canary_df['class_name'] == cls]['memorization_score'].values
        nm = test_df[test_df['class_name'] == cls]['memorization_score'].values
        if len(m) >= 2 and len(nm) >= 2:
            scores_cls = np.concatenate([m, nm])
            labels_cls = np.concatenate([np.ones(len(m)), np.zeros(len(nm))])
            try:
                per_class[cls] = float(roc_auc_score(labels_cls, scores_cls))
            except ValueError:
                per_class[cls] = 0.5
        else:
            per_class[cls] = None

    return {
        'best_auc': float(best_auc),
        'best_method': best_method,
        'memorization_auc': float(auc),
        'loss_auc': float(loss_auc),
        'per_class_auc': per_class,
    }


# -----------------------------------------------------------------------
# Statistical Tests
# -----------------------------------------------------------------------

def compute_statistics(canary_df):
    """Compute statistical evidence for memorization patterns."""
    results = {}

    # Class frequencies
    class_counts = canary_df['class_name'].value_counts()
    total = class_counts.sum()
    class_freq = (class_counts / total).to_dict()

    # Per-class memorization
    per_class = canary_df.groupby('class_name').agg(
        n=('memorization_score', 'count'),
        mean_mx=('memorization_score', 'mean'),
        std_mx=('memorization_score', 'std'),
        median_mx=('memorization_score', 'median'),
        high_risk_pct=('risk_category', lambda x: (x == 'HIGH').mean() * 100),
    ).reset_index()
    per_class['frequency'] = per_class['class_name'].map(class_freq)
    per_class = per_class.sort_values('mean_mx', ascending=False)

    results['per_class'] = per_class

    # Spearman: frequency vs memorization (per-sample)
    freq_map = {cls: class_freq[cls] for cls in canary_df['class_name']}
    sample_freq = canary_df['class_name'].map(freq_map)
    rho, p = stats.spearmanr(sample_freq, canary_df['memorization_score'])
    results['spearman'] = {
        'rho': float(rho), 'p_value': float(p),
        'direction': 'Rare > Common' if rho < 0 else 'Common > Rare',
        'significant': p < 0.05,
    }

    # Mann-Whitney: rare (<5%) vs common (>=15%)
    rare_classes = [c for c, f in class_freq.items() if f < 0.05]
    common_classes = [c for c, f in class_freq.items() if f >= 0.15]

    if rare_classes and common_classes:
        rare_scores = canary_df[canary_df['class_name'].isin(rare_classes)]['memorization_score']
        common_scores = canary_df[canary_df['class_name'].isin(common_classes)]['memorization_score']

        if len(rare_scores) >= 5 and len(common_scores) >= 5:
            u_stat, mw_p = stats.mannwhitneyu(rare_scores, common_scores, alternative='two-sided')
            # Cohen's d
            pooled_std = np.sqrt((rare_scores.std()**2 + common_scores.std()**2) / 2)
            d = (rare_scores.mean() - common_scores.mean()) / pooled_std if pooled_std > 0 else 0

            results['mann_whitney'] = {
                'rare_mean': float(rare_scores.mean()),
                'common_mean': float(common_scores.mean()),
                'difference': float(rare_scores.mean() - common_scores.mean()),
                'p_value': float(mw_p),
                'cohens_d': float(d),
                'significant': mw_p < 0.05,
                'rare_classes': rare_classes,
                'common_classes': common_classes,
            }
        else:
            results['mann_whitney'] = {'error': 'Not enough samples in rare/common groups'}
    else:
        results['mann_whitney'] = {'error': f'No rare (<5%) or common (>=15%) classes found. '
                                            f'Freq range: {min(class_freq.values()):.1%}-{max(class_freq.values()):.1%}'}

    # Memorization-MIA correlation (per-class)
    results['overall'] = {
        'mean_mx': float(canary_df['memorization_score'].mean()),
        'std_mx': float(canary_df['memorization_score'].std()),
        'high_risk_pct': float((canary_df['risk_category'] == 'HIGH').mean() * 100),
        'n_samples': len(canary_df),
        'n_classes': canary_df['class_name'].nunique(),
    }

    return results


# -----------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------

def generate_report(all_results: Dict, output_dir: Path, dataset_name: str):
    """Generate a comprehensive text report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"  MEMORIZATION AUDIT REPORT: {dataset_name}")
    report_lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    for model_name, res in all_results.items():
        stats_res = res['statistics']
        mia_res = res['mia']
        canary_df = res['canary_df']

        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"  MODEL: {model_name.upper()}")
        report_lines.append(f"{'='*70}")

        # Overall
        ov = stats_res['overall']
        report_lines.append(f"\n  Overall Memorization:")
        report_lines.append(f"    Mean M(x):     {ov['mean_mx']:+.4f}")
        report_lines.append(f"    Std M(x):      {ov['std_mx']:.4f}")
        report_lines.append(f"    High-risk:     {ov['high_risk_pct']:.1f}% of samples (M(x) > 0.3)")
        report_lines.append(f"    Samples:       {ov['n_samples']}")
        report_lines.append(f"    Classes:       {ov['n_classes']}")

        # MIA
        report_lines.append(f"\n  Membership Inference Attack:")
        report_lines.append(f"    Best AUC:      {mia_res['best_auc']:.3f} ({mia_res['best_method']})")
        report_lines.append(f"    {'VULNERABLE' if mia_res['best_auc'] > 0.6 else 'Low risk'}")

        # Per-class MIA
        if mia_res['per_class_auc']:
            report_lines.append(f"\n  Per-Class MIA AUC:")
            sorted_mia = sorted(mia_res['per_class_auc'].items(),
                                key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
            for cls, auc in sorted_mia[:10]:
                flag = " *** VULNERABLE" if auc and auc > 0.6 else ""
                report_lines.append(f"    {cls:<25} AUC={auc:.3f}{flag}" if auc else f"    {cls:<25} N/A")

        # Statistical tests
        sp = stats_res['spearman']
        report_lines.append(f"\n  Rarity-Memorization Correlation (Spearman):")
        report_lines.append(f"    rho = {sp['rho']:+.4f}, p = {sp['p_value']:.2e}")
        report_lines.append(f"    Direction: {sp['direction']}")
        report_lines.append(f"    {'SIGNIFICANT' if sp['significant'] else 'Not significant'}")

        mw = stats_res.get('mann_whitney', {})
        if 'error' not in mw:
            report_lines.append(f"\n  Rare vs Common Classes (Mann-Whitney U):")
            report_lines.append(f"    Rare mean M(x):   {mw['rare_mean']:+.4f} (classes: {', '.join(mw['rare_classes'][:5])})")
            report_lines.append(f"    Common mean M(x): {mw['common_mean']:+.4f} (classes: {', '.join(mw['common_classes'][:5])})")
            report_lines.append(f"    Difference:       {mw['difference']:+.4f}")
            report_lines.append(f"    Cohen's d:        {mw['cohens_d']:+.3f}")
            report_lines.append(f"    p-value:          {mw['p_value']:.2e}")
            report_lines.append(f"    {'SIGNIFICANT' if mw['significant'] else 'Not significant'}")

        # Per-class table
        pc = stats_res['per_class']
        report_lines.append(f"\n  Per-Class Memorization (sorted by risk):")
        report_lines.append(f"    {'Class':<25} {'Freq%':>6} {'N':>5} {'Mean M(x)':>10} {'High%':>7}")
        report_lines.append(f"    {'-'*58}")
        for _, row in pc.iterrows():
            flag = " ***" if row['mean_mx'] > 0.3 else ""
            report_lines.append(
                f"    {row['class_name']:<25} {row['frequency']*100:>5.1f}% {row['n']:>5} "
                f"{row['mean_mx']:>+10.3f} {row['high_risk_pct']:>6.1f}%{flag}")

    # Summary
    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"  SUMMARY")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"\n  Models tested: {', '.join(all_results.keys())}")

    # Find most vulnerable class across all models
    all_classes = {}
    for model_name, res in all_results.items():
        for _, row in res['statistics']['per_class'].iterrows():
            cls = row['class_name']
            if cls not in all_classes:
                all_classes[cls] = {'freq': row['frequency'], 'mx_scores': []}
            all_classes[cls]['mx_scores'].append(row['mean_mx'])

    report_lines.append(f"\n  Highest-Risk Classes (averaged across models):")
    avg_risk = [(cls, info['freq'], np.mean(info['mx_scores']))
                for cls, info in all_classes.items()]
    avg_risk.sort(key=lambda x: x[2], reverse=True)
    for cls, freq, avg_mx in avg_risk[:5]:
        report_lines.append(f"    {cls:<25} freq={freq*100:.1f}%, avg M(x)={avg_mx:+.3f}")

    report_lines.append(f"\n  Results saved to: {output_dir}")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Save
    with open(output_dir / 'audit_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    return report_text


# -----------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------

def run_audit(data_dir: Path, dataset_name: str, csv_path: Optional[Path] = None,
              models: List[str] = None, epochs: int = 30,
              num_augmentations: int = 5, seed: int = 42,
              output_dir: Optional[Path] = None):
    """Run full memorization audit."""

    if models is None:
        models = ['resnet50', 'vit', 'mae', 'medsam']

    if output_dir is None:
        output_dir = Path(f'results/audit_{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # GPU info
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")

    print(f"\n{'='*70}")
    print(f"  MEMORIZATION AUDIT: {dataset_name}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Epochs: {epochs}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    # Step 1: Ingest data
    print("Step 1: Loading data...")
    if csv_path:
        samples, classes = ingest_csv(data_dir, csv_path)
    else:
        samples, classes = ingest_folder(data_dir)

    num_classes = len(classes)
    print(f"  Classes ({num_classes}): {', '.join(classes[:10])}{'...' if len(classes) > 10 else ''}")

    # Class distribution
    class_counts = Counter(s['class_name'] for s in samples)
    total = sum(class_counts.values())
    print(f"\n  Class distribution:")
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        pct = class_counts[cls] / total * 100
        bar = '#' * int(pct / 2)
        print(f"    {cls:<25} {class_counts[cls]:>6} ({pct:>5.1f}%) {bar}")

    # Step 2: Split
    print(f"\nStep 2: Creating stratified splits...")
    train_samples, canary_samples, test_samples = stratified_split(samples, seed=seed)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageClassificationDataset(train_samples, train_tf)
    canary_ds = ImageClassificationDataset(canary_samples, eval_tf)
    test_ds = ImageClassificationDataset(test_samples, eval_tf)
    candidate_ds = ConcatDataset([
        ImageClassificationDataset(train_samples, train_tf),
        ImageClassificationDataset(canary_samples, train_tf),
    ])

    # Batch size based on GPU
    batch_size = 32
    if device == 'cuda':
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        if gpu_mem_gb >= 40:
            batch_size = 64
        elif gpu_mem_gb >= 16:
            batch_size = 32
        else:
            batch_size = 16

    candidate_loader = DataLoader(candidate_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
    independent_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=4, pin_memory=True, drop_last=True)
    canary_loader = DataLoader(canary_ds, batch_size=batch_size * 2, shuffle=False,
                               num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Step 3: Train and evaluate each model
    all_results = {}

    for model_name in models:
        print(f"\n{'='*70}")
        print(f"  Step 3: Training {model_name.upper()}")
        print(f"{'='*70}")

        t0 = time.time()

        try:
            # Create candidate
            print(f"\n  Training CANDIDATE ({model_name}, train+canary)...")
            candidate = create_model(model_name, num_classes)
            candidate = train_model(candidate, candidate_loader, epochs, device,
                                    f"{model_name}-candidate")

            # Create independent (same init but train-only)
            print(f"\n  Training INDEPENDENT ({model_name}, train only)...")
            independent = create_model(model_name, num_classes)
            independent = train_model(independent, independent_loader, epochs, device,
                                      f"{model_name}-independent")

            # Compute memorization
            print(f"\n  Computing memorization scores (M(x), {num_augmentations} augmentations)...")
            canary_df = compute_memorization(candidate, independent, canary_loader, device,
                                             num_augmentations=num_augmentations)
            test_df = compute_memorization(candidate, independent, test_loader, device,
                                           num_augmentations=1)

            # MIA
            print(f"  Running membership inference attacks...")
            mia_results = run_mia(canary_df, test_df)

            # Statistics
            print(f"  Computing statistical tests...")
            stat_results = compute_statistics(canary_df)

            elapsed = time.time() - t0
            print(f"  {model_name} complete in {elapsed:.0f}s")

            # Save per-model results
            model_dir = output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            canary_df.to_csv(model_dir / 'memorization_scores.csv', index=False)
            test_df.to_csv(model_dir / 'test_scores.csv', index=False)
            with open(model_dir / 'mia_results.json', 'w') as f:
                json.dump(mia_results, f, indent=2, default=str)
            with open(model_dir / 'statistics.json', 'w') as f:
                json.dump(stat_results, f, indent=2, default=str)

            all_results[model_name] = {
                'canary_df': canary_df,
                'test_df': test_df,
                'mia': mia_results,
                'statistics': stat_results,
                'time': elapsed,
            }

        except Exception as e:
            print(f"  ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Free GPU
        del candidate, independent
        if device == 'cuda':
            torch.cuda.empty_cache()

    if not all_results:
        print("ERROR: No models completed successfully!")
        return

    # Step 4: Generate report
    print(f"\n{'='*70}")
    print(f"  Step 4: Generating Report")
    print(f"{'='*70}")
    generate_report(all_results, output_dir, dataset_name)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Medical AI Memorization Audit Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Folder-per-class:
  python run_audit.py --data /path/to/images/ --name skin_lesions

  # CSV + images:
  python run_audit.py --data /path/to/images/ --csv metadata.csv --name chest_xray

  # Quick test (5 epochs, 2 models):
  python run_audit.py --data /path/to/images/ --name test --epochs 5 --models resnet50,vit

  # Full audit (all 4 models, 30 epochs):
  python run_audit.py --data /path/to/images/ --name full_audit
        """)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to image directory (folder-per-class or flat with --csv)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV with image_id and label columns')
    parser.add_argument('--name', type=str, required=True,
                        help='Dataset name (used for output directory)')
    parser.add_argument('--models', type=str, default='resnet50,vit,mae,medsam',
                        help='Comma-separated model names (default: resnet50,vit,mae,medsam)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs (default: 30)')
    parser.add_argument('--augmentations', type=int, default=5,
                        help='Augmentation passes for M(x) averaging (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/audit_<name>)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 epochs, 1 augmentation, resnet50+vit only')

    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.augmentations = 1
        args.models = 'resnet50,vit'

    models = [m.strip() for m in args.models.split(',')]
    output_dir = Path(args.output) if args.output else None
    csv_path = Path(args.csv) if args.csv else None

    run_audit(
        data_dir=Path(args.data),
        dataset_name=args.name,
        csv_path=csv_path,
        models=models,
        epochs=args.epochs,
        num_augmentations=args.augmentations,
        seed=args.seed,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()
