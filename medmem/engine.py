# ---------------------------------------------------------------------------
# Portions of this file (the differential candidate/independent M(x) scoring)
# are adapted from "Localizing Memorization in SSL Vision Encoders"
# (Wang et al., NeurIPS 2024):
# https://github.com/sprintml/LocalizingMemorizationInSSL
# The remaining code is original to this project.
# ---------------------------------------------------------------------------
"""
Core engine: model training, memorization scoring, MIA, statistical tests.

All based on the gold-standard differential training approach:
  M(x) = Loss_independent(x) - Loss_candidate(x)
"""

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional


# -----------------------------------------------------------------------
# Model creation
# -----------------------------------------------------------------------

SUPPORTED_MODELS = ['resnet50', 'vit', 'mae', 'medsam']


def create_model(model_name: str, num_classes: int):
    """Create a pretrained model by name."""
    if model_name == 'resnet50':
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'vit':
        import timm
        return timm.create_model('vit_base_patch16_224.augreg_in21k',
                                 pretrained=True, num_classes=num_classes)
    elif model_name == 'mae':
        import timm
        return timm.create_model('vit_base_patch16_224', pretrained=False,
                                 num_classes=num_classes)
    elif model_name == 'medsam':
        try:
            from src.models.base_model import ModelFactory
            import yaml
            with open('config/models/medsam.yaml') as f:
                cfg = yaml.safe_load(f)
            return ModelFactory.create('medsam', cfg, num_classes)
        except Exception as e:
            print(f"[medmem] MedSAM not available ({e}), skipping")
            return None
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_model(model, loader, epochs, device='cuda', name="model", lr=1e-4):
    """Train a model with standard settings."""
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(epochs):
        total_loss, n = 0, 0
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(f"    [{name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}")

    model.eval()
    return model


# -----------------------------------------------------------------------
# Memorization scoring (gold standard)
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_memorization(candidate, independent, loader, device='cuda',
                         num_augmentations=1):
    """
    Gold-standard memorization scoring.

    M(x) = Loss_independent(x) - Loss_candidate(x)
    Positive M(x) = candidate memorized sample x.

    Args:
        candidate: Model trained on train + canary.
        independent: Model trained on train only.
        loader: DataLoader for samples to score.
        device: 'cuda' or 'cpu'.
        num_augmentations: Number of augmentation passes to average over.

    Returns:
        DataFrame with image_id, class_name, loss_candidate, loss_independent,
        memorization_score, risk_category.
    """
    candidate.eval()
    independent.eval()
    use_amp = (device == 'cuda')

    all_records = []
    for _ in range(max(1, num_augmentations)):
        records = []
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            meta = batch[2] if len(batch) > 2 else None

            if use_amp:
                with torch.amp.autocast('cuda'):
                    cl = F.cross_entropy(candidate(images), labels, reduction='none')
                    il = F.cross_entropy(independent(images), labels, reduction='none')
            else:
                cl = F.cross_entropy(candidate(images), labels, reduction='none')
                il = F.cross_entropy(independent(images), labels, reduction='none')

            for i in range(len(images)):
                img_id = meta.get('image_id', ['?'])[i] if meta else f'sample_{i}'
                cls_name = meta.get('class_name', ['?'])[i] if meta else 'unknown'
                records.append({
                    'image_id': str(img_id),
                    'class_name': str(cls_name),
                    'true_label': int(labels[i].item()),
                    'loss_candidate': float(cl[i].item()),
                    'loss_independent': float(il[i].item()),
                })
        all_records.append(records)

    # Average over augmentations
    if num_augmentations <= 1:
        df = pd.DataFrame(all_records[0])
    else:
        merged = {}
        for pass_records in all_records:
            for r in pass_records:
                key = r['image_id']
                if key not in merged:
                    merged[key] = {**r, 'lc_sum': 0, 'li_sum': 0, 'count': 0}
                merged[key]['lc_sum'] += r['loss_candidate']
                merged[key]['li_sum'] += r['loss_independent']
                merged[key]['count'] += 1
        rows = []
        for m in merged.values():
            rows.append({
                'image_id': m['image_id'], 'class_name': m['class_name'],
                'true_label': m['true_label'],
                'loss_candidate': m['lc_sum'] / m['count'],
                'loss_independent': m['li_sum'] / m['count'],
            })
        df = pd.DataFrame(rows)

    df['memorization_score'] = df['loss_independent'] - df['loss_candidate']
    df['risk_category'] = df['memorization_score'].apply(
        lambda s: 'HIGH' if s > 0.3 else ('MODERATE' if s > 0.1 else 'LOW'))
    return df


# -----------------------------------------------------------------------
# Membership Inference Attacks
# -----------------------------------------------------------------------

def run_mia(canary_df, test_df):
    """Run membership inference attacks using memorization scores and losses."""
    member_scores = canary_df['memorization_score'].values
    nonmember_scores = test_df['memorization_score'].values
    all_scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])

    # M(x) threshold attack
    mx_auc = roc_auc_score(labels, all_scores) if len(np.unique(labels)) > 1 else 0.5

    # Loss-based attack
    member_loss = canary_df['loss_candidate'].values
    nonmember_loss = test_df['loss_candidate'].values
    loss_scores = -np.concatenate([member_loss, nonmember_loss])
    loss_auc = roc_auc_score(labels, loss_scores) if len(np.unique(labels)) > 1 else 0.5

    best_auc = max(mx_auc, loss_auc)
    best_method = 'memorization_score' if mx_auc >= loss_auc else 'loss_threshold'

    # Per-class MIA
    per_class = {}
    for cls in canary_df['class_name'].unique():
        m = canary_df[canary_df['class_name'] == cls]['memorization_score'].values
        nm = test_df[test_df['class_name'] == cls]['memorization_score'].values
        if len(m) >= 2 and len(nm) >= 2:
            s = np.concatenate([m, nm])
            l = np.concatenate([np.ones(len(m)), np.zeros(len(nm))])
            try:
                per_class[cls] = float(roc_auc_score(l, s))
            except ValueError:
                per_class[cls] = 0.5
        else:
            per_class[cls] = None

    return {
        'best_auc': float(best_auc),
        'best_method': best_method,
        'memorization_auc': float(mx_auc),
        'loss_auc': float(loss_auc),
        'per_class_auc': per_class,
    }


# -----------------------------------------------------------------------
# Statistical Tests
# -----------------------------------------------------------------------

def compute_statistics(canary_df):
    """Compute statistical evidence for memorization patterns."""
    class_counts = canary_df['class_name'].value_counts()
    total = class_counts.sum()
    class_freq = (class_counts / total).to_dict()

    # Per-class stats
    per_class = canary_df.groupby('class_name').agg(
        n=('memorization_score', 'count'),
        mean_mx=('memorization_score', 'mean'),
        std_mx=('memorization_score', 'std'),
        median_mx=('memorization_score', 'median'),
        high_risk_pct=('risk_category', lambda x: (x == 'HIGH').mean() * 100),
    ).reset_index()
    per_class['frequency'] = per_class['class_name'].map(class_freq)
    per_class = per_class.sort_values('mean_mx', ascending=False)

    # Spearman: frequency vs memorization
    sample_freq = canary_df['class_name'].map(class_freq)
    rho, p = stats.spearmanr(sample_freq, canary_df['memorization_score'])
    spearman = {
        'rho': float(rho), 'p_value': float(p),
        'direction': 'Rare > Common' if rho < 0 else 'Common > Rare',
        'significant': p < 0.05,
    }

    # Mann-Whitney: rare (<5%) vs common (>=15%)
    rare = [c for c, f in class_freq.items() if f < 0.05]
    common = [c for c, f in class_freq.items() if f >= 0.15]
    mann_whitney = {}
    if rare and common:
        r_scores = canary_df[canary_df['class_name'].isin(rare)]['memorization_score']
        c_scores = canary_df[canary_df['class_name'].isin(common)]['memorization_score']
        if len(r_scores) >= 5 and len(c_scores) >= 5:
            _, mw_p = stats.mannwhitneyu(r_scores, c_scores, alternative='two-sided')
            pooled_std = np.sqrt((r_scores.std()**2 + c_scores.std()**2) / 2)
            d = (r_scores.mean() - c_scores.mean()) / pooled_std if pooled_std > 0 else 0
            mann_whitney = {
                'rare_mean': float(r_scores.mean()),
                'common_mean': float(c_scores.mean()),
                'difference': float(r_scores.mean() - c_scores.mean()),
                'p_value': float(mw_p), 'cohens_d': float(d),
                'significant': mw_p < 0.05,
                'rare_classes': rare, 'common_classes': common,
            }

    return {
        'per_class': per_class,
        'spearman': spearman,
        'mann_whitney': mann_whitney,
        'overall': {
            'mean_mx': float(canary_df['memorization_score'].mean()),
            'std_mx': float(canary_df['memorization_score'].std()),
            'high_risk_pct': float((canary_df['risk_category'] == 'HIGH').mean() * 100),
            'n_samples': len(canary_df),
            'n_classes': canary_df['class_name'].nunique(),
        },
    }
