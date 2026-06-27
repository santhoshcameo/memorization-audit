#!/usr/bin/env python3
"""
DP-LoRA Experiment: Differential Privacy with Low-Rank Adaptation

Trains ViT on HAM10000 with DP-LoRA at different epsilon levels,
then measures memorization M(x) and compares to non-DP baseline.

This targets the most vulnerable dataset+model combination:
HAM10000 + ViT (baseline M(x) = +0.473, akiec M(x) = +1.76)

Usage:
    python scripts/run_dp_lora.py                    # all epsilons
    python scripts/run_dp_lora.py --epsilon 8.0      # single epsilon
    nohup python scripts/run_dp_lora.py > dp_lora.log 2>&1 &
"""

import argparse
import json
import sys
import time
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader, ConcatDataset

import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from peft import LoraConfig, get_peft_model, TaskType

from src.models.base_model import ModelFactory
from src.data.ham10000 import create_ham10000_dataloader, HAM10000Dataset
from src.data.base_dataset import resolve_path
from src.data.transforms import TransformFactory


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
RESULTS_ROOT = Path('results')
OUTPUT_DIR = RESULTS_ROOT / 'ham1000_dp_lora'
EPSILONS = [1.0, 3.0, 8.0]
EPOCHS = 30
BATCH_SIZE = 32
MAX_GRAD_NORM = 1.0
LR = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
TARGET_DELTA = 1e-5
NUM_AUGMENTATIONS = 5


def build_config():
    """Build config dict for HAM10000."""
    with open('config/experiments/ham1000_baseline.yaml') as f:
        exp_config = yaml.safe_load(f)
    with open('config/datasets/ham10000.yaml') as f:
        ds_config = yaml.safe_load(f)
    exp_config['dataset_config'] = ds_config
    return exp_config


def create_lora_vit(num_classes=7):
    """Create ViT with LoRA adapters on attention layers."""
    # Load pretrained ViT
    with open('config/models/vit.yaml') as f:
        model_config = yaml.safe_load(f)
    vit = ModelFactory.create('vit', model_config, num_classes)

    # The ViT backbone is a timm model — we need to wrap it for LoRA
    # Apply LoRA to the attention qkv projections in the backbone
    backbone = vit.backbone

    # Find target modules (timm ViT attention layers)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv", "proj"],  # attention QKV and output projection
        lora_dropout=0.1,
        bias="none",
    )

    # Apply LoRA to backbone
    lora_backbone = get_peft_model(backbone, lora_config)

    # Count params
    trainable = sum(p.numel() for p in lora_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_backbone.parameters())
    print(f"LoRA params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    # Replace backbone
    vit.backbone = lora_backbone
    # Make sure classifier is also trainable
    for p in vit.classifier.parameters():
        p.requires_grad = True

    return vit


def create_standard_vit(num_classes=7):
    """Create standard ViT (no LoRA, no DP) for baseline comparison."""
    with open('config/models/vit.yaml') as f:
        model_config = yaml.safe_load(f)
    return ModelFactory.create('vit', model_config, num_classes)


def make_dp_compatible(model):
    """Make model compatible with Opacus DP training."""
    # Opacus needs specific module types — validate and fix
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"  Fixing {len(errors)} Opacus compatibility issues...")
        model = ModuleValidator.fix(model)
    return model


def build_dataloaders(config):
    """Build train (candidate = train+canary), independent (train only), canary, test loaders."""
    ds_config = config['dataset_config']
    paths = ds_config['paths']
    root = resolve_path(paths['images'])
    metadata_path = resolve_path(paths['metadata'])

    # Use eval transforms for all (no augmentation during DP training — cleaner)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = HAM10000Dataset(root, metadata_path, 'train', transform=transform, config=ds_config)
    canary_ds = HAM10000Dataset(root, metadata_path, 'canary', transform=transform, config=ds_config)
    test_ds = HAM10000Dataset(root, metadata_path, 'test', transform=transform, config=ds_config)

    # Candidate sees train + canary
    candidate_ds = ConcatDataset([train_ds, canary_ds])

    candidate_loader = DataLoader(candidate_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
    independent_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=4, pin_memory=True, drop_last=True)
    canary_loader = DataLoader(canary_ds, batch_size=64, shuffle=False,
                               num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"  Candidate: {len(candidate_ds)} samples")
    print(f"  Independent: {len(train_ds)} samples")
    print(f"  Canary: {len(canary_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")

    return candidate_loader, independent_loader, canary_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, privacy_engine=None):
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    """Evaluate accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / max(total, 1)


@torch.no_grad()
def compute_memorization(candidate, independent, loader, device):
    """Compute per-sample M(x) = Loss_ind - Loss_cand."""
    candidate.eval()
    independent.eval()
    records = []

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        metadata = batch[2] if len(batch) > 2 else None

        cand_logits = candidate(images)
        ind_logits = independent(images)

        cand_loss = nn.functional.cross_entropy(cand_logits, labels, reduction='none')
        ind_loss = nn.functional.cross_entropy(ind_logits, labels, reduction='none')

        for i in range(len(images)):
            img_id = metadata.get('image_id', ['unknown'])[i] if metadata else f'sample_{i}'
            cls_name = metadata.get('class_name', ['unknown'])[i] if metadata else 'unknown'
            records.append({
                'image_id': str(img_id),
                'class_name': str(cls_name),
                'true_label': int(labels[i].item()),
                'loss_candidate': float(cand_loss[i].item()),
                'loss_independent': float(ind_loss[i].item()),
                'memorization_score': float(ind_loss[i].item() - cand_loss[i].item()),
            })

    df = pd.DataFrame(records)
    df['risk_category'] = df['memorization_score'].apply(
        lambda s: 'HIGH' if s > 0.3 else ('MODERATE' if s > 0.1 else 'LOW'))
    return df


def train_dp_lora(epsilon, config, device):
    """Train candidate + independent with DP-LoRA at given epsilon."""
    print(f"\n{'='*60}")
    print(f"  DP-LoRA Training: epsilon = {epsilon}")
    print(f"{'='*60}")

    candidate_loader, independent_loader, canary_loader, test_loader = build_dataloaders(config)

    # --- Train CANDIDATE (DP-LoRA, sees train+canary) ---
    print("\n--- Training CANDIDATE (DP-LoRA, train+canary) ---")
    candidate = create_lora_vit(num_classes=7)
    candidate = make_dp_compatible(candidate)
    candidate = candidate.to(device)

    optimizer = torch.optim.Adam(
        [p for p in candidate.parameters() if p.requires_grad], lr=LR)

    # Attach privacy engine
    privacy_engine = PrivacyEngine()
    candidate, optimizer, candidate_loader_dp = privacy_engine.make_private_with_epsilon(
        module=candidate,
        optimizer=optimizer,
        data_loader=candidate_loader,
        epochs=EPOCHS,
        target_epsilon=epsilon,
        target_delta=TARGET_DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print(f"  Noise multiplier: {optimizer.noise_multiplier:.4f}")
    print(f"  Target: epsilon={epsilon}, delta={TARGET_DELTA}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        loss = train_one_epoch(candidate, candidate_loader_dp, optimizer, device, privacy_engine)
        eps = privacy_engine.get_epsilon(TARGET_DELTA)
        if epoch == 0 or epoch == EPOCHS - 1 or (epoch + 1) % 10 == 0:
            acc = evaluate(candidate, canary_loader, device)
            print(f"  Epoch {epoch+1}/{EPOCHS}: loss={loss:.4f}, acc={acc:.3f}, eps={eps:.2f}")

    cand_time = time.time() - t0
    final_eps = privacy_engine.get_epsilon(TARGET_DELTA)
    cand_acc = evaluate(candidate, canary_loader, device)
    print(f"  Candidate done: {cand_time:.0f}s, final eps={final_eps:.2f}, acc={cand_acc:.3f}")

    # --- Train INDEPENDENT (DP-LoRA, sees train only) ---
    print("\n--- Training INDEPENDENT (DP-LoRA, train only) ---")
    independent = create_lora_vit(num_classes=7)
    independent = make_dp_compatible(independent)
    independent = independent.to(device)

    optimizer_ind = torch.optim.Adam(
        [p for p in independent.parameters() if p.requires_grad], lr=LR)

    privacy_engine_ind = PrivacyEngine()
    independent, optimizer_ind, independent_loader_dp = privacy_engine_ind.make_private_with_epsilon(
        module=independent,
        optimizer=optimizer_ind,
        data_loader=independent_loader,
        epochs=EPOCHS,
        target_epsilon=epsilon,
        target_delta=TARGET_DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    t0 = time.time()
    for epoch in range(EPOCHS):
        loss = train_one_epoch(independent, independent_loader_dp, optimizer_ind, device)
        if epoch == 0 or epoch == EPOCHS - 1 or (epoch + 1) % 10 == 0:
            eps_ind = privacy_engine_ind.get_epsilon(TARGET_DELTA)
            print(f"  Epoch {epoch+1}/{EPOCHS}: loss={loss:.4f}, eps={eps_ind:.2f}")

    ind_time = time.time() - t0
    ind_acc = evaluate(independent, canary_loader, device)
    print(f"  Independent done: {ind_time:.0f}s, acc={ind_acc:.3f}")

    # --- Compute memorization ---
    print("\n--- Computing memorization scores ---")
    canary_df = compute_memorization(candidate, independent, canary_loader, device)
    test_df = compute_memorization(candidate, independent, test_loader, device)

    # --- Results ---
    per_class = canary_df.groupby('class_name').agg(
        n=('memorization_score', 'count'),
        mean_mx=('memorization_score', 'mean'),
        high_risk_pct=('risk_category', lambda x: (x == 'HIGH').mean() * 100),
    ).sort_values('mean_mx', ascending=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS: epsilon = {epsilon}")
    print(f"{'='*60}")
    print(f"  Overall mean M(x): {canary_df['memorization_score'].mean():.4f}")
    print(f"  High-risk %: {(canary_df['risk_category'] == 'HIGH').mean()*100:.1f}%")
    print(f"  Candidate accuracy: {cand_acc:.3f}")
    print(f"  Final epsilon: {final_eps:.2f}")
    print(f"\n  Per-class memorization:")
    print(per_class.to_string())

    # Save results
    out_dir = OUTPUT_DIR / f'eps_{epsilon}'
    out_dir.mkdir(parents=True, exist_ok=True)
    canary_df.to_csv(out_dir / 'memorization_scores.csv', index=False)
    test_df.to_csv(out_dir / 'test_scores.csv', index=False)
    per_class.to_csv(out_dir / 'per_class_stats.csv')

    summary = {
        'epsilon': epsilon,
        'final_epsilon': float(final_eps),
        'delta': TARGET_DELTA,
        'lora_rank': LORA_RANK,
        'epochs': EPOCHS,
        'mean_mx': float(canary_df['memorization_score'].mean()),
        'high_risk_pct': float((canary_df['risk_category'] == 'HIGH').mean() * 100),
        'candidate_accuracy': float(cand_acc),
        'independent_accuracy': float(ind_acc),
        'noise_multiplier': float(optimizer.noise_multiplier),
        'training_time_seconds': float(cand_time + ind_time),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    del candidate, independent
    torch.cuda.empty_cache()

    return summary, per_class


def main():
    parser = argparse.ArgumentParser(description='DP-LoRA experiment on HAM10000 + ViT')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Run single epsilon (default: all [1.0, 3.0, 8.0])')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Opacus: {opacus.__version__}")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")

    config = build_config()
    epsilons = [args.epsilon] if args.epsilon else EPSILONS

    all_summaries = []
    all_per_class = {}

    for eps in epsilons:
        summary, per_class = train_dp_lora(eps, config, device)
        all_summaries.append(summary)
        all_per_class[eps] = per_class

    # --- Load baseline for comparison ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON: DP-LoRA vs Baseline (no DP)")
    print(f"{'='*60}")

    baseline_csv = RESULTS_ROOT / 'ham1000_baseline' / 'memorization' / 'vit_memorization_scores.csv'
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
        baseline_per_class = baseline_df.groupby('class_name')['memorization_score'].mean()

        print(f"\n{'Class':<12} {'Baseline':>10} ", end='')
        for eps in epsilons:
            print(f"{'eps='+str(eps):>10} ", end='')
        print(f"{'Reduction':>10}")
        print("-" * (12 + 10 + 10 * len(epsilons) + 10))

        for cls in baseline_per_class.sort_values(ascending=False).index:
            bl = baseline_per_class[cls]
            print(f"{cls:<12} {bl:>10.3f} ", end='')
            for eps in epsilons:
                if eps in all_per_class and cls in all_per_class[eps].index:
                    dp_val = all_per_class[eps].loc[cls, 'mean_mx']
                    print(f"{dp_val:>10.3f} ", end='')
                else:
                    print(f"{'N/A':>10} ", end='')
            # Show reduction for tightest epsilon
            if epsilons[0] in all_per_class and cls in all_per_class[epsilons[0]].index:
                dp_min = all_per_class[epsilons[0]].loc[cls, 'mean_mx']
                reduction = (1 - dp_min / bl) * 100 if bl != 0 else 0
                print(f"{reduction:>9.1f}%")
            else:
                print(f"{'N/A':>10}")

    # Save combined summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'experiment_summary.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
