#!/usr/bin/env python3
"""
LoRA without DP — ablation baseline for DP-LoRA experiment.
Same LoRA config (rank=8, attention layers), no differential privacy noise.
Trains candidate + independent on HAM10000 with ViT.
"""

import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import pandas as pd
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from peft import LoraConfig, get_peft_model
import timm

from src.data.ham10000 import HAM10000Dataset
from src.data.base_dataset import resolve_path

OUTPUT_DIR = Path('results/ham1000_lora_nodp')
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4

def build_loaders():
    with open('config/experiments/ham1000_baseline.yaml') as f:
        exp = yaml.safe_load(f)
    with open('config/datasets/ham10000.yaml') as f:
        ds = yaml.safe_load(f)
    exp['dataset_config'] = ds
    paths = ds['paths']
    root = resolve_path(paths['images'])
    meta = resolve_path(paths['metadata'])

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = HAM10000Dataset(root, meta, 'train', transform=tf, config=ds)
    canary_ds = HAM10000Dataset(root, meta, 'canary', transform=tf, config=ds)
    test_ds = HAM10000Dataset(root, meta, 'test', transform=tf, config=ds)
    cand_ds = ConcatDataset([train_ds, canary_ds])

    cand_loader = DataLoader(cand_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    ind_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    canary_loader = DataLoader(canary_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    return cand_loader, ind_loader, canary_loader, test_loader

def make_lora_vit():
    vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=7)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['attn.qkv','attn.proj'], lora_dropout=0.1, bias='none')
    model = get_peft_model(vit, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LoRA trainable: {trainable:,} ({100*trainable/sum(p.numel() for p in model.parameters()):.2f}%)')
    return model

def train(model, loader, epochs, device):
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(epochs):
        total_loss, n = 0, 0
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item(); n += 1
        if ep == 0 or ep == epochs-1 or (ep+1) % 10 == 0:
            print(f'  Epoch {ep+1}/{epochs}: loss={total_loss/n:.4f}')
    model.eval()
    return model

@torch.no_grad()
def score(candidate, independent, loader, device):
    candidate.eval(); independent.eval()
    records = []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        meta = batch[2] if len(batch) > 2 else None
        cl = nn.functional.cross_entropy(candidate(imgs), labels, reduction='none')
        il = nn.functional.cross_entropy(independent(imgs), labels, reduction='none')
        for i in range(len(imgs)):
            records.append({
                'image_id': str(meta.get('image_id',['?'])[i]) if meta else f's_{i}',
                'class_name': str(meta.get('class_name',['?'])[i]) if meta else '?',
                'loss_candidate': float(cl[i]), 'loss_independent': float(il[i]),
                'memorization_score': float(il[i]-cl[i]),
            })
    df = pd.DataFrame(records)
    df['risk_category'] = df['memorization_score'].apply(lambda s: 'HIGH' if s>0.3 else ('MOD' if s>0.1 else 'LOW'))
    return df

def main():
    device = 'cuda'
    print('='*60)
    print('  LoRA WITHOUT DP (ablation baseline)')
    print('='*60)
    cand_loader, ind_loader, canary_loader, test_loader = build_loaders()

    print('\n--- Candidate (LoRA, train+canary, NO DP) ---')
    candidate = make_lora_vit().to(device)
    t0 = time.time()
    candidate = train(candidate, cand_loader, EPOCHS, device)
    cand_time = time.time() - t0

    # Accuracy
    correct = total = 0
    with torch.no_grad():
        for b in canary_loader:
            p = candidate(b[0].to(device)).argmax(1)
            correct += (p == b[1].to(device)).sum().item(); total += len(b[1])
    cand_acc = correct/total
    print(f'  Done: {cand_time:.0f}s, acc={cand_acc:.3f}')

    print('\n--- Independent (LoRA, train only, NO DP) ---')
    independent = make_lora_vit().to(device)
    t0 = time.time()
    independent = train(independent, ind_loader, EPOCHS, device)
    ind_time = time.time() - t0
    correct = total = 0
    with torch.no_grad():
        for b in canary_loader:
            p = independent(b[0].to(device)).argmax(1)
            correct += (p == b[1].to(device)).sum().item(); total += len(b[1])
    ind_acc = correct/total
    print(f'  Done: {ind_time:.0f}s, acc={ind_acc:.3f}')

    print('\n--- Memorization scores ---')
    canary_df = score(candidate, independent, canary_loader, device)
    test_df = score(candidate, independent, test_loader, device)

    pc = canary_df.groupby('class_name').agg(
        n=('memorization_score','count'),
        mean_mx=('memorization_score','mean'),
        high_pct=('risk_category', lambda x: (x=='HIGH').mean()*100)
    ).sort_values('mean_mx', ascending=False)

    print(f'\n  Overall mean M(x): {canary_df.memorization_score.mean():.4f}')
    print(f'  High-risk %: {(canary_df.risk_category=="HIGH").mean()*100:.1f}%')
    print(f'  Candidate acc: {cand_acc:.3f}')
    print(f'\n  Per-class:')
    print(pc.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canary_df.to_csv(OUTPUT_DIR / 'memorization_scores.csv', index=False)
    test_df.to_csv(OUTPUT_DIR / 'test_scores.csv', index=False)
    pc.to_csv(OUTPUT_DIR / 'per_class_stats.csv')
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump({'mean_mx': float(canary_df.memorization_score.mean()),
                    'high_risk_pct': float((canary_df.risk_category=='HIGH').mean()*100),
                    'candidate_accuracy': cand_acc, 'independent_accuracy': ind_acc,
                    'lora_rank': 8, 'epochs': EPOCHS, 'dp': False,
                    'time_seconds': cand_time+ind_time}, f, indent=2)
    print(f'\nSaved to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
