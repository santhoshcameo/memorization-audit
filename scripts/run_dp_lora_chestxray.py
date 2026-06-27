#!/usr/bin/env python3
"""DP-LoRA on ChestXray — second dataset validation."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import pandas as pd
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import timm

from src.data.chestxray import ChestXrayDataset
from src.data.base_dataset import resolve_path

OUTPUT_DIR = Path('results/chestxray_dp_lora')
EPSILONS = [1.0, 3.0, 8.0]
EPOCHS = 30; BATCH_SIZE = 32; LR = 1e-4; MAX_GRAD_NORM = 1.0; TARGET_DELTA = 1e-5

def build_loaders():
    with open('config/experiments/chestxray_baseline.yaml') as f:
        exp = yaml.safe_load(f)
    with open('config/datasets/chestxray.yaml') as f:
        ds = yaml.safe_load(f)
    exp['dataset_config'] = ds
    root = resolve_path(ds['paths']['images'])
    meta = resolve_path(ds['paths']['metadata'])
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_ds = ChestXrayDataset(root, meta, 'train', transform=tf, config=ds)
    canary_ds = ChestXrayDataset(root, meta, 'canary', transform=tf, config=ds)
    test_ds = ChestXrayDataset(root, meta, 'test', transform=tf, config=ds)
    cand_ds = ConcatDataset([train_ds, canary_ds])
    return (DataLoader(cand_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
            DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
            DataLoader(canary_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True))

def make_lora_vit(nc=14):
    vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['attn.qkv','attn.proj'], lora_dropout=0.1, bias='none')
    m = get_peft_model(vit, cfg)
    m = ModuleValidator.fix(m) if ModuleValidator.validate(m, strict=False) else m
    print(f'  LoRA: {sum(p.numel() for p in m.parameters() if p.requires_grad):,} trainable')
    return m

def train_ep(model, loader, optimizer, device):
    model.train(); tl = n = 0
    for b in loader:
        imgs, labels = b[0].to(device), b[1].to(device)
        optimizer.zero_grad(); loss = nn.CrossEntropyLoss()(model(imgs), labels)
        loss.backward(); optimizer.step(); tl += loss.item(); n += 1
    return tl/max(n,1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); c = t = 0
    for b in loader:
        p = model(b[0].to(device)).argmax(1); c += (p==b[1].to(device)).sum().item(); t += len(b[1])
    return c/max(t,1)

@torch.no_grad()
def score(cand, ind, loader, device):
    cand.eval(); ind.eval(); recs = []
    for b in loader:
        imgs, labels = b[0].to(device), b[1].to(device)
        meta = b[2] if len(b)>2 else None
        cl = nn.functional.cross_entropy(cand(imgs), labels, reduction='none')
        il = nn.functional.cross_entropy(ind(imgs), labels, reduction='none')
        for i in range(len(imgs)):
            recs.append({'image_id': str(meta.get('image_id',['?'])[i]) if meta else f's_{i}',
                'class_name': str(meta.get('class_name',['?'])[i]) if meta else '?',
                'loss_candidate': float(cl[i]), 'loss_independent': float(il[i]),
                'memorization_score': float(il[i]-cl[i])})
    df = pd.DataFrame(recs)
    df['risk_category'] = df['memorization_score'].apply(lambda s: 'HIGH' if s>0.3 else ('MOD' if s>0.1 else 'LOW'))
    return df

def run_epsilon(eps, cand_loader, ind_loader, canary_loader, test_loader, device):
    print(f'\n{"="*60}\n  ChestXray DP-LoRA: epsilon={eps}\n{"="*60}')
    # Candidate
    print('--- Candidate (DP-LoRA, train+canary) ---')
    cand = make_lora_vit(14).to(device)
    opt = torch.optim.Adam([p for p in cand.parameters() if p.requires_grad], lr=LR)
    pe = PrivacyEngine()
    cand, opt, dl = pe.make_private_with_epsilon(module=cand, optimizer=opt, data_loader=cand_loader,
        epochs=EPOCHS, target_epsilon=eps, target_delta=TARGET_DELTA, max_grad_norm=MAX_GRAD_NORM)
    print(f'  Noise: {opt.noise_multiplier:.4f}')
    t0 = time.time()
    for ep in range(EPOCHS):
        loss = train_ep(cand, dl, opt, device)
        if ep==0 or ep==EPOCHS-1 or (ep+1)%10==0:
            print(f'  Epoch {ep+1}/{EPOCHS}: loss={loss:.4f}, eps={pe.get_epsilon(TARGET_DELTA):.2f}')
    cand_time = time.time()-t0
    cand_acc = evaluate(cand, canary_loader, device)
    print(f'  Done: {cand_time:.0f}s, acc={cand_acc:.3f}')

    # Independent
    print('--- Independent (DP-LoRA, train only) ---')
    ind = make_lora_vit(14).to(device)
    opt2 = torch.optim.Adam([p for p in ind.parameters() if p.requires_grad], lr=LR)
    pe2 = PrivacyEngine()
    ind, opt2, dl2 = pe2.make_private_with_epsilon(module=ind, optimizer=opt2, data_loader=ind_loader,
        epochs=EPOCHS, target_epsilon=eps, target_delta=TARGET_DELTA, max_grad_norm=MAX_GRAD_NORM)
    t0 = time.time()
    for ep in range(EPOCHS):
        loss = train_ep(ind, dl2, opt2, device)
        if ep==0 or ep==EPOCHS-1 or (ep+1)%10==0:
            print(f'  Epoch {ep+1}/{EPOCHS}: loss={loss:.4f}')
    ind_time = time.time()-t0
    ind_acc = evaluate(ind, canary_loader, device)
    print(f'  Done: {ind_time:.0f}s, acc={ind_acc:.3f}')

    # Score
    canary_df = score(cand, ind, canary_loader, device)
    pc = canary_df.groupby('class_name').agg(n=('memorization_score','count'),
        mean_mx=('memorization_score','mean')).sort_values('mean_mx', ascending=False)
    print(f'\n  Mean M(x): {canary_df.memorization_score.mean():.4f}, Acc: {cand_acc:.3f}')
    print(pc.to_string())

    out = OUTPUT_DIR/f'eps_{eps}'; out.mkdir(parents=True, exist_ok=True)
    canary_df.to_csv(out/'memorization_scores.csv', index=False)
    with open(out/'summary.json','w') as f:
        json.dump({'epsilon':eps,'mean_mx':float(canary_df.memorization_score.mean()),
            'accuracy':cand_acc,'noise':float(opt.noise_multiplier),'time':cand_time+ind_time},f,indent=2)
    del cand, ind; torch.cuda.empty_cache()
    return {'eps':eps,'mean_mx':float(canary_df.memorization_score.mean()),'acc':cand_acc}, pc

def main():
    device = 'cuda'
    print(f'Device: {device}')
    cand_loader, ind_loader, canary_loader, test_loader = build_loaders()
    results = []
    for eps in EPSILONS:
        s, pc = run_epsilon(eps, cand_loader, ind_loader, canary_loader, test_loader, device)
        results.append(s)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR/'experiment_summary.json','w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll done! Results: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
