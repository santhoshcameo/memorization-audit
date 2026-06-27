"""
Data ingestion and splitting for memorization auditing.

Supports:
  - Folder-per-class: data_dir/class_name/image.jpg
  - CSV + images: CSV with image_id and label columns + flat image directory
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageClassificationDataset(Dataset):
    """Generic image classification dataset."""

    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform or EVAL_TRANSFORM
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


def ingest_folder(data_dir) -> Tuple[List[Dict], List[str]]:
    """Load from folder-per-class: data_dir/class_name/image.jpg"""
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_dir}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for cls in classes:
        for p in sorted((data_dir / cls).iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append({
                    'image_path': p, 'image_id': p.stem,
                    'class_name': cls, 'class_idx': class_to_idx[cls],
                })
    print(f"[medmem] Found {len(samples)} images across {len(classes)} classes")
    return samples, classes


def ingest_csv(data_dir, csv_path, image_col='image_id', label_col='label'
               ) -> Tuple[List[Dict], List[str]]:
    """Load from CSV with image paths and labels."""
    data_dir, csv_path = Path(data_dir), Path(csv_path)
    df = pd.read_csv(csv_path)

    # Auto-detect columns
    for col in ['image_id', 'filename', 'image', 'path', 'file']:
        if col in df.columns:
            image_col = col
            break
    for col in ['label', 'class', 'dx', 'class_name', 'category', 'target']:
        if col in df.columns:
            label_col = col
            break

    classes = sorted(df[label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []

    for _, row in df.iterrows():
        img_name = str(row[image_col])
        cls = str(row[label_col])
        img_path = None
        for ext in [''] + [e for e in IMAGE_EXTENSIONS]:
            candidate = data_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            candidate = data_dir / cls / img_name
            if candidate.exists():
                img_path = candidate
        if img_path:
            samples.append({
                'image_path': img_path, 'image_id': img_name,
                'class_name': cls, 'class_idx': class_to_idx[cls],
            })

    print(f"[medmem] Found {len(samples)} images across {len(classes)} classes")
    return samples, classes


def stratified_split(samples, train_frac=0.70, canary_frac=0.15, seed=42):
    """Stratified split into train/canary/test."""
    rng = np.random.RandomState(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s['class_name'], []).append(s)

    train, canary, test = [], [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_frac))
        n_canary = max(1, int(n * canary_frac))
        train.extend(items[:n_train])
        canary.extend(items[n_train:n_train + n_canary])
        test.extend(items[n_train + n_canary:])

    print(f"[medmem] Split: {len(train)} train, {len(canary)} canary, {len(test)} test")
    return train, canary, test
