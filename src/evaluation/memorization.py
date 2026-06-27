# ---------------------------------------------------------------------------
# Portions of this file (the differential candidate/independent M(x) scoring)
# are adapted from "Localizing Memorization in SSL Vision Encoders"
# (Wang et al., NeurIPS 2024):
# https://github.com/sprintml/LocalizingMemorizationInSSL
# The remaining code is original to this project.
# ---------------------------------------------------------------------------
"""
Memorization Measurement - FULLY OPTIMIZED FOR A100
Computes memorization scores using differential privacy approach

OPTIMIZATIONS:
1. Process ALL samples in large batches (not one at a time)
2. Batch augmentations across multiple samples
3. Run both models in parallel batches
4. Use full GPU memory for maximum throughput
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm
import pandas as pd

from ..data.ham10000 import HAM10000Dataset
from ..data.transforms import get_memorization_transform


class AugmentedBatchDataset(Dataset):
    """
    Wrapper dataset that creates multiple augmented versions of each sample.
    This allows efficient batching of augmented samples.
    """

    # Class-level constants for ImageNet normalization (used for denormalization)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, base_dataset: Dataset, num_augmentations: int = 5):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations

        # Strong augmentation transform for memorization measurement
        from torchvision import transforms
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # For denormalization - tensors created on CPU, will be moved to match input device in __getitem__
        self._mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.base_dataset) * self.num_augmentations

    def __getitem__(self, idx):
        # Map to original sample and augmentation index
        sample_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        # Get original sample
        sample = self.base_dataset[sample_idx]
        if len(sample) == 3:
            image, label, metadata = sample
        else:
            image, label = sample
            metadata = {'image_id': f'sample_{sample_idx}', 'class_name': 'unknown', 'idx': sample_idx}

        # Denormalize for augmentation - ensure tensors are on same device as image
        # Move mean/std to image's device to avoid device mismatch errors
        mean = self._mean.to(image.device)
        std = self._std.to(image.device)
        image_denorm = image * std + mean

        # NOTE: We do NOT clamp to [0,1] here to avoid information loss.
        # Some normalized values may denormalize to slightly outside [0,1].
        # Clamping would destroy this information asymmetrically across samples,
        # which could confound memorization measurement.
        # The augmentation transforms handle edge values appropriately.

        # Apply augmentation
        augmented_image = self.augment_transform(image_denorm)

        return augmented_image, label, sample_idx, aug_idx


class MemorizationScorer:
    """
    Measures memorization using differential privacy approach - OPTIMIZED

    Memorization score M(x) for sample x:
        M(x) = Loss_independent(x) - Loss_candidate(x)

    OPTIMIZATIONS:
    1. Process large batches of samples simultaneously
    2. Compute losses for both models in parallel
    3. Use full GPU memory
    4. Efficient augmentation batching
    """

    def __init__(self,
                 candidate_model: nn.Module,
                 independent_model: nn.Module,
                 device: str = 'cuda',
                 model_name: str = 'resnet50'):
        """
        Initialize memorization scorer

        Args:
            candidate_model: Model trained on train + canary
            independent_model: Model trained on train only
            device: Device to use
            model_name: Name of the model (for batch size optimization)
        """
        self.candidate_model = candidate_model.to(device)
        self.independent_model = independent_model.to(device)
        self.device = device
        self.model_name = model_name

        # Set to eval mode
        self.candidate_model.eval()
        self.independent_model.eval()

        # Enable AMP for faster inference
        self.use_amp = device == 'cuda'

        # Get optimal batch size from GPU profile
        from ..utils.gpu_config import get_gpu_profile_name, get_memory_config, get_inference_batch_size
        self.gpu_profile = get_gpu_profile_name()
        mem_config = get_memory_config(self.gpu_profile)
        self.non_blocking = mem_config.get('non_blocking', True)
        self.inference_batch_size = get_inference_batch_size(model_name, self.gpu_profile)

        print("Memorization Scorer initialized - OPTIMIZED VERSION")
        print(f"  Device: {device}")
        print(f"  Model: {model_name}")
        print(f"  GPU Profile: {self.gpu_profile}")
        print(f"  Inference batch size: {self.inference_batch_size}")
        print(f"  Mixed Precision: {'ENABLED' if self.use_amp else 'DISABLED'}")
        print(f"  Non-blocking transfers: {self.non_blocking}")

    @torch.no_grad()
    def compute_batch_losses(self,
                            images: torch.Tensor,
                            labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute losses for a batch on both models simultaneously.

        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)

        Returns:
            Tuple of (candidate_losses, independent_losses) - per-sample losses
        """
        images = images.to(self.device, non_blocking=self.non_blocking)
        labels = labels.to(self.device, non_blocking=self.non_blocking)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                # Candidate model
                outputs_candidate = self.candidate_model(images)
                losses_candidate = F.cross_entropy(outputs_candidate, labels, reduction='none')

                # Independent model
                outputs_independent = self.independent_model(images)
                losses_independent = F.cross_entropy(outputs_independent, labels, reduction='none')
        else:
            outputs_candidate = self.candidate_model(images)
            losses_candidate = F.cross_entropy(outputs_candidate, labels, reduction='none')

            outputs_independent = self.independent_model(images)
            losses_independent = F.cross_entropy(outputs_independent, labels, reduction='none')

        return losses_candidate, losses_independent

    def evaluate_dataset_optimized(self,
                                   base_dataset: Dataset,
                                   num_augmentations: int = 5,
                                   batch_size: int = 256) -> pd.DataFrame:
        """
        Evaluate memorization on entire dataset - FULLY OPTIMIZED

        This version:
        1. Creates augmented dataset with all augmentations pre-defined
        2. Processes in large batches for maximum GPU utilization
        3. Aggregates results per original sample

        Args:
            base_dataset: Original dataset (canary samples)
            num_augmentations: Number of augmentations per sample
            batch_size: Batch size for inference (can be large!)

        Returns:
            DataFrame with results per original sample
        """
        # Create augmented dataset
        aug_dataset = AugmentedBatchDataset(base_dataset, num_augmentations)

        # Create dataloader with large batch size for maximum GPU utilization
        # Get optimal settings from GPU profile
        from ..utils.gpu_config import get_dataloader_config
        dl_config = get_dataloader_config(self.gpu_profile)

        aug_loader = DataLoader(
            aug_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dl_config.get('num_workers', 8),
            pin_memory=dl_config.get('pin_memory', True),
            persistent_workers=dl_config.get('persistent_workers', True),
            prefetch_factor=dl_config.get('prefetch_factor', 4),
            drop_last=False
        )

        num_samples = len(base_dataset)
        total_aug_samples = len(aug_dataset)

        print(f"Measuring memorization on {num_samples} samples...")
        print(f"Using {num_augmentations} augmentations per sample ({total_aug_samples} total)")
        print(f"Batch size: {batch_size} (optimized for GPU)")
        print(f"Number of batches: {len(aug_loader)}")
        print()

        # Storage for losses per (sample_idx, aug_idx) - use numpy arrays to avoid type issues
        candidate_losses = np.zeros((num_samples, num_augmentations), dtype=np.float32)
        independent_losses = np.zeros((num_samples, num_augmentations), dtype=np.float32)

        # Process all augmented samples in large batches
        for batch in tqdm(aug_loader, desc="Computing memorization (batched)"):
            images, labels, sample_indices, aug_indices = batch

            # Compute losses for both models in one go
            loss_cand, loss_ind = self.compute_batch_losses(images, labels)

            # Store results - vectorized for efficiency
            loss_cand_np = loss_cand.cpu().numpy()
            loss_ind_np = loss_ind.cpu().numpy()
            sample_indices_np = sample_indices.numpy()
            aug_indices_np = aug_indices.numpy()

            # Vectorized assignment using advanced indexing
            candidate_losses[sample_indices_np, aug_indices_np] = loss_cand_np
            independent_losses[sample_indices_np, aug_indices_np] = loss_ind_np

        # Aggregate losses per sample (mean across augmentations)
        mean_candidate_losses = candidate_losses.mean(axis=1)
        mean_independent_losses = independent_losses.mean(axis=1)
        memorization_scores = mean_independent_losses - mean_candidate_losses

        # Build results DataFrame
        results = []
        for i in range(num_samples):
            # Get metadata from original dataset
            sample = base_dataset[i]
            if len(sample) == 3:
                _, label, metadata = sample
                if isinstance(metadata, dict):
                    image_id = metadata.get('image_id', f'sample_{i}')
                    class_name = metadata.get('class_name', 'unknown')
                else:
                    image_id = f'sample_{i}'
                    class_name = 'unknown'
            else:
                _, label = sample
                image_id = f'sample_{i}'
                class_name = 'unknown'

            label_val = label.item() if isinstance(label, torch.Tensor) else label

            results.append({
                'image_id': image_id,
                'true_label': label_val,
                'class_name': class_name,
                'loss_candidate': mean_candidate_losses[i],
                'loss_independent': mean_independent_losses[i],
                'memorization_score': memorization_scores[i]
            })

        df = pd.DataFrame(results)

        # Add risk categorization
        df['risk_category'] = self._categorize_risk(df['memorization_score'])

        return df

    def evaluate_dataset(self,
                        dataloader: DataLoader,
                        num_augmentations: int = 5,
                        use_feature_distance: bool = False) -> pd.DataFrame:
        """
        Wrapper for backward compatibility - redirects to optimized version

        Args:
            dataloader: DataLoader with samples to evaluate
            num_augmentations: Number of augmentations per sample
            use_feature_distance: Ignored in optimized version

        Returns:
            DataFrame with results
        """
        # Use model-specific inference batch size
        return self.evaluate_dataset_optimized(
            dataloader.dataset,
            num_augmentations=num_augmentations,
            batch_size=self.inference_batch_size
        )

    def _categorize_risk(self, scores: pd.Series) -> pd.Series:
        """Categorize memorization risk"""
        def categorize(score):
            if score > 0.3:
                return 'high'
            elif score > 0.1:
                return 'moderate'
            else:
                return 'low'

        return scores.apply(categorize)

    def compute_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Compute summary statistics"""
        scores = results_df['memorization_score']

        summary = {
            'mean': scores.mean(),
            'std': scores.std(),
            'median': scores.median(),
            'min': scores.min(),
            'max': scores.max(),
            'q25': scores.quantile(0.25),
            'q75': scores.quantile(0.75),
            'high_risk_count': (results_df['risk_category'] == 'high').sum(),
            'high_risk_percentage': (results_df['risk_category'] == 'high').mean() * 100,
            'moderate_risk_count': (results_df['risk_category'] == 'moderate').sum(),
            'low_risk_count': (results_df['risk_category'] == 'low').sum(),
        }

        return summary

    def compute_per_class_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-class statistics"""
        class_stats = []

        for class_name in results_df['class_name'].unique():
            class_df = results_df[results_df['class_name'] == class_name]
            scores = class_df['memorization_score']

            stats = {
                'class': class_name,
                'count': len(class_df),
                'mean': scores.mean(),
                'std': scores.std(),
                'median': scores.median(),
                'min': scores.min(),
                'max': scores.max(),
                'high_risk_count': (class_df['risk_category'] == 'high').sum(),
                'high_risk_percentage': (class_df['risk_category'] == 'high').mean() * 100
            }

            class_stats.append(stats)

        return pd.DataFrame(class_stats).sort_values('mean', ascending=False)


def measure_memorization(candidate_checkpoint: Path,
                        independent_checkpoint: Path,
                        config: Dict,
                        output_dir: Path,
                        num_augmentations: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Measure memorization for a model pair - OPTIMIZED VERSION
    """
    from ..models import ModelFactory
    from ..utils.gpu_config import get_gpu_profile_name, setup_cuda_optimizations, get_inference_batch_size

    print("="*80)
    print("MEMORIZATION MEASUREMENT - OPTIMIZED FOR A100")
    print("="*80)

    # Ensure CUDA optimizations are applied
    setup_cuda_optimizations()

    # Get model config
    model_name = config.get('model', 'resnet50')
    model_config = config.get('model_configs', {}).get(model_name, {})

    # Get num_classes from dataset config - validate it's properly specified
    dataset_config = config.get('dataset_config', {})
    classes_config = dataset_config.get('classes', {})

    if isinstance(classes_config, dict) and 'names' in classes_config:
        num_classes = len(classes_config['names'])
    else:
        raise ValueError(
            f"Cannot determine num_classes for memorization measurement.\n"
            f"Dataset config must include 'classes.names' list.\n"
            f"Found: {classes_config}\n"
            f"Please ensure dataset config has 'classes.names' properly defined."
        )

    if num_classes == 0:
        raise ValueError(
            f"num_classes is 0 - dataset config 'classes.names' is empty.\n"
            f"Please check your dataset configuration."
        )

    print(f"Dataset: {config.get('dataset', 'unknown')}")
    print(f"Model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print()

    def _create_and_load(checkpoint_path, label):
        """Create model and load checkpoint."""
        model = ModelFactory.create(model_name, model_config, num_classes)
        model.load_checkpoint(checkpoint_path)
        return model

    # Load candidate model
    print(f"Loading candidate model from: {candidate_checkpoint}")
    candidate_model = _create_and_load(candidate_checkpoint, "candidate")
    print("✅ Candidate model loaded")

    # Load independent model
    print(f"Loading independent model from: {independent_checkpoint}")
    independent_model = _create_and_load(independent_checkpoint, "independent")
    print("✅ Independent model loaded")
    print()

    # Create memorization scorer with model-specific batch size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = MemorizationScorer(candidate_model, independent_model, device, model_name=model_name)

    # Create dataloader for canary samples
    dataset_name = config.get('dataset', 'ham10000')
    if dataset_name == 'ham10000':
        from ..data.ham10000 import create_ham10000_dataloader as create_dataloader
    elif dataset_name == 'chestxray':
        from ..data.chestxray import create_chestxray_dataloader as create_dataloader
    elif dataset_name == 'odir5k':
        from ..data.odir5k import create_odir5k_dataloader as create_dataloader
    elif dataset_name == 'retinal_oct':
        from ..data.retinal_oct import create_retinal_oct_dataloader as create_dataloader
    elif dataset_name == 'kvasir_capsule':
        from ..data.kvasir_capsule import create_kvasir_capsule_dataloader as create_dataloader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Using dataloader for: {dataset_name}")

    # Get MODEL-SPECIFIC inference batch size from GPU profile
    gpu_profile = get_gpu_profile_name()
    eval_batch_size = get_inference_batch_size(model_name, gpu_profile)

    print(f"GPU Profile: {gpu_profile}")
    print(f"Model-specific inference batch size: {eval_batch_size}")

    canary_loader = create_dataloader(
        config,
        split='canary',
        shuffle=False,
        batch_size=32,  # Base dataloader batch size
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    print(f"Measuring memorization on {len(canary_loader.dataset)} canary samples")
    print(f"Using {num_augmentations} augmentations per sample")
    print(f"Total forward passes: {len(canary_loader.dataset) * num_augmentations}")
    print()

    # Compute memorization scores using OPTIMIZED method
    results_df = scorer.evaluate_dataset_optimized(
        canary_loader.dataset,
        num_augmentations=num_augmentations,
        batch_size=eval_batch_size
    )

    # Compute statistics
    summary_stats = scorer.compute_summary_statistics(results_df)
    per_class_stats = scorer.compute_per_class_statistics(results_df)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{model_name}_memorization_scores.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✅ Results saved: {results_path}")

    per_class_path = output_dir / f"{model_name}_per_class_stats.csv"
    per_class_stats.to_csv(per_class_path, index=False)
    print(f"✅ Per-class stats saved: {per_class_path}")

    # Print summary
    print()
    print("="*80)
    print("MEMORIZATION SUMMARY (Canary Samples)")
    print("="*80)
    print(f"Mean memorization: {summary_stats['mean']:.4f}")
    print(f"Std: {summary_stats['std']:.4f}")
    print(f"Max: {summary_stats['max']:.4f}")
    print(f"High risk samples: {summary_stats['high_risk_count']} ({summary_stats['high_risk_percentage']:.1f}%)")
    print()
    print("Per-class memorization:")
    print(per_class_stats[['class', 'count', 'mean', 'high_risk_percentage']].to_string(index=False))
    print("="*80)

    # Also evaluate test set for MIA validation (non-member samples)
    print()
    print("="*80)
    print("EVALUATING TEST SET (For MIA Validation)")
    print("="*80)

    try:
        test_loader = create_dataloader(
            config,
            split='test',
            shuffle=False,
            batch_size=32,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        print(f"Evaluating {len(test_loader.dataset)} test samples (non-members for MIA)")

        # Compute losses for test samples (non-members)
        test_results_df = scorer.evaluate_dataset_optimized(
            test_loader.dataset,
            num_augmentations=num_augmentations,
            batch_size=eval_batch_size
        )

        # Save test results for MIA
        test_results_path = output_dir / f"{model_name}_test_scores.csv"
        test_results_df.to_csv(test_results_path, index=False)
        print(f"✅ Test set scores saved: {test_results_path}")
        print(f"   This file is required for MIA validation")

    except Exception as e:
        print(f"⚠️ Could not evaluate test set: {e}")
        print("   MIA validation will not be possible without test set evaluation.")
        print("   Ensure your dataset has a 'test' split defined.")

    print("="*80)

    return results_df, summary_stats
