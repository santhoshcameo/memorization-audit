"""
Differential Trainer
Trains candidate and independent models for memorization measurement
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
import copy
import numpy as np
import random

from .trainer import BaseTrainer
from ..data.ham10000 import create_ham10000_dataloader


def _reset_seed(seed: int = 42):
    """
    Reset all random seeds for reproducible training.
    CRITICAL: Must be called before each training run to ensure
    candidate and independent models experience identical training conditions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DifferentialTrainer:
    """
    Differential trainer for memorization experiments
    
    Trains two models:
    1. Candidate: Sees train + canary samples
    2. Independent: Sees only train samples
    
    Memorization = difference in performance between these models
    """
    
    def __init__(self,
                    model_factory_fn,
                    config: Dict,
                    experiment_name: str = "differential",
                    output_dir: Path = Path("results"),
                    model_name: str = None):
            """
            Initialize differential trainer

            Args:
                model_factory_fn: Function to create model (called twice)
                config: Complete configuration
                experiment_name: Name of experiment
                output_dir: Output directory
                model_name: Model name for batch size lookup (e.g., 'resnet50', 'vit', 'mae', 'medsam')
            """
            self.model_factory_fn = model_factory_fn
            self.config = config
            self.experiment_name = experiment_name
            self.output_dir = Path(output_dir)
            self.model_name = model_name or experiment_name  # Use experiment_name as fallback
            
            # FIX: Remove duplicate experiment_name from path
            if self.output_dir.name == experiment_name:
                self.output_dir = self.output_dir.parent
            
            # Create output directories
            self.candidate_dir = self.output_dir / "candidate"
            self.independent_dir = self.output_dir / "independent"
            self.candidate_dir.mkdir(parents=True, exist_ok=True)
            self.independent_dir.mkdir(parents=True, exist_ok=True)
        
            print("="*80)
            print("DIFFERENTIAL TRAINING SETUP")
            print("="*80)
            print(f"Experiment: {experiment_name}")
            print(f"Output: {output_dir}")
            print()
    
    def create_models(self) -> Tuple[nn.Module, nn.Module]:
        """
        Create candidate and independent models with IDENTICAL initialization.

        CRITICAL: Both models must start from the exact same weights.
        Only the training data differs (candidate sees canary, independent doesn't).
        This ensures M(x) measures memorization, not initialization differences.

        Returns:
            Tuple of (candidate_model, independent_model)
        """
        print("Creating models...")

        # Create base model
        candidate_model = self.model_factory_fn()
        print(f"✅ Candidate model created")

        # CRITICAL FIX: Use deepcopy to ensure IDENTICAL initialization
        # Previously created two separate models with different random weights
        independent_model = copy.deepcopy(candidate_model)
        print(f"✅ Independent model created (deepcopy of candidate for identical init)")

        # Verify identical initialization
        candidate_params = sum(p.numel() for p in candidate_model.parameters())
        independent_params = sum(p.numel() for p in independent_model.parameters())

        if candidate_params != independent_params:
            raise ValueError("Models have different architectures!")

        # Verify weights are identical
        candidate_state = candidate_model.state_dict()
        independent_state = independent_model.state_dict()
        for key in candidate_state:
            if not torch.equal(candidate_state[key], independent_state[key]):
                raise ValueError(f"Models have different weights for {key}! Deepcopy failed.")

        print(f"   Parameters: {candidate_params:,}")
        print(f"   ✅ Verified: Both models have IDENTICAL initial weights")
        print()

        return candidate_model, independent_model
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create data loaders for candidate and independent models

        Returns:
            Tuple of (candidate_loader, independent_loader, test_loader)
        """
        print("Creating data loaders...")

        # Get dataset name
        dataset_name = self.config.get('dataset', 'ham10000')

        # Get model-specific batch_size from GPU profile
        from ..utils.gpu_config import (
            get_model_batch_size, get_dataloader_config, get_gpu_profile_name,
            get_gradient_accumulation, print_gpu_info
        )

        # Use self.model_name which is set during init
        model_name = self.model_name or self.config.get('model', 'resnet50')
        gpu_profile = get_gpu_profile_name()
        print_gpu_info()

        # Get training config overrides (if any)
        training_config = self.config.get('training', {})

        # Use model-specific batch size unless explicitly overridden
        if 'batch_size' in training_config and training_config['batch_size']:
            batch_size = training_config['batch_size']
        else:
            batch_size = get_model_batch_size(model_name, gpu_profile)

        # Get optimal DataLoader settings
        dl_config = get_dataloader_config(gpu_profile)
        num_workers = training_config.get('num_workers', dl_config.get('num_workers', 4))
        pin_memory = dl_config.get('pin_memory', True)
        persistent_workers = dl_config.get('persistent_workers', True) and num_workers > 0
        prefetch_factor = dl_config.get('prefetch_factor', 2)

        # Get gradient accumulation for effective larger batch
        grad_accum = get_gradient_accumulation(model_name, gpu_profile)
        effective_batch = batch_size * grad_accum

        # CRITICAL: Inject GPU profile settings into config for BaseTrainer
        if 'training' not in self.config:
            self.config['training'] = {}
        self.config['training']['gradient_accumulation_steps'] = grad_accum
        # Ensure AMP is enabled
        if 'amp' not in self.config['training']:
            self.config['training']['amp'] = True

        print(f"   GPU Profile: {gpu_profile}")
        print(f"   Model: {model_name}")
        print(f"   Batch size: {batch_size} (effective: {effective_batch} with grad_accum={grad_accum})")
        print(f"   Gradient accumulation: {grad_accum}")
        print(f"   AMP: {self.config['training'].get('amp', False)}")
        print(f"   Num workers: {num_workers}")
        print(f"   Pin memory: {pin_memory}, Persistent workers: {persistent_workers}")
        print(f"   Prefetch factor: {prefetch_factor}")

        # Import correct dataloader function
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

        # CRITICAL: Ensure batch_size is valid
        if not batch_size or batch_size < 1:
            print(f"   [ERROR] Invalid batch_size={batch_size}, using fallback 64")
            batch_size = 64

        # Get drop_last setting (better GPU utilization with consistent batch sizes)
        drop_last = dl_config.get('drop_last', True)

        # Common DataLoader kwargs for optimal performance
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'drop_last': drop_last,  # Important for GPU utilization
        }
        # Remove None values (but batch_size is guaranteed to be valid now)
        loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

        # Candidate loader (train + canary)
        candidate_loader = create_dataloader(
            self.config,
            mode='candidate',
            **loader_kwargs
        )
        print(f"✅ Candidate loader: {len(candidate_loader.dataset)} samples")

        # Independent loader (train only)
        independent_loader = create_dataloader(
            self.config,
            mode='independent',
            **loader_kwargs
        )
        print(f"✅ Independent loader: {len(independent_loader.dataset)} samples")

        # Test loader (optional, for validation)
        try:
            test_loader = create_dataloader(
                self.config,
                split='test',
                shuffle=False,
                **loader_kwargs
            )
            print(f"✅ Test loader: {len(test_loader.dataset)} samples")
        except:
            test_loader = None
            print(f"⚠️ Test loader not available")

        print()

        return candidate_loader, independent_loader, test_loader
    
    def train_candidate(self,
                       candidate_model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train candidate model
        
        Args:
            candidate_model: Candidate model
            train_loader: Training data (train + canary)
            val_loader: Validation data
        
        Returns:
            Training history
        """
        print("="*80)
        print("TRAINING CANDIDATE MODEL")
        print("="*80)
        print("This model sees BOTH train and canary samples")
        print()
        
        trainer = BaseTrainer(
            model=candidate_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            experiment_name=f"{self.experiment_name}_candidate",
            output_dir=self.candidate_dir
        )
        
        history = trainer.train()
        
        print()
        print(f"✅ Candidate training complete")
        print(f"   Best accuracy: {trainer.best_metric:.2f}%")
        print(f"   Best epoch: {trainer.best_epoch + 1}")
        print()
        
        return history
    
    def train_independent(self,
                         independent_model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train independent model
        
        Args:
            independent_model: Independent model
            train_loader: Training data (train only, NO canary)
            val_loader: Validation data
        
        Returns:
            Training history
        """
        print("="*80)
        print("TRAINING INDEPENDENT MODEL")
        print("="*80)
        print("This model sees ONLY train samples (NO canary)")
        print()
        
        trainer = BaseTrainer(
            model=independent_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            experiment_name=f"{self.experiment_name}_independent",
            output_dir=self.independent_dir
        )
        
        history = trainer.train()
        
        print()
        print(f"✅ Independent training complete")
        print(f"   Best accuracy: {trainer.best_metric:.2f}%")
        print(f"   Best epoch: {trainer.best_epoch + 1}")
        print()
        
        return history
    
    def train_both(self) -> Dict[str, Dict]:
        """
        Train both candidate and independent models
        
        Returns:
            Dictionary with both training histories
        """
        print("\n" + "="*80)
        print("DIFFERENTIAL TRAINING")
        print("="*80)
        print()
        print("Training TWO models:")
        print("  1. Candidate:   Sees train + canary")
        print("  2. Independent: Sees train only")
        print()
        print("Memorization = Difference in performance on canary samples")
        print("="*80)
        print()
        
        # Get seed from config
        seed = self.config.get('training', {}).get('seed', 42)

        # CRITICAL: Reset seed before model creation for reproducibility
        _reset_seed(seed)
        print(f"🔒 Random seed set to {seed} for reproducible model initialization")

        # Create models (with identical initialization via deepcopy)
        candidate_model, independent_model = self.create_models()

        # Create data loaders
        candidate_loader, independent_loader, test_loader = self.create_dataloaders()

        # CRITICAL: Reset seed before candidate training
        # This ensures identical random sequences for augmentation, dropout, etc.
        _reset_seed(seed)
        print(f"🔒 Random seed reset to {seed} before candidate training")

        # Train candidate
        candidate_history = self.train_candidate(
            candidate_model,
            candidate_loader,
            test_loader
        )

        # CRITICAL: Reset seed before independent training
        # Both models must experience identical random sequences
        _reset_seed(seed)
        print(f"🔒 Random seed reset to {seed} before independent training")

        # Train independent
        independent_history = self.train_independent(
            independent_model,
            independent_loader,
            test_loader
        )
        
        # Summary
        print("="*80)
        print("DIFFERENTIAL TRAINING COMPLETE")
        print("="*80)
        print()
        print("Candidate Model:")
        print(f"  Training samples: {len(candidate_loader.dataset)}")
        print(f"  Best accuracy: {max(candidate_history['train_acc']):.2f}%")
        print(f"  Checkpoints: {self.candidate_dir / 'checkpoints'}")
        print()
        print("Independent Model:")
        print(f"  Training samples: {len(independent_loader.dataset)}")
        print(f"  Best accuracy: {max(independent_history['train_acc']):.2f}%")
        print(f"  Checkpoints: {self.independent_dir / 'checkpoints'}")
        print()
        print("Next step: Measure memorization on canary samples")
        print("="*80)
        
        return {
            'candidate': candidate_history,
            'independent': independent_history
        }


def train_differential_model(model_name: str,
                            config: Dict,
                            output_dir: Path) -> Dict[str, Dict]:
    """
    Convenience function to train model in differential setup
    
    Args:
        model_name: Model name ('resnet50', 'vit', 'mae', 'medsam')
        config: Complete configuration
        output_dir: Output directory
    
    Returns:
        Training histories for both models
    """
    from ..models import ModelFactory
    
    # Get model config
    model_config = config.get('model_configs', {}).get(model_name, {})

    # Get num_classes from dataset config - MUST be specified, no default
    dataset_config = config.get('dataset_config', {})
    classes_config = dataset_config.get('classes', {})

    if isinstance(classes_config, dict) and 'names' in classes_config:
        num_classes = len(classes_config['names'])
    else:
        # No default - raise error for explicit configuration
        raise ValueError(
            f"Cannot determine num_classes for model '{model_name}'.\n"
            f"Dataset config must include 'classes.names' list.\n"
            f"Found dataset_config.classes = {classes_config}\n"
            f"Please ensure your dataset config (config/datasets/<dataset>.yaml) has:\n"
            f"  classes:\n"
            f"    names:\n"
            f"      - class1\n"
            f"      - class2\n"
            f"      - ..."
        )
    
    # Model factory function
    def create_model():
        return ModelFactory.create(model_name, model_config, num_classes)
    
    # Create differential trainer
    trainer = DifferentialTrainer(
        model_factory_fn=create_model,
        config=config,
        experiment_name=model_name,
        output_dir=output_dir / model_name,
        model_name=model_name  # Pass model_name for GPU profile batch size lookup
    )
    
    # Train both models
    histories = trainer.train_both()
    
    return histories


# Example usage
if __name__ == "__main__":
    print("Differential Trainer")
    print()
    print("Usage:")
    print("  from src.training.differential_trainer import train_differential_model")
    print("  from src.utils.config_loader import load_experiment_config")
    print()
    print("  config = load_experiment_config('baseline')")
    print("  train_differential_model('resnet50', config, Path('results/baseline'))")
    print()
    print("This trains BOTH candidate and independent models automatically!")