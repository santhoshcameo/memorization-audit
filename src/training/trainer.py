"""
Base Trainer Class
Handles training loop for all models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import time

from ..utils.logging import ExperimentLogger
from ..utils.reproducibility import set_seed


class BaseTrainer:
    """
    Base trainer for all models
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    - Early stopping
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[Dict] = None,
                 experiment_name: str = "experiment",
                 output_dir: Path = Path("results")):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            experiment_name: Name of experiment
            output_dir: Directory to save outputs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        
        # Setup logger
        self.logger = ExperimentLogger(
            experiment_name=experiment_name,
            output_dir=self.output_dir / "logs",
            config=config
        )
        
        # Get training config
        train_config = self.config.get('training', {})
        
        # Device
        self.device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training hyperparameters
        self.epochs = train_config.get('epochs', 50)
        self.seed = train_config.get('seed', 42)
        
        # Optimizer
        self.optimizer = self._setup_optimizer(train_config)
        
        # Scheduler
        self.scheduler = self._setup_scheduler(train_config)
        
        # Loss function
        self.criterion = self._setup_criterion(train_config)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Gradient accumulation
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        
        # Mixed precision
        self.use_amp = train_config.get('amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Gradient clipping
        self.grad_clip = train_config.get('grad_clip', {})
        
        # Early stopping
        self.early_stopping = train_config.get('early_stopping', {})
        self.early_stop_counter = 0

        # Memory management
        self._setup_memory_management()

        self.logger.logger.info(f"Trainer initialized")
        self.logger.logger.info(f"  Device: {self.device}")
        self.logger.logger.info(f"  Epochs: {self.epochs}")
        self.logger.logger.info(f"  Train batches: {len(train_loader)}")
        if val_loader:
            self.logger.logger.info(f"  Val batches: {len(val_loader)}")
    
    def _setup_optimizer(self, train_config: Dict) -> torch.optim.Optimizer:
        """Setup optimizer"""
        opt_config = train_config.get('optimizer', {})
        opt_type = opt_config.get('type', 'AdamW')
        opt_params = opt_config.get('params', {})
        
        # Get learning rate
        lr = opt_params.get('lr', 1e-4)
        
        # Get parameter groups (for layer-wise LR)
        if hasattr(self.model, 'get_learning_rate_groups'):
            param_groups = self.model.get_learning_rate_groups(lr)
            self.logger.logger.info(f"Using layer-wise learning rates ({len(param_groups)} groups)")
        else:
            param_groups = self.model.parameters()
        
        # Create optimizer
        if opt_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=opt_params.get('weight_decay', 0.01),
                betas=opt_params.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'Adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=opt_params.get('weight_decay', 0.0)
            )
        elif opt_type == 'SGD':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=opt_params.get('momentum', 0.9),
                weight_decay=opt_params.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        self.logger.logger.info(f"Optimizer: {opt_type} (lr={lr})")
        return optimizer
    
    def _setup_scheduler(self, train_config: Dict) -> Optional[Any]:
        """Setup learning rate scheduler"""
        sched_config = train_config.get('scheduler', {})
        sched_type = sched_config.get('type', None)
        
        if sched_type is None:
            return None
        
        sched_params = sched_config.get('params', {})
        
        if sched_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_params.get('T_max', self.epochs),
                eta_min=sched_params.get('eta_min', 1e-6)
            )
        elif sched_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_params.get('step_size', 30),
                gamma=sched_params.get('gamma', 0.1)
            )
        elif sched_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_params.get('factor', 0.5),
                patience=sched_params.get('patience', 5)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")
        
        self.logger.logger.info(f"Scheduler: {sched_type}")
        return scheduler
    
    def _setup_memory_management(self):
        """Setup GPU memory management - OPTIMIZED for A100"""
        try:
            from ..utils.gpu_config import (
                get_memory_config, get_gpu_profile_name, clear_gpu_cache,
                setup_cuda_optimizations
            )
            self.gpu_profile = get_gpu_profile_name()
            mem_config = get_memory_config(self.gpu_profile)

            # CRITICAL: empty_cache_freq = 0 means DISABLED (for A100)
            # This was a major bottleneck - cache clearing is extremely slow
            self.empty_cache_freq = mem_config.get('empty_cache_freq', 0)
            self.non_blocking = mem_config.get('non_blocking', True)
            self.clear_gpu_cache = clear_gpu_cache

            # Ensure CUDA optimizations are applied
            setup_cuda_optimizations()

            if self.empty_cache_freq > 0:
                self.logger.logger.info(f"Memory management: clear cache every {self.empty_cache_freq} batches")
            else:
                self.logger.logger.info(f"Memory management: cache clearing DISABLED (optimal for {self.gpu_profile})")
            self.logger.logger.info(f"Non-blocking transfers: {self.non_blocking}")
        except Exception as e:
            self.empty_cache_freq = 0
            self.non_blocking = True
            self.clear_gpu_cache = lambda: None
            self.gpu_profile = 'unknown'

    def _maybe_clear_cache(self, batch_idx: int):
        """Clear GPU cache periodically to prevent fragmentation (DISABLED for A100)"""
        # Only clear if empty_cache_freq > 0 (disabled for A100)
        if self.empty_cache_freq > 0 and batch_idx > 0 and batch_idx % self.empty_cache_freq == 0:
            self.clear_gpu_cache()

    def _setup_criterion(self, train_config: Dict) -> nn.Module:
        """Setup loss function"""
        loss_type = train_config.get('loss', 'CrossEntropyLoss')
        
        if loss_type == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
        
        return criterion
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            # Transfer to GPU with non_blocking for async performance
            # This allows CPU to prepare next batch while GPU processes current
            images = images.to(self.device, non_blocking=getattr(self, 'non_blocking', True))
            labels = labels.to(self.device, non_blocking=getattr(self, 'non_blocking', True))
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.grad_accum_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps
                
                # Backward pass
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.grad_clip.get('enabled', False):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip.get('max_norm', 1.0)
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * self.grad_accum_steps * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

            # Periodic cache clearing for memory management
            self._maybe_clear_cache(batch_idx)
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Unpack batch
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            # Transfer to GPU with non_blocking for async performance
            images = images.to(self.device, non_blocking=getattr(self, 'non_blocking', True))
            labels = labels.to(self.device, non_blocking=getattr(self, 'non_blocking', True))
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def train(self) -> Dict[str, list]:
        """Full training loop"""
        self.logger.log_event("training_started")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.logger.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy']
            }, step=epoch)
            
            if val_metrics:
                self.logger.log_metrics({
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy']
                }, step=epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('accuracy', 0))
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            current_metric = val_metrics.get('accuracy', train_metrics['accuracy'])
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self._check_early_stopping(current_metric):
                self.logger.logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print epoch summary
            self.logger.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                + (f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.2f}%" if val_metrics else "")
            )
        
        # Training complete
        elapsed = time.time() - start_time
        self.logger.log_event("training_completed", {
            'duration_seconds': elapsed,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric
        })
        
        self.logger.finalize("completed")
        
        return self.history
    
    def _check_early_stopping(self, current_metric: float) -> bool:
        """Check if should stop early"""
        if not self.early_stopping.get('enabled', False):
            return False
        
        patience = self.early_stopping.get('patience', 10)
        
        if current_metric > self.best_metric:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        return self.early_stop_counter >= patience
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.experiment_name}_best.pth" if is_best else f"{self.experiment_name}_last.pth"
        filepath = checkpoint_dir / filename
        
        self.model.save_checkpoint(
            path=filepath,
            epoch=self.current_epoch,
            optimizer=self.optimizer,
            metrics={
                'best_metric': self.best_metric,
                'current_metric': self.history['val_acc'][-1] if self.history['val_acc'] else 0
            }
        )


# Example usage
if __name__ == "__main__":
    print("Base trainer defined")
    print("Use DifferentialTrainer for candidate/independent training")