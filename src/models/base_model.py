"""
Base Model Class
Abstract base class for all models
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models
    
    All model implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, 
                 num_classes: int = 7,
                 pretrained: bool = False,
                 config: Optional[Dict] = None):
        """
        Initialize base model
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            config: Model configuration dictionary
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.config = config or {}
        
        # To be set by subclasses
        self.backbone = None
        self.classifier = None
        self.feature_dim = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output logits (B, num_classes)
        """
        pass
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for memorization measurement
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Features (B, feature_dim)
        """
        pass
    
    def get_feature_dim(self) -> int:
        """Get dimension of extracted features"""
        return self.feature_dim
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✅ Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("✅ Backbone unfrozen")
    
    def freeze_layers(self, layer_names: list):
        """
        Freeze specific layers
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        print(f"✅ Frozen layers: {layer_names}")
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters
        
        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }
    
    def print_model_info(self):
        """Print model information"""
        params = self.count_parameters()
        
        print("="*60)
        print(f"Model: {self.__class__.__name__}")
        print("="*60)
        print(f"Number of classes: {self.num_classes}")
        print(f"Pretrained: {self.pretrained}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"\nParameters:")
        print(f"  Total:     {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Frozen:    {params['frozen']:,}")
        print("="*60)
    
    def save_checkpoint(self, path: Path, epoch: int, optimizer=None, 
                       metrics: Optional[Dict] = None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state (optional)
            metrics: Training metrics (optional)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'num_classes': self.num_classes,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path,
                       load_optimizer: bool = False,
                       optimizer=None) -> Dict:
        """
        Load model checkpoint.

        Handles:
        - Opacus _module. prefix (stripped automatically)
        - GroupNorm vs BatchNorm mismatch (strict=False fallback)

        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer to load state into

        Returns:
            Checkpoint dictionary
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # Handle Opacus _module. prefix
        if state_dict and any(k.startswith('_module.') for k in state_dict.keys()):
            print("  Stripping Opacus _module. prefix from checkpoint keys")
            state_dict = {k.replace('_module.', '', 1): v for k, v in state_dict.items()}

        # Try strict load first, fall back to non-strict for architecture mismatches
        # (e.g. checkpoint has GroupNorm but model has BatchNorm)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            if 'Missing key' in str(e) or 'Unexpected key' in str(e):
                print(f"  Strict load failed, using non-strict load: {str(e)[:120]}...")
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
                if unexpected:
                    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            else:
                raise

        # Load optimizer state
        if load_optimizer and optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✅ Checkpoint loaded: {path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")

        return checkpoint
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    @torch.no_grad()
    def predict_batch(self, dataloader, device='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict on entire dataloader
        
        Args:
            dataloader: DataLoader
            device: Device to use
        
        Returns:
            Tuple of (all_predictions, all_probabilities)
        """
        self.eval()
        self.to(device)
        
        all_preds = []
        all_probs = []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            preds, probs = self.predict(x)
            
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        
        return all_preds, all_probs
    
    def get_learning_rate_groups(self, base_lr: float) -> list:
        """
        Get parameter groups for layer-wise learning rates
        
        Args:
            base_lr: Base learning rate
        
        Returns:
            List of parameter groups
        """
        # Default: single learning rate for all parameters
        return [{'params': self.parameters(), 'lr': base_lr}]


class ModelFactory:
    """
    Factory for creating models from configuration
    """
    
    @staticmethod
    def create(model_name: str, config: Dict, num_classes: int = 7) -> BaseModel:
        """
        Create model from configuration
        
        Args:
            model_name: Model name ('resnet50', 'vit', 'mae', 'medsam')
            config: Model configuration
            num_classes: Number of classes
        
        Returns:
            Model instance
        """
        if model_name == 'resnet50':
            from .resnet import ResNet50Model
            return ResNet50Model(num_classes=num_classes, config=config)
        
        elif model_name == 'vit':
            from .vit import ViTModel
            return ViTModel(num_classes=num_classes, config=config)
        
        elif model_name == 'mae':
            from .mae import MAEModel
            return MAEModel(num_classes=num_classes, config=config)
        
        elif model_name == 'medsam':
            from .medsam import MedSAMModel
            return MedSAMModel(num_classes=num_classes, config=config)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")


# Example usage
if __name__ == "__main__":
    print("Base model class defined successfully")
    print("\nAvailable through ModelFactory:")
    print("  - resnet50: ResNet50 pretrained on ImageNet-1K")
    print("  - vit: ViT-Base pretrained on ImageNet-21K")
    print("  - mae: MAE trained from scratch")
    print("  - medsam: MedSAM pretrained on medical images")