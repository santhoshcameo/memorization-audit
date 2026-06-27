"""
Vision Transformer (ViT) Model
ViT-Base pretrained on ImageNet-21K (14M images)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from .base_model import BaseModel

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️ timm not installed. Install with: pip install timm")


class ViTModel(BaseModel):
    """
    Vision Transformer (ViT) architecture
    
    Pretrained on ImageNet-21K (14M images, 21K classes)
    Expected to show LOWER memorization than ResNet50 due to:
    - Larger pretraining dataset (14M vs 1.28M)
    - Better regularization (attention, drop_path)
    """
    
    def __init__(self,
                 num_classes: int = 7,
                 config: Optional[Dict] = None):
        """
        Initialize ViT
        
        Args:
            num_classes: Number of output classes
            config: Model configuration
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT. Install with: pip install timm")
        
        super().__init__(num_classes=num_classes, config=config)
        
        # Get config
        arch_config = self.config.get('architecture', {})
        pretrained = arch_config.get('pretrained', True)
        variant = arch_config.get('variant', 'vit_base_patch16_224')
        
        # Load ViT from timm
        if pretrained:
            pretrained_dataset = arch_config.get('pretrained_dataset', 'imagenet21k')
            
            # Try multiple model names (timm naming changes over versions)
            if pretrained_dataset == 'imagenet21k':
                model_names = [
                    'vit_base_patch16_224.augreg_in21k',  # Latest
                    'vit_base_patch16_224_in21k',          # Older
                    'vit_base_patch16_224',                # Fallback
                ]
                print(f"Loading {variant} pretrained on ImageNet-21K (14M images)...")
            else:
                model_names = [
                    'vit_base_patch16_224.augreg_in1k',    # Latest
                    'vit_base_patch16_224',                # Fallback
                ]
                print(f"Loading {variant} pretrained on ImageNet-1K...")
            
            # Try each model name until one works
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(
                        model_name,
                        pretrained=True,
                        num_classes=num_classes
                    )
                    print(f"  Successfully loaded: {model_name}")
                    break
                except RuntimeError:
                    continue
            
            if self.backbone is None:
                raise RuntimeError(f"Could not load any ViT model. Tried: {model_names}")
                
        else:
            print(f"Initializing {variant} from scratch...")
            self.backbone = timm.create_model(
                variant,
                pretrained=False,
                num_classes=num_classes
            )
        
        # Feature dimension (CLS token embedding)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-Base
        
        # Classifier is the head
        self.classifier = self.backbone.head
        
        print(f"✅ ViT initialized (num_classes={num_classes})")
        print(f"   Variant: {variant}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Feature dim: {self.feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features (before classification head)
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Features (B, 768)
        """
        # Forward through patch embedding
        x = self.backbone.patch_embed(x)
        
        # Add CLS token
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Forward through transformer blocks
        x = self.backbone.blocks(x)
        
        # Apply final norm
        x = self.backbone.norm(x)
        
        # Extract CLS token (first token)
        cls_features = x[:, 0]
        
        return cls_features
    
    def get_learning_rate_groups(self, base_lr: float) -> list:
        """
        Get parameter groups for layer-wise learning rate decay
        
        ViT benefits from layer-wise LR decay:
        - Early blocks: smaller LR
        - Later blocks: larger LR
        - Head: full LR
        
        Args:
            base_lr: Base learning rate
        
        Returns:
            List of parameter groups
        """
        fine_tuning = self.config.get('fine_tuning', {})
        layer_decay_config = fine_tuning.get('layer_decay', {})
        
        if not layer_decay_config.get('enabled', False):
            return super().get_learning_rate_groups(base_lr)
        
        decay_rate = layer_decay_config.get('decay_rate', 0.65)
        num_layers = len(self.backbone.blocks)
        
        param_groups = []
        
        # Patch embedding
        param_groups.append({
            'params': self.backbone.patch_embed.parameters(),
            'lr': base_lr * (decay_rate ** num_layers)
        })
        
        # Transformer blocks (with decay)
        for i, block in enumerate(self.backbone.blocks):
            layer_idx = i + 1
            lr_scale = decay_rate ** (num_layers - layer_idx)
            param_groups.append({
                'params': block.parameters(),
                'lr': base_lr * lr_scale
            })
        
        # Norm and head (full LR)
        param_groups.append({
            'params': list(self.backbone.norm.parameters()) +
                     list(self.backbone.head.parameters()),
            'lr': base_lr
        })
        
        return param_groups
    
    def enable_stochastic_depth(self, drop_path_rate: float = 0.1):
        """
        Enable stochastic depth (DropPath) for regularization
        
        Args:
            drop_path_rate: Drop path rate
        """
        for block in self.backbone.blocks:
            if hasattr(block, 'drop_path'):
                block.drop_path.drop_prob = drop_path_rate
        
        print(f"✅ Stochastic depth enabled (drop_path={drop_path_rate})")


def create_vit(num_classes: int = 7,
              pretrained: bool = True,
              pretrained_dataset: str = 'imagenet21k',
              freeze_backbone: bool = False) -> ViTModel:
    """
    Create ViT model
    
    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        pretrained_dataset: 'imagenet21k' or 'imagenet1k'
        freeze_backbone: Freeze backbone (only train head)
    
    Returns:
        ViTModel instance
    """
    config = {
        'architecture': {
            'pretrained': pretrained,
            'pretrained_dataset': pretrained_dataset,
            'num_classes': num_classes
        }
    }
    
    model = ViTModel(num_classes=num_classes, config=config)
    
    if freeze_backbone:
        # Freeze all except head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("✅ Backbone frozen (linear probe mode)")
    
    return model


# Example usage
if __name__ == "__main__":
    if TIMM_AVAILABLE:
        print("Testing ViT model...")
        
        # Create model
        model = create_vit(num_classes=7, pretrained=False)
        
        # Print info
        model.print_model_info()
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        
        print("\nTesting forward pass:")
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
        
        print("\nTesting feature extraction:")
        with torch.no_grad():
            features = model.extract_features(dummy_input)
            print(f"  Features shape: {features.shape}")
        
        print("\n✅ ViT model working!")
    else:
        print("❌ timm not available. Install with: pip install timm")