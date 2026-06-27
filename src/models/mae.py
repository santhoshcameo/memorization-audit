"""
MAE (Masked Autoencoder) Model
Trained from random initialization (baseline)
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


class MAEModel(BaseModel):
    """
    MAE (Masked Autoencoder) architecture
    
    Trained from RANDOM INITIALIZATION (no pretraining)
    Serves as baseline to validate memorization measurement
    
    Expected to show WEAK memorization (similar to or lower than MedSAM)
    because no pretraining = no prior knowledge to leak
    """
    
    def __init__(self,
                 num_classes: int = 7,
                 config: Optional[Dict] = None):
        """
        Initialize MAE
        
        Args:
            num_classes: Number of output classes
            config: Model configuration
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for MAE. Install with: pip install timm")
        
        super().__init__(num_classes=num_classes, config=config)
        
        # Get config
        arch_config = self.config.get('architecture', {})
        variant = arch_config.get('variant', 'vit_base_patch16')
        
        # MAE is trained from scratch (no pretraining!)
        print(f"Initializing MAE ({variant}) from RANDOM initialization...")
        print("⚠️ This is a baseline model (no pretraining)")
        
        # Use ViT backbone (same as ViT but no pretrained weights)
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,  # From scratch!
            num_classes=num_classes
        )
        
        # Feature dimension
        self.feature_dim = self.backbone.embed_dim  # 768
        
        # Classifier
        self.classifier = self.backbone.head
        
        print(f"✅ MAE initialized (num_classes={num_classes})")
        print(f"   Pretrained: FALSE (training from scratch)")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Purpose: Baseline to validate memorization measurement")
    
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
        Extract CLS token features
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Features (B, 768)
        """
        # Same as ViT
        x = self.backbone.patch_embed(x)
        
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        cls_features = x[:, 0]
        
        return cls_features


def create_mae(num_classes: int = 7) -> MAEModel:
    """
    Create MAE model (from scratch)
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        MAEModel instance
    """
    config = {
        'architecture': {
            'pretrained': False,  # Always False for MAE baseline
            'num_classes': num_classes
        }
    }
    
    return MAEModel(num_classes=num_classes, config=config)


# Example usage
if __name__ == "__main__":
    if TIMM_AVAILABLE:
        print("Testing MAE model...")
        
        # Create model
        model = create_mae(num_classes=7)
        
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
        
        print("\n✅ MAE model working!")
        print("\n⚠️ Remember: MAE is trained from SCRATCH (no pretraining)")
        print("   This serves as a baseline to validate memorization measurement")
    else:
        print("❌ timm not available. Install with: pip install timm")