"""
ResNet50 Model
ResNet50 pretrained on ImageNet-1K (1.28M images)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict
from .base_model import BaseModel


class ResNet50Model(BaseModel):
    """
    ResNet50 architecture
    
    Pretrained on ImageNet-1K (1.28M images, 1000 classes)
    Expected to show HIGH memorization due to limited pretraining
    """
    
    def __init__(self,
                 num_classes: int = 7,
                 config: Optional[Dict] = None):
        """
        Initialize ResNet50
        
        Args:
            num_classes: Number of output classes
            config: Model configuration
        """
        super().__init__(num_classes=num_classes, config=config)
        
        # Get config
        arch_config = self.config.get('architecture', {})
        pretrained = arch_config.get('pretrained', True)
        
        # Load ResNet50
        if pretrained:
            print("Loading ResNet50 pretrained on ImageNet-1K...")
            try:
                # New API (torchvision >= 0.13)
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = models.resnet50(weights=weights)
            except AttributeError:
                # Old API (torchvision < 0.13)
                print("  (Using older torchvision API)")
                self.backbone = models.resnet50(pretrained=True)
        else:
            print("Initializing ResNet50 from scratch...")
            try:
                self.backbone = models.resnet50(weights=None)
            except TypeError:
                self.backbone = models.resnet50(pretrained=False)
        
        # Feature dimension (before final FC)
        self.feature_dim = self.backbone.fc.in_features  # 2048
        
        # Replace final FC layer
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Store classifier reference
        self.classifier = self.backbone.fc
        
        print(f"✅ ResNet50 initialized (num_classes={num_classes})")
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
        Extract features from layer4 (before final FC)
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Features (B, 2048)
        """
        # Forward through all layers except FC
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_learning_rate_groups(self, base_lr: float) -> list:
        """
        Get parameter groups for layer-wise learning rates
        
        Can use different LRs for:
        - Early layers (conv1, layer1, layer2)
        - Middle layers (layer3, layer4)
        - Classifier (fc)
        
        Args:
            base_lr: Base learning rate
        
        Returns:
            List of parameter groups
        """
        # Check if layer-wise LR is enabled
        fine_tuning = self.config.get('fine_tuning', {})
        layer_wise_lr = fine_tuning.get('layer_wise_lr', {})
        
        if not layer_wise_lr.get('enabled', False):
            # Single learning rate for all
            return super().get_learning_rate_groups(base_lr)
        
        # Layer-wise learning rates
        backbone_scale = layer_wise_lr.get('backbone_lr_scale', 0.1)
        
        param_groups = [
            # Early layers (smaller LR)
            {
                'params': list(self.backbone.conv1.parameters()) +
                         list(self.backbone.bn1.parameters()) +
                         list(self.backbone.layer1.parameters()) +
                         list(self.backbone.layer2.parameters()),
                'lr': base_lr * backbone_scale * 0.1
            },
            # Middle layers
            {
                'params': list(self.backbone.layer3.parameters()) +
                         list(self.backbone.layer4.parameters()),
                'lr': base_lr * backbone_scale
            },
            # Classifier (full LR)
            {
                'params': self.backbone.fc.parameters(),
                'lr': base_lr
            }
        ]
        
        return param_groups


def create_resnet50(num_classes: int = 7,
                   pretrained: bool = True,
                   freeze_backbone: bool = False) -> ResNet50Model:
    """
    Create ResNet50 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Load ImageNet pretrained weights
        freeze_backbone: Freeze backbone (only train classifier)
    
    Returns:
        ResNet50Model instance
    """
    config = {
        'architecture': {
            'pretrained': pretrained,
            'num_classes': num_classes
        }
    }
    
    model = ResNet50Model(num_classes=num_classes, config=config)
    
    if freeze_backbone:
        # Freeze all except FC
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("✅ Backbone frozen (linear probe mode)")
    
    return model


# Example usage
if __name__ == "__main__":
    print("Testing ResNet50 model...")
    
    # Create model
    model = create_resnet50(num_classes=7, pretrained=False)
    
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
    
    print("\n✅ ResNet50 model working!")