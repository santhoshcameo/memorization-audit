"""
MedSAM Model - FIXED
Handles both ViT and ResNet fallback encoders properly
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path
from .base_model import BaseModel


class MedSAMModel(BaseModel):
    """
    MedSAM architecture - FIXED
    
    Based on SAM ViT-B encoder pretrained on 1.57M medical images
    
    FIXES:
    - Handles both ViT and ResNet fallback encoders
    - Proper dimension handling for neck
    - Better error messages
    """
    
    def __init__(self,
                 num_classes: int = 7,
                 config: Optional[Dict] = None):
        """
        Initialize MedSAM
        
        Args:
            num_classes: Number of output classes
            config: Model configuration
        """
        super().__init__(num_classes=num_classes, config=config)
        
        # Get config
        arch_config = self.config.get('architecture', {})
        pretrained_path = arch_config.get('pretrained_source', 'medsam_vit_b.pth')

        # Resolve relative path to project root
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            pretrained_path = project_root / pretrained_path

        print("Initializing MedSAM...")

        # Build encoder
        self.encoder_type = self._build_encoder()

        # Feature dimension after neck
        self.feature_dim = 256

        # Try to load pretrained weights
        if pretrained_path.exists():
            self._load_medsam_checkpoint(str(pretrained_path))
            print(f"✅ Loaded MedSAM weights from: {pretrained_path}")
        else:
            print(f"⚠️ MedSAM checkpoint not found: {pretrained_path}")
            print(f"   Using random initialization (not recommended for publication!)")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"✅ MedSAM initialized (num_classes={num_classes})")
        print(f"   Encoder type: {self.encoder_type}")
        print(f"   Feature dim: {self.feature_dim}")
    
    def _build_encoder(self):
        """Build encoder with fallback support"""
        try:
            import timm
            
            # Try ViT encoder (preferred) - REMOVED global_pool parameter
            encoder = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0  # This removes the classification head
            )
            
            # Add neck for ViT (768 -> 256)
            self.encoder = encoder
            self.neck = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
            )
            
            print("✅ Using ViT-B encoder")
            return 'ViT-B'
            
        except Exception as e:
            print(f"⚠️ Could not load ViT encoder: {e}")
            print("   Using ResNet50 fallback")
            
            # Fallback: ResNet50
            import torchvision.models as models
            resnet = models.resnet50(pretrained=False)
            
            # Remove final FC
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            
            # Neck for ResNet (2048 -> 256)
            self.neck = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            
            return 'ResNet50'
    
    def _load_medsam_checkpoint(self, checkpoint_path: str):
        """
        Load MedSAM/SAM checkpoint - handles multiple checkpoint formats

        Supported checkpoint structures:
        1. MedSAM format: {'model': {'image_encoder': {...}, ...}}
        2. SAM format: {'image_encoder.xxx': ..., 'mask_decoder.xxx': ...}
        3. Direct state dict: {'blocks.0.xxx': ..., 'patch_embed.xxx': ...}
        """
        try:
            # Load checkpoint - handle older PyTorch versions that don't support weights_only
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Older PyTorch doesn't support weights_only parameter
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            loaded = False

            # Format 1: MedSAM format with nested 'model' key
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model_dict = checkpoint['model']
                if isinstance(model_dict, dict) and 'image_encoder' in model_dict:
                    encoder_weights = model_dict['image_encoder']
                    self._load_encoder_weights(encoder_weights)
                    print("✅ Loaded MedSAM image encoder weights (nested format)")
                    loaded = True
                elif isinstance(model_dict, dict):
                    # Try to extract image_encoder weights from model dict
                    encoder_weights = {k.replace('image_encoder.', ''): v
                                      for k, v in model_dict.items()
                                      if k.startswith('image_encoder.')}
                    if encoder_weights:
                        self._load_encoder_weights(encoder_weights)
                        print("✅ Loaded MedSAM image encoder weights (flat model format)")
                        loaded = True

            # Format 2: SAM format with 'image_encoder.' prefix at root level
            if not loaded and isinstance(checkpoint, dict):
                encoder_weights = {k.replace('image_encoder.', ''): v
                                  for k, v in checkpoint.items()
                                  if k.startswith('image_encoder.')}
                if encoder_weights:
                    self._load_encoder_weights(encoder_weights)
                    print("✅ Loaded SAM image encoder weights (root level)")
                    loaded = True

            # Format 3: Direct ViT state dict (e.g., from timm)
            if not loaded and isinstance(checkpoint, dict):
                # Check if it looks like a ViT state dict
                vit_keys = ['blocks.0.attn.qkv.weight', 'patch_embed.proj.weight', 'cls_token']
                has_vit_keys = any(any(vk in k for vk in vit_keys) for k in checkpoint.keys())
                if has_vit_keys:
                    self._load_encoder_weights(checkpoint)
                    print("✅ Loaded ViT weights directly")
                    loaded = True

            if not loaded:
                # Print diagnostic info
                if isinstance(checkpoint, dict):
                    print(f"⚠️ Could not match checkpoint format. Keys present: {list(checkpoint.keys())[:10]}...")
                else:
                    print(f"⚠️ Unexpected checkpoint type: {type(checkpoint)}")
                print("   Using random initialization for encoder")

        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("   Proceeding with random initialization")
    
    def _load_encoder_weights(self, state_dict):
        """Load weights into encoder with key mapping"""
        try:
            our_dict = self.encoder.state_dict()
            
            filtered_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('image_encoder.', '')
                
                if new_k in our_dict and our_dict[new_k].shape == v.shape:
                    filtered_dict[new_k] = v
            
            self.encoder.load_state_dict(filtered_dict, strict=False)
            print(f"✅ Loaded {len(filtered_dict)}/{len(our_dict)} encoder weights")
        
        except Exception as e:
            print(f"⚠️ Could not load encoder weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Logits (B, num_classes)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from encoder
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Features (B, 256)
        """
        # Pass through encoder
        # IMPORTANT: Use self.encoder(x) (full __call__ path), NOT
        # self.encoder.forward_features(x). Opacus per-sample gradient
        # hooks require the module __call__ path to fire properly.
        if self.encoder_type == 'ViT-B':
            # With num_classes=0, encoder(x) returns (B, 768) after
            # CLS token selection via forward_features + forward_head.
            features = self.encoder(x)  # (B, 768)

        else:  # ResNet50
            features = self.encoder(x)  # (B, 2048, 1, 1)
            features = features.flatten(1)  # (B, 2048)
        
        # Apply neck - NOW features are (B, 768) or (B, 2048)
        features = self.neck(features)  # (B, 256)
        
        return features


def create_medsam(num_classes: int = 7,
                 checkpoint_path: Optional[str] = 'medsam_vit_b.pth') -> MedSAMModel:
    """
    Create MedSAM model
    
    Args:
        num_classes: Number of output classes
        checkpoint_path: Path to medsam_vit_b.pth
    
    Returns:
        MedSAMModel instance
    """
    config = {
        'architecture': {
            'pretrained': checkpoint_path is not None,
            'pretrained_source': checkpoint_path,
            'num_classes': num_classes
        }
    }
    
    return MedSAMModel(num_classes=num_classes, config=config)


# Example usage
if __name__ == "__main__":
    print("Testing MedSAM model...")
    
    # Create model
    model = create_medsam(num_classes=7)
    
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
    
    print("\n✅ MedSAM model working!")