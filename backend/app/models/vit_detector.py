"""
Vision Transformer (ViT) Model for Deepfake Detection

This module implements a Vision Transformer architecture for detecting deepfakes.
Uses pre-trained ViT-B/16 from timm library with custom classification head.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional
import numpy as np


class ViTDeepfakeDetector(nn.Module):
    """
    Vision Transformer model for deepfake detection
    
    Features:
    - Pre-trained ViT-B/16 backbone
    - Custom classification head with dropout
    - Attention map extraction for explainability
    - Support for fine-tuning strategies
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Initialize ViT detector
        
        Args:
            model_name: Name of the ViT model from timm
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate in classification head
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        # Load pre-trained ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        def hook_fn(module, input, output):
            # Store attention weights from each transformer block
            if hasattr(module, 'attn'):
                self.attention_weights.append(output)
        
        # Register hooks on transformer blocks
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks:
                block.register_forward_hook(hook_fn)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, num_classes)
        """
        # Clear previous attention weights
        self.attention_weights = []
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_attention_maps(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get attention maps from specified layer
        
        Args:
            layer_idx: Index of transformer layer (-1 for last layer)
            
        Returns:
            Attention maps tensor or None
        """
        if not self.attention_weights:
            return None
        
        # Get attention from specified layer
        if layer_idx < 0:
            layer_idx = len(self.attention_weights) + layer_idx
        
        if 0 <= layer_idx < len(self.attention_weights):
            return self.attention_weights[layer_idx]
        
        return None
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without classification
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N transformer blocks
            if hasattr(self.backbone, 'blocks'):
                blocks = list(self.backbone.blocks)
                for block in blocks[-num_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True


class ViTDetectorWithUncertainty(ViTDeepfakeDetector):
    """
    ViT Detector with Monte Carlo Dropout for uncertainty estimation
    """
    
    def __init__(self, *args, mc_samples: int = 10, **kwargs):
        """
        Initialize ViT detector with uncertainty estimation
        
        Args:
            mc_samples: Number of Monte Carlo samples for uncertainty
            *args, **kwargs: Arguments for base ViTDeepfakeDetector
        """
        super().__init__(*args, **kwargs)
        self.mc_samples = mc_samples
    
    def enable_dropout(self):
        """Enable dropout during inference for MC Dropout"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (mean_logits, std_logits, epistemic_uncertainty)
        """
        self.eval()
        self.enable_dropout()
        
        # Collect predictions from multiple forward passes
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.forward(x)
                predictions.append(logits)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (mc_samples, B, num_classes)
        
        # Calculate mean and standard deviation
        mean_logits = predictions.mean(dim=0)
        std_logits = predictions.std(dim=0)
        
        # Calculate epistemic uncertainty (predictive entropy)
        probs = torch.softmax(predictions, dim=-1)
        mean_probs = probs.mean(dim=0)
        epistemic_uncertainty = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10),
            dim=-1
        )
        
        return mean_logits, std_logits, epistemic_uncertainty


def create_vit_detector(
    model_type: str = 'base',
    pretrained: bool = True,
    uncertainty: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create ViT detector
    
    Args:
        model_type: Type of ViT model ('base', 'large', 'huge')
        pretrained: Whether to use pretrained weights
        uncertainty: Whether to use uncertainty estimation
        **kwargs: Additional arguments for detector
        
    Returns:
        ViT detector model
    """
    model_names = {
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224',
        'huge': 'vit_huge_patch14_224'
    }
    
    model_name = model_names.get(model_type, 'vit_base_patch16_224')
    
    if uncertainty:
        return ViTDetectorWithUncertainty(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )
    else:
        return ViTDeepfakeDetector(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )


if __name__ == "__main__":
    # Test the model
    model = create_vit_detector(model_type='base', uncertainty=True)
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    # Standard prediction
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # Prediction with uncertainty
    mean_logits, std_logits, uncertainty = model.predict_with_uncertainty(x)
    print(f"Mean logits: {mean_logits.shape}")
    print(f"Std logits: {std_logits.shape}")
    print(f"Uncertainty: {uncertainty.shape}")
    
    # Get attention maps
    attn_maps = model.get_attention_maps(layer_idx=-1)
    if attn_maps is not None:
        print(f"Attention maps shape: {attn_maps.shape}")
