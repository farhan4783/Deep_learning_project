"""
Audio-Visual Synchronization Detector

This module implements audio-visual sync detection for deepfake videos.
Analyzes lip movements and audio to detect mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class LipFeatureExtractor(nn.Module):
    """
    Extract visual features from lip/mouth region
    """
    
    def __init__(self, feature_dim: int = 512):
        """
        Initialize lip feature extractor
        
        Args:
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        # 3D CNN for spatiotemporal features
        self.conv3d_layers = nn.Sequential(
            # Input: (B, 3, T, H, W) - T temporal frames
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract lip features
        
        Args:
            x: Input tensor (B, 3, T, H, W) - lip region frames
            
        Returns:
            Features (B, feature_dim)
        """
        features = self.conv3d_layers(x)
        features = features.view(features.size(0), -1)
        return features


class AudioFeatureExtractor(nn.Module):
    """
    Extract audio features from mel-spectrogram
    """
    
    def __init__(self, feature_dim: int = 512):
        """
        Initialize audio feature extractor
        
        Args:
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        # 2D CNN for mel-spectrogram
        self.conv_layers = nn.Sequential(
            # Input: (B, 1, T, F) - time x frequency
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features
        
        Args:
            x: Input mel-spectrogram (B, 1, T, F)
            
        Returns:
            Features (B, feature_dim)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return features


class AudioVisualSyncDetector(nn.Module):
    """
    Audio-Visual Synchronization Detector
    
    Detects mismatches between lip movements and audio to identify deepfakes
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize AV sync detector
        
        Args:
            feature_dim: Dimension of extracted features
            fusion_dim: Dimension of fused features
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lip_extractor = LipFeatureExtractor(feature_dim)
        self.audio_extractor = AudioFeatureExtractor(feature_dim)
        
        # Cross-modal attention
        self.audio_to_visual_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.visual_to_audio_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Sync score predictor
        self.sync_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + 1, 128),  # +1 for sync score
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(
        self,
        lip_frames: torch.Tensor,
        audio_spec: torch.Tensor,
        return_sync_score: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            lip_frames: Lip region frames (B, 3, T, H, W)
            audio_spec: Audio mel-spectrogram (B, 1, T, F)
            return_sync_score: Whether to return sync score
            
        Returns:
            Tuple of (logits, sync_score)
        """
        # Extract features
        visual_features = self.lip_extractor(lip_frames)  # (B, feature_dim)
        audio_features = self.audio_extractor(audio_spec)  # (B, feature_dim)
        
        # Reshape for attention (add sequence dimension)
        visual_features_seq = visual_features.unsqueeze(1)  # (B, 1, feature_dim)
        audio_features_seq = audio_features.unsqueeze(1)  # (B, 1, feature_dim)
        
        # Cross-modal attention
        audio_attended, _ = self.audio_to_visual_attn(
            query=audio_features_seq,
            key=visual_features_seq,
            value=visual_features_seq
        )
        
        visual_attended, _ = self.visual_to_audio_attn(
            query=visual_features_seq,
            key=audio_features_seq,
            value=audio_features_seq
        )
        
        # Squeeze back
        audio_attended = audio_attended.squeeze(1)
        visual_attended = visual_attended.squeeze(1)
        
        # Concatenate all features
        fused_features = torch.cat([
            visual_features,
            audio_features,
            visual_attended,
            audio_attended
        ], dim=1)
        
        # Fusion
        fused = self.fusion(fused_features)
        
        # Predict sync score
        sync_score = self.sync_predictor(fused)
        
        # Classification with sync score
        classifier_input = torch.cat([fused, sync_score], dim=1)
        logits = self.classifier(classifier_input)
        
        if return_sync_score:
            return logits, sync_score
        else:
            return logits, None


class SyncNetLoss(nn.Module):
    """
    Custom loss function for sync detection
    Combines classification loss and sync score loss
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize loss
        
        Args:
            alpha: Weight for sync score loss
        """
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        sync_scores: torch.Tensor,
        labels: torch.Tensor,
        sync_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            logits: Classification logits (B, num_classes)
            sync_scores: Predicted sync scores (B, 1)
            labels: Classification labels (B,)
            sync_labels: Ground truth sync labels (B, 1)
            
        Returns:
            Combined loss
        """
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Sync score loss
        sync_loss = self.bce_loss(sync_scores, sync_labels)
        
        # Combined loss
        total_loss = (1 - self.alpha) * cls_loss + self.alpha * sync_loss
        
        return total_loss


def create_av_sync_detector(**kwargs) -> AudioVisualSyncDetector:
    """
    Factory function to create AV sync detector
    
    Args:
        **kwargs: Arguments for detector
        
    Returns:
        AV sync detector model
    """
    return AudioVisualSyncDetector(**kwargs)


if __name__ == "__main__":
    # Test the model
    model = create_av_sync_detector()
    
    # Test inputs
    lip_frames = torch.randn(2, 3, 16, 112, 112)  # 16 frames of lip region
    audio_spec = torch.randn(2, 1, 100, 80)  # Mel-spectrogram
    
    # Forward pass
    logits, sync_score = model(lip_frames, audio_spec, return_sync_score=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Sync score shape: {sync_score.shape}")
    print(f"Sync scores: {sync_score.squeeze().detach().numpy()}")
