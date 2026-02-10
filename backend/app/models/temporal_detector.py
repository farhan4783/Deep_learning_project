"""
Temporal Inconsistency Detector for Video Deepfakes

This module implements temporal analysis for detecting deepfakes in videos.
Uses LSTM/Transformer to analyze frame sequences and optical flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
import cv2


class OpticalFlowExtractor:
    """
    Extract optical flow features from video frames
    """
    
    def __init__(self, method: str = 'farneback'):
        """
        Initialize optical flow extractor
        
        Args:
            method: Optical flow method ('farneback', 'lucas-kanade')
        """
        self.method = method
    
    def extract(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Extract optical flow between two frames
        
        Args:
            frame1: First frame (H, W, 3)
            frame2: Second frame (H, W, 3)
            
        Returns:
            Optical flow (H, W, 2)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
        
        return flow
    
    def extract_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract optical flow for a sequence of frames
        
        Args:
            frames: List of frames (each H, W, 3)
            
        Returns:
            Optical flow sequence (T-1, H, W, 2)
        """
        flows = []
        for i in range(len(frames) - 1):
            flow = self.extract(frames[i], frames[i + 1])
            flows.append(flow)
        
        return np.stack(flows)


class TemporalFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for individual frames
    """
    
    def __init__(self, feature_dim: int = 512):
        """
        Initialize feature extractor
        
        Args:
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Input: (B, 3, 224, 224)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # (B, 64, 56, 56)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # (B, 128, 28, 28)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # (B, 256, 14, 14)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from frame
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Features (B, feature_dim)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return features


class TemporalLSTMDetector(nn.Module):
    """
    LSTM-based temporal inconsistency detector
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize temporal LSTM detector
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.feature_extractor = TemporalFeatureExtractor(feature_dim)
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through temporal detector
        
        Args:
            frames: Input frames (B, T, C, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Extract features for each frame
        frames_flat = frames.view(batch_size * seq_len, C, H, W)
        features = self.feature_extractor(frames_flat)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_dim)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        else:
            return logits, None


class TemporalTransformerDetector(nn.Module):
    """
    Transformer-based temporal inconsistency detector
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize temporal transformer detector
        
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_extractor = TemporalFeatureExtractor(feature_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, feature_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal transformer
        
        Args:
            frames: Input frames (B, T, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Extract features for each frame
        frames_flat = frames.view(batch_size * seq_len, C, H, W)
        features = self.feature_extractor(frames_flat)
        features = features.view(batch_size, seq_len, -1)
        
        # Add positional encoding
        features = features + self.pos_encoding[:, :seq_len, :]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        
        # Transformer encoding
        encoded = self.transformer(features)
        
        # Use CLS token for classification
        cls_output = encoded[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits


def create_temporal_detector(
    detector_type: str = 'lstm',
    **kwargs
) -> nn.Module:
    """
    Factory function to create temporal detector
    
    Args:
        detector_type: Type of detector ('lstm', 'transformer')
        **kwargs: Additional arguments for detector
        
    Returns:
        Temporal detector model
    """
    if detector_type == 'lstm':
        return TemporalLSTMDetector(**kwargs)
    elif detector_type == 'transformer':
        return TemporalTransformerDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == "__main__":
    # Test LSTM detector
    lstm_model = create_temporal_detector('lstm')
    
    # Test input: batch of 2, sequence of 16 frames
    x = torch.randn(2, 16, 3, 224, 224)
    
    logits, attention = lstm_model(x, return_attention=True)
    print(f"LSTM Logits shape: {logits.shape}")
    print(f"LSTM Attention shape: {attention.shape}")
    
    # Test Transformer detector
    transformer_model = create_temporal_detector('transformer')
    logits = transformer_model(x)
    print(f"Transformer Logits shape: {logits.shape}")
