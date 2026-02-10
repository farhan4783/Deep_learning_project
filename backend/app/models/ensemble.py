"""
Enhanced Ensemble System for Deepfake Detection

This module implements an advanced ensemble that combines multiple detection models
with weighted voting, uncertainty quantification, and adaptive model selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

# Import individual detectors
from .detector import EfficientNetDetector, XceptionDetector, CustomCNNDetector
from .vit_detector import ViTDeepfakeDetector, ViTDetectorWithUncertainty
from .temporal_detector import TemporalLSTMDetector, TemporalTransformerDetector
from .audio_visual_sync import AudioVisualSyncDetector


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble of multiple deepfake detectors
    
    Features:
    - Configurable model weights
    - Uncertainty-aware fusion
    - Dynamic weight adjustment
    - Multi-modal support (image, video, audio-visual)
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        use_uncertainty: bool = True
    ):
        """
        Initialize weighted ensemble
        
        Args:
            models: Dictionary of model_name -> model
            weights: Dictionary of model_name -> weight (None for equal weights)
            use_uncertainty: Whether to use uncertainty for weighting
        """
        super().__init__()
        
        self.models = nn.ModuleDict(models)
        self.use_uncertainty = use_uncertainty
        
        # Initialize weights
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        
        # Learnable weight parameters (optional)
        self.learnable_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight))
            for name, weight in self.weights.items()
        })
    
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[torch.Tensor]]:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor
            return_individual: Whether to return individual model predictions
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Tuple of (ensemble_logits, individual_predictions, uncertainty)
        """
        individual_preds = {}
        uncertainties = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            model.eval()
            
            with torch.no_grad():
                if isinstance(model, ViTDetectorWithUncertainty) and self.use_uncertainty:
                    logits, std, epistemic = model.predict_with_uncertainty(x)
                    uncertainties[name] = epistemic
                else:
                    logits = model(x)
                
                individual_preds[name] = logits
        
        # Weighted fusion
        ensemble_logits = torch.zeros_like(list(individual_preds.values())[0])
        
        # Normalize weights
        weight_sum = sum(F.softplus(self.learnable_weights[name]) 
                        for name in self.models.keys())
        
        for name, logits in individual_preds.items():
            weight = F.softplus(self.learnable_weights[name]) / weight_sum
            
            # Adjust weight based on uncertainty if available
            if name in uncertainties and self.use_uncertainty:
                uncertainty_weight = 1.0 / (1.0 + uncertainties[name].mean())
                weight = weight * uncertainty_weight
            
            ensemble_logits += weight * logits
        
        # Calculate ensemble uncertainty
        ensemble_uncertainty = None
        if return_uncertainty and uncertainties:
            ensemble_uncertainty = torch.stack(list(uncertainties.values())).mean(dim=0)
        
        return_vals = [ensemble_logits]
        return_vals.append(individual_preds if return_individual else None)
        return_vals.append(ensemble_uncertainty if return_uncertainty else None)
        
        return tuple(return_vals)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        weight_sum = sum(F.softplus(self.learnable_weights[name]).item() 
                        for name in self.models.keys())
        
        return {
            name: F.softplus(self.learnable_weights[name]).item() / weight_sum
            for name in self.models.keys()
        }


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that selects models based on input characteristics
    """
    
    def __init__(
        self,
        image_models: Dict[str, nn.Module],
        video_models: Optional[Dict[str, nn.Module]] = None,
        av_models: Optional[Dict[str, nn.Module]] = None
    ):
        """
        Initialize adaptive ensemble
        
        Args:
            image_models: Models for image detection
            video_models: Models for video temporal analysis
            av_models: Models for audio-visual sync detection
        """
        super().__init__()
        
        self.image_ensemble = WeightedEnsemble(image_models)
        
        if video_models:
            self.video_ensemble = WeightedEnsemble(video_models)
        else:
            self.video_ensemble = None
        
        if av_models:
            self.av_ensemble = WeightedEnsemble(av_models)
        else:
            self.av_ensemble = None
        
        # Meta-learner to combine different modalities
        self.meta_classifier = nn.Sequential(
            nn.Linear(2 * (1 + int(video_models is not None) + int(av_models is not None)), 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(
        self,
        image: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        lip_frames: Optional[torch.Tensor] = None,
        audio_spec: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive model selection
        
        Args:
            image: Input image (B, 3, H, W)
            video_frames: Video frames for temporal analysis (B, T, 3, H, W)
            lip_frames: Lip region frames (B, 3, T, H, W)
            audio_spec: Audio spectrogram (B, 1, T, F)
            
        Returns:
            Tuple of (final_logits, modality_predictions)
        """
        modality_preds = {}
        
        # Image-based detection
        image_logits, _, _ = self.image_ensemble(image)
        modality_preds['image'] = image_logits
        
        # Video temporal analysis
        if video_frames is not None and self.video_ensemble is not None:
            video_logits, _, _ = self.video_ensemble(video_frames)
            modality_preds['video'] = video_logits
        
        # Audio-visual sync detection
        if lip_frames is not None and audio_spec is not None and self.av_ensemble is not None:
            av_logits, _, _ = self.av_ensemble(lip_frames, audio_spec)
            modality_preds['audio_visual'] = av_logits
        
        # Combine modality predictions
        combined_features = torch.cat(list(modality_preds.values()), dim=1)
        final_logits = self.meta_classifier(combined_features)
        
        return final_logits, modality_preds


class EnsembleWithCalibration(nn.Module):
    """
    Ensemble with temperature scaling for probability calibration
    """
    
    def __init__(self, base_ensemble: nn.Module):
        """
        Initialize calibrated ensemble
        
        Args:
            base_ensemble: Base ensemble model
        """
        super().__init__()
        
        self.base_ensemble = base_ensemble
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temperature scaling
        
        Returns:
            Tuple of (calibrated_logits, calibrated_probs)
        """
        logits, *other_outputs = self.base_ensemble(*args, **kwargs)
        
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        calibrated_probs = F.softmax(calibrated_logits, dim=1)
        
        return calibrated_logits, calibrated_probs
    
    def set_temperature(self, valid_loader, device='cuda'):
        """
        Tune temperature parameter on validation set
        
        Args:
            valid_loader: Validation data loader
            device: Device to use
        """
        self.eval()
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in valid_loader:
                images, labels = batch
                images = images.to(device)
                
                logits, *_ = self.base_ensemble(images)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")


def create_ensemble(
    ensemble_type: str = 'weighted',
    model_configs: Optional[Dict] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create ensemble
    
    Args:
        ensemble_type: Type of ensemble ('weighted', 'adaptive', 'calibrated')
        model_configs: Configuration for individual models
        **kwargs: Additional arguments
        
    Returns:
        Ensemble model
    """
    if model_configs is None:
        # Default configuration
        image_models = {
            'efficientnet': EfficientNetDetector(),
            'xception': XceptionDetector(),
            'custom_cnn': CustomCNNDetector(),
            'vit': ViTDeepfakeDetector()
        }
    else:
        image_models = model_configs.get('image_models', {})
    
    if ensemble_type == 'weighted':
        return WeightedEnsemble(image_models, **kwargs)
    
    elif ensemble_type == 'adaptive':
        video_models = model_configs.get('video_models', {}) if model_configs else {}
        av_models = model_configs.get('av_models', {}) if model_configs else {}
        
        return AdaptiveEnsemble(
            image_models=image_models,
            video_models=video_models if video_models else None,
            av_models=av_models if av_models else None
        )
    
    elif ensemble_type == 'calibrated':
        base_ensemble = WeightedEnsemble(image_models, **kwargs)
        return EnsembleWithCalibration(base_ensemble)
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


if __name__ == "__main__":
    # Test weighted ensemble
    print("Testing Weighted Ensemble...")
    
    models = {
        'efficientnet': EfficientNetDetector(),
        'xception': XceptionDetector(),
        'vit': ViTDeepfakeDetector()
    }
    
    ensemble = create_ensemble('weighted', model_configs={'image_models': models})
    
    x = torch.randn(2, 3, 224, 224)
    logits, individual, uncertainty = ensemble(x, return_individual=True, return_uncertainty=True)
    
    print(f"Ensemble logits shape: {logits.shape}")
    print(f"Individual predictions: {list(individual.keys())}")
    print(f"Model weights: {ensemble.get_model_weights()}")
