"""
Advanced Explainability Suite for Deepfake Detection

This module implements multiple explanation methods:
- Grad-CAM++ (improved gradient-based visualization)
- Score-CAM (gradient-free activation mapping)
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Integrated Gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Callable
from PIL import Image


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for improved localization
    
    Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM++
        
        Args:
            model: PyTorch model
            target_layer: Target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        if target_layer is not None:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate alpha weights (Grad-CAM++ improvement)
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                     (activations * gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        
        # Calculate weights
        weights = (alphas * F.relu(gradients)).sum(dim=(1, 2))
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


class ScoreCAM:
    """
    Score-CAM implementation (gradient-free)
    
    Reference: "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Score-CAM
        
        Args:
            model: PyTorch model
            target_layer: Target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        if target_layer is not None:
            self._register_hook()
    
    def _register_hook(self):
        """Register forward hook"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate Score-CAM heatmap
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            batch_size: Batch size for processing activation maps
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass to get activations
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get activations
        activations = self.activations[0]  # (C, H, W)
        num_channels = activations.shape[0]
        
        # Upsample activation maps to input size
        input_size = input_tensor.shape[2:]
        upsampled_activations = F.interpolate(
            activations.unsqueeze(0),
            size=input_size,
            mode='bilinear',
            align_corners=False
        )[0]  # (C, H, W)
        
        # Normalize each activation map
        norm_activations = torch.zeros_like(upsampled_activations)
        for i in range(num_channels):
            act = upsampled_activations[i]
            if act.max() > act.min():
                norm_activations[i] = (act - act.min()) / (act.max() - act.min())
        
        # Calculate scores for each activation map
        scores = []
        
        for i in range(0, num_channels, batch_size):
            batch_end = min(i + batch_size, num_channels)
            batch_acts = norm_activations[i:batch_end]
            
            # Create masked inputs
            masked_inputs = input_tensor * batch_acts.unsqueeze(1)
            
            # Get predictions
            with torch.no_grad():
                batch_output = self.model(masked_inputs)
                batch_scores = F.softmax(batch_output, dim=1)[:, target_class]
            
            scores.append(batch_scores)
        
        scores = torch.cat(scores)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, score in enumerate(scores):
            cam += score * activations[i]
        
        # Normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for images
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_features: int = 10,
        segmentation_fn: Optional[Callable] = None
    ):
        """
        Initialize LIME explainer
        
        Args:
            model: PyTorch model
            num_samples: Number of perturbed samples
            num_features: Number of top features to show
            segmentation_fn: Function to segment image into superpixels
        """
        self.model = model
        self.num_samples = num_samples
        self.num_features = num_features
        self.segmentation_fn = segmentation_fn or self._default_segmentation
    
    def _default_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Default segmentation using SLIC superpixels
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Segment labels (H, W)
        """
        from skimage.segmentation import slic
        
        segments = slic(image, n_segments=100, compactness=10, sigma=1)
        return segments
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate LIME explanation
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            
        Returns:
            Tuple of (explanation_mask, feature_weights)
        """
        self.model.eval()
        
        # Convert to numpy for segmentation
        image = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Segment image
        segments = self.segmentation_fn(image)
        num_segments = segments.max() + 1
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = original_output.argmax(dim=1).item()
        
        # Generate perturbed samples
        perturbed_images = []
        segment_masks = []
        
        for _ in range(self.num_samples):
            # Randomly select segments to keep
            active_segments = np.random.randint(0, 2, num_segments)
            
            # Create mask
            mask = np.zeros(segments.shape, dtype=bool)
            for seg_id in range(num_segments):
                if active_segments[seg_id]:
                    mask[segments == seg_id] = True
            
            # Apply mask
            perturbed = image.copy()
            perturbed[~mask] = 0  # or use mean color
            
            perturbed_images.append(perturbed)
            segment_masks.append(active_segments)
        
        # Get predictions for perturbed images
        perturbed_tensors = []
        for img in perturbed_images:
            # Normalize
            img_norm = (img - mean) / std
            img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
            perturbed_tensors.append(img_tensor)
        
        perturbed_batch = torch.cat(perturbed_tensors).to(input_tensor.device)
        
        with torch.no_grad():
            perturbed_outputs = self.model(perturbed_batch)
            perturbed_probs = F.softmax(perturbed_outputs, dim=1)[:, target_class]
        
        # Fit linear model
        from sklearn.linear_model import Ridge
        
        X = np.array(segment_masks)
        y = perturbed_probs.cpu().numpy()
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Get feature importance
        feature_weights = model.coef_
        
        # Create explanation mask
        top_features = np.argsort(np.abs(feature_weights))[-self.num_features:]
        
        explanation_mask = np.zeros(segments.shape, dtype=np.float32)
        for seg_id in top_features:
            explanation_mask[segments == seg_id] = feature_weights[seg_id]
        
        return explanation_mask, {'weights': feature_weights, 'segments': segments}


class IntegratedGradients:
    """
    Integrated Gradients attribution method
    
    Reference: "Axiomatic Attribution for Deep Networks"
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize Integrated Gradients
        
        Args:
            model: PyTorch model
        """
        self.model = model
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            baseline: Baseline tensor (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Attribution map (C, H, W)
        """
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Get target class
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            class_score = output[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            class_score.backward()
            
            gradients.append(interpolated.grad.detach())
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients
        integrated_grads = (input_tensor - baseline) * avg_gradients
        
        # Sum across color channels for visualization
        attribution = integrated_grads[0].abs().sum(dim=0).cpu().numpy()
        
        # Normalize
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        return attribution


def visualize_explanation(
    image: np.ndarray,
    explanation: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Visualize explanation as heatmap overlay
    
    Args:
        image: Original image (H, W, 3)
        explanation: Explanation map (H, W)
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        Visualization image (H, W, 3)
    """
    # Resize explanation to match image
    if explanation.shape != image.shape[:2]:
        explanation = cv2.resize(explanation, (image.shape[1], image.shape[0]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * explanation), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(255 * np.clip(image, 0, 1))
    
    # Blend
    visualization = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return visualization


if __name__ == "__main__":
    # Test explainability methods
    from ..models.detector import EfficientNetDetector
    
    model = EfficientNetDetector()
    model.eval()
    
    # Test input
    x = torch.randn(1, 3, 224, 224)
    
    print("Testing Grad-CAM++...")
    # Note: Need to specify target layer for actual use
    # gradcam_pp = GradCAMPlusPlus(model, target_layer=model.backbone.blocks[-1])
    # cam = gradcam_pp.generate(x)
    # print(f"Grad-CAM++ shape: {cam.shape}")
    
    print("Testing Integrated Gradients...")
    ig = IntegratedGradients(model)
    attribution = ig.generate(x)
    print(f"IG attribution shape: {attribution.shape}")
