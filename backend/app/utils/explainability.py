import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


def generate_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: Optional[str] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model explanation
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor (1, C, H, W)
        target_class: Target class for explanation
        target_layer: Name of target layer (if None, uses last conv layer)
        
    Returns:
        Heatmap as numpy array (H, W, 3) in RGB format
    """
    model.eval()
    
    # Store gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer
    # For EfficientNet, this is typically in the backbone
    target_module = None
    
    # Try to find the last conv layer in backbone
    if hasattr(model, 'backbone'):
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_module = module
    
    if target_module is None:
        # Fallback: find any conv layer
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_module = module
    
    if target_module is None:
        # If no conv layer found, return blank heatmap
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Register hooks
    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get gradients and activations
    if len(gradients) == 0 or len(activations) == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    # Calculate weights
    weights = np.mean(grads, axis=(1, 2))
    
    # Calculate weighted combination
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    
    # Apply ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
    
    # Resize to input size
    input_size = input_tensor.shape[2]
    cam = cv2.resize(cam, (input_size, input_size))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    original_image = input_tensor[0].cpu().numpy()
    original_image = np.transpose(original_image, (1, 2, 0))
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = std * original_image + mean
    original_image = np.clip(original_image, 0, 1)
    original_image = np.uint8(255 * original_image)
    
    # Blend
    alpha = 0.4
    blended = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return blended


def get_attention_weights(model, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract attention weights from attention-based models
    
    Args:
        model: PyTorch model with attention layers
        input_tensor: Input tensor
        
    Returns:
        Attention weights as numpy array
    """
    # This is a placeholder - implement based on specific model architecture
    # For models with explicit attention layers
    model.eval()
    
    with torch.no_grad():
        # Forward pass and extract attention weights
        # This depends on the model architecture
        pass
    
    return np.zeros((224, 224))
