import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import face_recognition


def preprocess_image(image: Image.Image, size: int = 380) -> torch.Tensor:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        size: Target size for resizing
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)


def extract_face(image: np.ndarray, margin: float = 0.2) -> np.ndarray:
    """
    Extract face from image using face_recognition library
    
    Args:
        image: Input image as numpy array (RGB)
        margin: Margin around face bounding box (0.2 = 20%)
        
    Returns:
        Cropped face image or None if no face detected
    """
    try:
        # Detect face locations
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return None
        
        # Use the first detected face
        top, right, bottom, left = face_locations[0]
        
        # Add margin
        height, width = image.shape[:2]
        margin_h = int((bottom - top) * margin)
        margin_w = int((right - left) * margin)
        
        top = max(0, top - margin_h)
        bottom = min(height, bottom + margin_h)
        left = max(0, left - margin_w)
        right = min(width, right + margin_w)
        
        # Crop face
        face_image = image[top:bottom, left:right]
        
        return face_image
        
    except Exception as e:
        print(f"Face extraction error: {e}")
        return None


def apply_augmentation(image: Image.Image) -> Image.Image:
    """
    Apply data augmentation to image
    
    Args:
        image: Input PIL Image
        
    Returns:
        Augmented PIL Image
    """
    # Convert to numpy for albumentations
    import albumentations as A
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(p=0.2),
    ])
    
    image_np = np.array(image)
    augmented = transform(image=image_np)
    
    return Image.fromarray(augmented['image'])
