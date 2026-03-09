import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import base64
from io import BytesIO

from app.models.detector import load_model, EfficientNetDetector, XceptionDetector, CustomCNNDetector
from app.models.frequency_detector import FrequencyDetector
from app.core.config import settings
from app.utils.image_processing import preprocess_image, extract_face
from app.utils.video_processing import extract_frames
from app.utils.explainability import generate_gradcam


class DetectionService:
    """Service for deepfake detection"""
    
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all detection models"""
        try:
            # Load EfficientNet
            efficientnet_path = settings.MODEL_PATH / settings.EFFICIENTNET_MODEL
            if efficientnet_path.exists():
                self.models['efficientnet'] = load_model(
                    efficientnet_path, 
                    model_type='efficientnet',
                    device=self.device
                )
                print("✅ Loaded EfficientNet model")
            else:
                # Load without pretrained weights for demo
                self.models['efficientnet'] = EfficientNetDetector().to(self.device)
                self.models['efficientnet'].eval()
                print("⚠️ Using untrained EfficientNet model (demo mode)")
            
            # Load Xception
            xception_path = settings.MODEL_PATH / settings.XCEPTION_MODEL
            if xception_path.exists():
                self.models['xception'] = load_model(
                    xception_path,
                    model_type='xception',
                    device=self.device
                )
                print("✅ Loaded Xception model")
            else:
                self.models['xception'] = XceptionDetector().to(self.device)
                self.models['xception'].eval()
                print("⚠️ Using untrained Xception model (demo mode)")
            
            # Load Custom CNN
            custom_path = settings.MODEL_PATH / settings.CUSTOM_CNN_MODEL
            if custom_path.exists():
                self.models['custom_cnn'] = load_model(
                    custom_path,
                    model_type='custom_cnn',
                    device=self.device
                )
                print("✅ Loaded Custom CNN model")
            else:
                self.models['custom_cnn'] = CustomCNNDetector().to(self.device)
                self.models['custom_cnn'].eval()
                print("⚠️ Using untrained Custom CNN model (demo mode)")
            
            # Load Frequency Detector
            self.models['frequency_detector'] = FrequencyDetector().to(self.device)
            self.models['frequency_detector'].eval()
            print("⚠️ Using untrained Frequency Detector model (demo mode)")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    async def detect_image(
        self,
        image_path: Path,
        explain: bool = True
    ) -> Dict:
        """
        Detect deepfake in an image
        
        Args:
            image_path: Path to image file
            explain: Whether to generate Grad-CAM explanation
            
        Returns:
            Detection results dictionary
        """
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Try to extract face, fallback to full image
        face_image = extract_face(np.array(image))
        if face_image is not None:
            image = Image.fromarray(face_image)
        
        # Get predictions from all models
        model_predictions = {}
        
        # EfficientNet (380x380)
        efficientnet_input = preprocess_image(image, size=380)
        efficientnet_input = efficientnet_input.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            efficientnet_output = self.models['efficientnet'](efficientnet_input)
            efficientnet_probs = F.softmax(efficientnet_output, dim=1)[0]
            model_predictions['efficientnet'] = {
                'real': float(efficientnet_probs[0]),
                'fake': float(efficientnet_probs[1])
            }
        
        # Xception (299x299)
        xception_input = preprocess_image(image, size=299)
        xception_input = xception_input.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            xception_output = self.models['xception'](xception_input)
            xception_probs = F.softmax(xception_output, dim=1)[0]
            model_predictions['xception'] = {
                'real': float(xception_probs[0]),
                'fake': float(xception_probs[1])
            }
        
        # Custom CNN (224x224)
        custom_input = preprocess_image(image, size=224)
        custom_input = custom_input.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            custom_output = self.models['custom_cnn'](custom_input)
            custom_probs = F.softmax(custom_output, dim=1)[0]
            model_predictions['custom_cnn'] = {
                'real': float(custom_probs[0]),
                'fake': float(custom_probs[1])
            }
            
        # Frequency Detector
        with torch.no_grad():
            frequency_output = self.models['frequency_detector'](custom_input)
            frequency_probs = F.softmax(frequency_output, dim=1)[0]
            model_predictions['frequency_detector'] = {
                'real': float(frequency_probs[0]),
                'fake': float(frequency_probs[1])
            }
        
        # Ensemble prediction (weighted average)
        if settings.USE_ENSEMBLE:
            weights = {'efficientnet': 0.35, 'xception': 0.35, 'custom_cnn': 0.15, 'frequency_detector': 0.15}
            ensemble_fake_prob = sum(
                model_predictions[model]['fake'] * weights[model]
                for model in weights
            )
        else:
            ensemble_fake_prob = model_predictions['efficientnet']['fake']
        
        is_fake = ensemble_fake_prob > settings.CONFIDENCE_THRESHOLD
        
        # Generate explanation if requested
        heatmap_base64 = None
        if explain:
            heatmap = generate_gradcam(
                self.models['efficientnet'],
                efficientnet_input,
                target_class=1 if is_fake else 0
            )
            
            # Convert heatmap to base64
            heatmap_img = Image.fromarray(heatmap)
            buffered = BytesIO()
            heatmap_img.save(buffered, format="PNG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        processing_time = time.time() - start_time
        
        return {
            "is_fake": is_fake,
            "confidence": float(ensemble_fake_prob),
            "prediction_scores": {
                "real": float(1 - ensemble_fake_prob),
                "fake": float(ensemble_fake_prob)
            },
            "model_predictions": model_predictions,
            "explanation": "Grad-CAM heatmap shows regions that influenced the prediction" if explain else None,
            "heatmap_base64": heatmap_base64,
            "processing_time": processing_time
        }
    
    async def detect_video(
        self,
        video_path: Path,
        frame_sampling: str = "uniform",
        num_frames: int = 32,
        explain: bool = False
    ) -> Dict:
        """
        Detect deepfake in a video
        
        Args:
            video_path: Path to video file
            frame_sampling: Frame sampling strategy
            num_frames: Number of frames to analyze
            explain: Whether to generate explanations
            
        Returns:
            Aggregated detection results
        """
        start_time = time.time()
        
        # Extract frames
        frames = extract_frames(
            video_path,
            num_frames=num_frames,
            sampling=frame_sampling
        )
        
        if len(frames) == 0:
            raise ValueError("No frames could be extracted from video")
        
        # Analyze each frame
        frame_predictions = []
        
        for frame in frames:
            # Convert frame to PIL Image
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Get prediction for this frame
            # Use a simplified version without explanation for speed
            face_image = extract_face(np.array(frame_image))
            if face_image is not None:
                frame_image = Image.fromarray(face_image)
            
            # Use EfficientNet for video frames
            frame_input = preprocess_image(frame_image, size=380)
            frame_input = frame_input.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.models['efficientnet'](frame_input)
                probs = F.softmax(output, dim=1)[0]
                frame_predictions.append(float(probs[1]))  # Fake probability
        
        # Aggregate predictions
        avg_fake_prob = np.mean(frame_predictions)
        max_fake_prob = np.max(frame_predictions)
        std_fake_prob = np.std(frame_predictions)
        
        is_fake = avg_fake_prob > settings.CONFIDENCE_THRESHOLD
        
        processing_time = time.time() - start_time
        
        return {
            "is_fake": is_fake,
            "confidence": float(avg_fake_prob),
            "prediction_scores": {
                "real": float(1 - avg_fake_prob),
                "fake": float(avg_fake_prob)
            },
            "model_predictions": {
                "frames_analyzed": len(frames),
                "average_fake_probability": float(avg_fake_prob),
                "max_fake_probability": float(max_fake_prob),
                "std_fake_probability": float(std_fake_prob),
                "frame_predictions": frame_predictions
            },
            "explanation": f"Analyzed {len(frames)} frames from video",
            "heatmap_base64": None,
            "processing_time": processing_time
        }
