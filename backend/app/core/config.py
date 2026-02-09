from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "DeepGuard"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # Model Settings
    MODEL_PATH: Path = Path("models")
    EFFICIENTNET_MODEL: str = "efficientnet_b4_deepfake.pth"
    XCEPTION_MODEL: str = "xception_deepfake.pth"
    CUSTOM_CNN_MODEL: str = "custom_cnn_deepfake.pth"
    
    # Processing Settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    # Image Processing
    IMAGE_SIZE: int = 380
    BATCH_SIZE: int = 32
    
    # Video Processing
    DEFAULT_FRAME_SAMPLING: str = "uniform"
    DEFAULT_NUM_FRAMES: int = 32
    MAX_VIDEO_FRAMES: int = 100
    
    # Model Inference
    DEVICE: str = "cuda"  # or "cpu"
    USE_ENSEMBLE: bool = True
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Explainability
    ENABLE_GRADCAM: bool = True
    GRADCAM_LAYER: str = "layer4"
    
    # Upload Settings
    UPLOAD_DIR: Path = Path("uploads")
    TEMP_DIR: Path = Path("temp")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create necessary directories
settings.MODEL_PATH.mkdir(exist_ok=True)
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.TEMP_DIR.mkdir(exist_ok=True)
