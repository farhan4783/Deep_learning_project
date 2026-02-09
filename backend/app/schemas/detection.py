from pydantic import BaseModel
from typing import Optional, List, Dict

class DetectionResponse(BaseModel):
    """Response model for detection results"""
    is_fake: bool
    confidence: float
    prediction_scores: Dict[str, float]
    model_predictions: Optional[Dict[str, Dict[str, float]]] = None
    explanation: Optional[str] = None
    heatmap_base64: Optional[str] = None
    processing_time: float
    
class BatchDetectionResponse(BaseModel):
    """Response model for batch detection"""
    results: List[Dict]
