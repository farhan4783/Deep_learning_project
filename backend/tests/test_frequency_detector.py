import pytest
import torch
from app.models.frequency_detector import FrequencyDetector

def test_frequency_detector_initialization():
    """Test that the model initializes correctly with default params"""
    model = FrequencyDetector()
    assert isinstance(model, FrequencyDetector)
    
def test_frequency_detector_forward():
    """Test that the model processes an image batch correctly"""
    model = FrequencyDetector(num_classes=2)
    model.eval()
    
    # Create a dummy image batch (B=2, C=3, H=224, W=224)
    dummy_input = torch.rand(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()

def test_frequency_detector_get_features():
    """Test that the model can extract features"""
    model = FrequencyDetector()
    model.eval()
    
    dummy_input = torch.rand(1, 3, 224, 224)
    
    with torch.no_grad():
        features = model.get_features(dummy_input)
        
    # The CNN preserves spatial dimensions while max-pooling 3 times
    # 224 -> 112 -> 56 -> 28
    assert features.shape == (1, 128, 28, 28)
