import torch
import torch.nn as nn
from torchvision import models
import timm

class EfficientNetDetector(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, model_name='efficientnet_b4', num_classes=2, pretrained=True):
        super(EfficientNetDetector, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 380, 380)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features for visualization"""
        return self.backbone(x)


class XceptionDetector(nn.Module):
    """Modified Xception architecture for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionDetector, self).__init__()
        
        # Load Xception backbone
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 299, 299)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features for visualization"""
        return self.backbone(x)


class AttentionBlock(nn.Module):
    """Self-attention block for feature enhancement"""
    
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention map
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class CustomCNNDetector(nn.Module):
    """Custom CNN with attention mechanisms"""
    
    def __init__(self, num_classes=2):
        super(CustomCNNDetector, self).__init__()
        
        # Convolutional blocks
        self.conv1 = self._make_conv_block(3, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        
        # Attention blocks
        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(512)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention1(x)
        x = self.conv4(x)
        x = self.attention2(x)
        output = self.classifier(x)
        return output
    
    def get_features(self, x):
        """Extract features for visualization"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention1(x)
        x = self.conv4(x)
        x = self.attention2(x)
        return x


def load_model(model_path, model_type='efficientnet', device='cuda'):
    """Load a trained model from checkpoint"""
    
    if model_type == 'efficientnet':
        model = EfficientNetDetector()
    elif model_type == 'xception':
        model = XceptionDetector()
    elif model_type == 'custom_cnn':
        model = CustomCNNDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights if path exists
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test models
    from pathlib import Path
    
    print("Testing EfficientNet Detector...")
    efficientnet = EfficientNetDetector()
    x = torch.randn(2, 3, 380, 380)
    out = efficientnet(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting Xception Detector...")
    xception = XceptionDetector()
    x = torch.randn(2, 3, 299, 299)
    out = xception(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting Custom CNN Detector...")
    custom = CustomCNNDetector()
    x = torch.randn(2, 3, 224, 224)
    out = custom(x)
    print(f"Output shape: {out.shape}")
