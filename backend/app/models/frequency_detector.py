import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDetector(nn.Module):
    """
    Frequency Domain Detector for deepfake artifacts using 2D FFT.
    Extracts magnitude spectrum and classifies it.
    """
    
    def __init__(self, num_classes=2):
        super(FrequencyDetector, self).__init__()
        
        # Simple CNN to process the 2D FFT magnitude spectrum
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """
        x: Input image tensor of shape (B, C, H, W)
           Values should be normalized, typically in [-1, 1] or [0, 1]
        """
        # Compute 2D FFT
        # x is (B, 3, H, W)
        # torch.fft.fft2 computes 2D FFT over the last two dimensions
        fft_result = torch.fft.fft2(x)
        
        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        
        # Compute magnitude spectrum (log scale to compress dynamic range)
        # Add epsilon to prevent log(0)
        magnitude_spectrum = torch.log(torch.abs(fft_shifted) + 1e-8)
        
        # Normalize the spectrum for the CNN
        # Normalize per channel over the batch
        B, C, H, W = magnitude_spectrum.shape
        mag_flat = magnitude_spectrum.view(B, C, -1)
        mag_min = mag_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        mag_max = mag_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        normalized_spectrum = (magnitude_spectrum - mag_min) / (mag_max - mag_min + 1e-8)
        
        # Pass through CNN
        features = self.conv1(normalized_spectrum)
        features = self.conv2(features)
        features = self.conv3(features)
        
        output = self.classifier(features)
        return output

    def get_features(self, x):
        """Extract features for visualization/ensemble"""
        fft_result = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        magnitude_spectrum = torch.log(torch.abs(fft_shifted) + 1e-8)
        
        B, C, H, W = magnitude_spectrum.shape
        mag_flat = magnitude_spectrum.view(B, C, -1)
        mag_min = mag_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        mag_max = mag_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        normalized_spectrum = (magnitude_spectrum - mag_min) / (mag_max - mag_min + 1e-8)
        
        features = self.conv1(normalized_spectrum)
        features = self.conv2(features)
        features = self.conv3(features)
        return features
