"""
Frequency Domain Analyzer for Deepfake Detection

This module implements frequency domain analysis to detect GAN artifacts,
compression inconsistencies, and other frequency-based anomalies.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from scipy import fftpack
from scipy.fft import dct, idct


class FrequencyAnalyzer:
    """
    Analyze images in frequency domain to detect deepfake artifacts
    """
    
    def __init__(self):
        """Initialize frequency analyzer"""
        pass
    
    def compute_fft(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 2D Fast Fourier Transform
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            
        Returns:
            FFT magnitude spectrum (H, W)
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute FFT
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        
        # Magnitude spectrum
        magnitude = np.abs(fft_shift)
        
        # Log scale for visualization
        magnitude_log = np.log1p(magnitude)
        
        return magnitude_log
    
    def compute_dct(self, image: np.ndarray, block_size: int = 8) -> np.ndarray:
        """
        Compute block-wise Discrete Cosine Transform
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            block_size: Size of DCT blocks (typically 8 for JPEG)
            
        Returns:
            DCT coefficients
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = image.shape
        
        # Pad image to be divisible by block_size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        h, w = image.shape
        
        # Compute DCT for each block
        dct_blocks = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[i:i+block_size, j:j+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_blocks[i:i+block_size, j:j+block_size] = dct_block
        
        return dct_blocks
    
    def detect_compression_artifacts(
        self,
        image: np.ndarray,
        block_size: int = 8
    ) -> Tuple[float, np.ndarray]:
        """
        Detect JPEG compression artifacts
        
        Args:
            image: Input image (H, W, 3)
            block_size: DCT block size
            
        Returns:
            Tuple of (artifact_score, artifact_map)
        """
        # Compute DCT
        dct_coeffs = self.compute_dct(image, block_size)
        
        # Analyze high-frequency components
        h, w = dct_coeffs.shape
        artifact_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = dct_coeffs[i:i+block_size, j:j+block_size]
                
                # High-frequency energy (bottom-right of DCT block)
                hf_energy = np.sum(np.abs(block[4:, 4:]))
                
                # Normalize by total energy
                total_energy = np.sum(np.abs(block)) + 1e-10
                hf_ratio = hf_energy / total_energy
                
                artifact_map[i:i+block_size, j:j+block_size] = hf_ratio
        
        # Overall artifact score
        artifact_score = np.mean(artifact_map)
        
        return artifact_score, artifact_map
    
    def detect_gan_fingerprint(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect GAN-specific frequency patterns
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Tuple of (gan_score, frequency_heatmap)
        """
        # Compute FFT for each channel
        channels = cv2.split(image)
        fft_magnitudes = []
        
        for channel in channels:
            fft_mag = self.compute_fft(channel)
            fft_magnitudes.append(fft_mag)
        
        # Average across channels
        avg_fft = np.mean(fft_magnitudes, axis=0)
        
        # Analyze radial frequency distribution
        h, w = avg_fft.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radial profile
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Bin by radius
        max_radius = int(np.sqrt(center_x**2 + center_y**2))
        radial_profile = np.zeros(max_radius)
        
        for r in range(max_radius):
            mask = (radius >= r) & (radius < r + 1)
            if np.any(mask):
                radial_profile[r] = np.mean(avg_fft[mask])
        
        # Detect anomalies in radial profile
        # GANs often have specific frequency peaks
        profile_diff = np.diff(radial_profile)
        gan_score = np.std(profile_diff)
        
        # Create heatmap
        frequency_heatmap = avg_fft
        
        return gan_score, frequency_heatmap
    
    def analyze_noise_pattern(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze noise patterns in the image
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Dictionary with noise statistics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply high-pass filter to extract noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # Noise statistics
        noise_std = np.std(noise)
        noise_mean = np.mean(np.abs(noise))
        
        # Local noise variance
        kernel_size = 16
        local_vars = []
        
        h, w = gray.shape
        for i in range(0, h - kernel_size, kernel_size):
            for j in range(0, w - kernel_size, kernel_size):
                block = noise[i:i+kernel_size, j:j+kernel_size]
                local_vars.append(np.var(block))
        
        noise_variance_std = np.std(local_vars)
        
        return {
            'noise_std': float(noise_std),
            'noise_mean': float(noise_mean),
            'noise_variance_std': float(noise_variance_std)
        }
    
    def comprehensive_analysis(
        self,
        image: np.ndarray
    ) -> Dict[str, any]:
        """
        Perform comprehensive frequency domain analysis
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Dictionary with all analysis results
        """
        # Compression artifacts
        compression_score, compression_map = self.detect_compression_artifacts(image)
        
        # GAN fingerprint
        gan_score, frequency_heatmap = self.detect_gan_fingerprint(image)
        
        # Noise analysis
        noise_stats = self.analyze_noise_pattern(image)
        
        # Combine scores
        results = {
            'compression_score': float(compression_score),
            'compression_map': compression_map,
            'gan_score': float(gan_score),
            'frequency_heatmap': frequency_heatmap,
            'noise_stats': noise_stats,
            'overall_anomaly_score': float(
                0.4 * compression_score +
                0.4 * (gan_score / 10.0) +  # Normalize
                0.2 * noise_stats['noise_variance_std']
            )
        }
        
        return results


class FrequencyFeatureExtractor(nn.Module):
    """
    Neural network to extract features from frequency domain
    """
    
    def __init__(self, feature_dim: int = 256):
        """
        Initialize frequency feature extractor
        
        Args:
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        # Process FFT magnitude spectrum
        self.fft_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = feature_dim
        self.fc = nn.Linear(256, feature_dim)
    
    def forward(self, fft_input: torch.Tensor) -> torch.Tensor:
        """
        Extract features from FFT
        
        Args:
            fft_input: FFT magnitude spectrum (B, 1, H, W)
            
        Returns:
            Features (B, feature_dim)
        """
        features = self.fft_encoder(fft_input)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


def create_frequency_analyzer() -> FrequencyAnalyzer:
    """
    Factory function to create frequency analyzer
    
    Returns:
        Frequency analyzer instance
    """
    return FrequencyAnalyzer()


if __name__ == "__main__":
    # Test frequency analyzer
    analyzer = create_frequency_analyzer()
    
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Comprehensive analysis
    results = analyzer.comprehensive_analysis(test_image)
    
    print("Frequency Analysis Results:")
    print(f"Compression Score: {results['compression_score']:.4f}")
    print(f"GAN Score: {results['gan_score']:.4f}")
    print(f"Overall Anomaly Score: {results['overall_anomaly_score']:.4f}")
    print(f"Noise Stats: {results['noise_stats']}")
