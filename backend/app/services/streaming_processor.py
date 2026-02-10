"""
Streaming Video Processor for Real-time Deepfake Detection

This module processes video streams chunk-by-chunk with progressive results.
"""

import torch
import numpy as np
import cv2
from typing import AsyncGenerator, Dict, Optional, List
from pathlib import Path
import asyncio
from datetime import datetime
import hashlib

from ..models.ensemble import WeightedEnsemble
from ..utils.video_processing import extract_frames, get_video_info
from ..api.websocket import send_detection_progress, send_frame_result


class StreamingVideoProcessor:
    """
    Process video streams with real-time progress updates
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        chunk_size: int = 16,
        frame_skip: int = 1
    ):
        """
        Initialize streaming processor
        
        Args:
            model: Detection model
            device: Device to use ('cuda' or 'cpu')
            chunk_size: Number of frames to process at once
            frame_skip: Skip every N frames for faster processing
        """
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.frame_skip = frame_skip
        
        self.model.to(device)
        self.model.eval()
    
    async def process_video_stream(
        self,
        video_path: Path,
        task_id: str,
        send_updates: bool = True
    ) -> AsyncGenerator[Dict, None]:
        """
        Process video stream with progressive results
        
        Args:
            video_path: Path to video file
            task_id: Task identifier for WebSocket updates
            send_updates: Whether to send WebSocket updates
            
        Yields:
            Dictionary with frame results
        """
        # Get video info
        if send_updates:
            await send_detection_progress(
                task_id=task_id,
                stage='preprocessing',
                progress=0,
                message='Analyzing video...'
            )
        
        video_info = get_video_info(video_path)
        total_frames = video_info['total_frames']
        fps = video_info['fps']
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_idx = 0
        processed_frames = 0
        frame_buffer = []
        results = []
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_idx % (self.frame_skip + 1) != 0:
                    frame_idx += 1
                    continue
                
                # Add to buffer
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)
                
                # Process chunk when buffer is full
                if len(frame_buffer) >= self.chunk_size:
                    chunk_results = await self._process_chunk(
                        frame_buffer,
                        task_id,
                        processed_frames,
                        total_frames,
                        send_updates
                    )
                    
                    for result in chunk_results:
                        results.append(result)
                        yield result
                    
                    processed_frames += len(frame_buffer)
                    frame_buffer = []
                
                frame_idx += 1
            
            # Process remaining frames
            if frame_buffer:
                chunk_results = await self._process_chunk(
                    frame_buffer,
                    task_id,
                    processed_frames,
                    total_frames,
                    send_updates
                )
                
                for result in chunk_results:
                    results.append(result)
                    yield result
        
        finally:
            cap.release()
        
        # Send final summary
        if send_updates:
            await self._send_summary(task_id, results, video_info)
    
    async def _process_chunk(
        self,
        frames: List[np.ndarray],
        task_id: str,
        start_idx: int,
        total_frames: int,
        send_updates: bool
    ) -> List[Dict]:
        """
        Process a chunk of frames
        
        Args:
            frames: List of frames to process
            task_id: Task identifier
            start_idx: Starting frame index
            total_frames: Total frames in video
            send_updates: Whether to send updates
            
        Returns:
            List of frame results
        """
        # Preprocess frames
        preprocessed = self._preprocess_frames(frames)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(preprocessed)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1)[0]
        
        # Collect results
        results = []
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            frame_idx = start_idx + i
            
            result = {
                'frame_idx': frame_idx,
                'prediction': 'fake' if pred.item() == 1 else 'real',
                'confidence': conf.item(),
                'timestamp': frame_idx / 30.0  # Assuming 30 FPS
            }
            
            results.append(result)
            
            # Send WebSocket update
            if send_updates:
                await send_frame_result(
                    task_id=task_id,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    confidence=conf.item(),
                    prediction=result['prediction']
                )
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        return results
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess frames for model input
        
        Args:
            frames: List of frames (H, W, 3)
            
        Returns:
            Preprocessed tensor (B, 3, H, W)
        """
        # Resize frames
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (224, 224))
            resized.append(resized_frame)
        
        # Convert to tensor
        frames_array = np.stack(resized).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames_array = (frames_array - mean) / std
        
        # Convert to PyTorch tensor
        frames_tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2)).float()
        frames_tensor = frames_tensor.to(self.device)
        
        return frames_tensor
    
    async def _send_summary(
        self,
        task_id: str,
        results: List[Dict],
        video_info: Dict
    ):
        """
        Send summary of video analysis
        
        Args:
            task_id: Task identifier
            results: List of frame results
            video_info: Video metadata
        """
        # Calculate statistics
        fake_count = sum(1 for r in results if r['prediction'] == 'fake')
        real_count = len(results) - fake_count
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # Overall prediction
        overall_prediction = 'fake' if fake_count > real_count else 'real'
        overall_confidence = fake_count / len(results) if overall_prediction == 'fake' else real_count / len(results)
        
        summary = {
            'total_frames_analyzed': len(results),
            'fake_frames': fake_count,
            'real_frames': real_count,
            'overall_prediction': overall_prediction,
            'overall_confidence': overall_confidence,
            'average_confidence': avg_confidence,
            'video_duration': video_info['duration'],
            'fps': video_info['fps']
        }
        
        await send_detection_progress(
            task_id=task_id,
            stage='complete',
            progress=100,
            message='Video analysis complete',
            **summary
        )


class AdaptiveFrameSampler:
    """
    Adaptive frame sampling based on content analysis
    """
    
    def __init__(self, base_sample_rate: int = 5):
        """
        Initialize adaptive sampler
        
        Args:
            base_sample_rate: Base frame sampling rate
        """
        self.base_sample_rate = base_sample_rate
        self.prev_frame = None
    
    def should_sample_frame(self, frame: np.ndarray, frame_idx: int) -> bool:
        """
        Determine if frame should be sampled
        
        Args:
            frame: Current frame
            frame_idx: Frame index
            
        Returns:
            Whether to sample this frame
        """
        # Always sample at base rate
        if frame_idx % self.base_sample_rate == 0:
            self.prev_frame = frame
            return True
        
        # Sample if significant change from previous frame
        if self.prev_frame is not None:
            diff = cv2.absdiff(frame, self.prev_frame)
            change_score = np.mean(diff)
            
            # Sample if change is significant
            if change_score > 30:  # Threshold for significant change
                self.prev_frame = frame
                return True
        
        return False


def create_streaming_processor(
    model: torch.nn.Module,
    **kwargs
) -> StreamingVideoProcessor:
    """
    Factory function to create streaming processor
    
    Args:
        model: Detection model
        **kwargs: Additional arguments
        
    Returns:
        Streaming processor instance
    """
    return StreamingVideoProcessor(model, **kwargs)


if __name__ == "__main__":
    # Test streaming processor
    from ..models.detector import EfficientNetDetector
    
    model = EfficientNetDetector()
    processor = create_streaming_processor(model, device='cpu')
    
    print("Streaming processor created successfully")
