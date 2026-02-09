import cv2
import numpy as np
from pathlib import Path
from typing import List


def extract_frames(
    video_path: Path,
    num_frames: int = 32,
    sampling: str = "uniform"
) -> List[np.ndarray]:
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        sampling: Sampling strategy ('uniform', 'random', 'keyframes')
        
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    
    if sampling == "uniform":
        # Uniformly sample frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    elif sampling == "random":
        # Randomly sample frames
        frame_indices = np.random.choice(total_frames, size=num_frames, replace=False)
        frame_indices = np.sort(frame_indices)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    elif sampling == "keyframes":
        # Extract keyframes (simplified - just use uniform for now)
        # In production, use proper keyframe detection
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    
    cap.release()
    
    return frames


def get_video_info(video_path: Path) -> dict:
    """
    Get video metadata
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return info
