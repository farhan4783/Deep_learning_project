from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import aiofiles
from pathlib import Path
import uuid

from app.services.detection_service import DetectionService
from app.core.config import settings
from app.schemas.detection import DetectionResponse, BatchDetectionResponse

router = APIRouter()
detection_service = DetectionService()

@router.post("/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    explain: bool = Form(default=True)
):
    """
    Detect deepfake in an uploaded image
    
    Args:
        file: Image file to analyze
        explain: Whether to generate Grad-CAM explanation
        
    Returns:
        Detection results with confidence scores and optional heatmap
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}"
        )
    
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    temp_path = settings.TEMP_DIR / f"{file_id}{file_ext}"
    
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Perform detection
        result = await detection_service.detect_image(
            image_path=temp_path,
            explain=explain
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary file
        if temp_path.exists():
            temp_path.unlink()

@router.post("/video", response_model=DetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    frame_sampling: str = Form(default="uniform"),
    num_frames: int = Form(default=32),
    explain: bool = Form(default=False)
):
    """
    Detect deepfake in an uploaded video
    
    Args:
        file: Video file to analyze
        frame_sampling: Sampling strategy ('uniform', 'random', 'keyframes')
        num_frames: Number of frames to analyze
        explain: Whether to generate explanations for key frames
        
    Returns:
        Aggregated detection results across frames
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}"
        )
    
    # Validate num_frames
    if num_frames > settings.MAX_VIDEO_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"num_frames exceeds maximum of {settings.MAX_VIDEO_FRAMES}"
        )
    
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    temp_path = settings.TEMP_DIR / f"{file_id}{file_ext}"
    
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Perform detection
        result = await detection_service.detect_video(
            video_path=temp_path,
            frame_sampling=frame_sampling,
            num_frames=num_frames,
            explain=explain
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary file
        if temp_path.exists():
            temp_path.unlink()

@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: List[UploadFile] = File(...),
    explain: bool = Form(default=False)
):
    """
    Detect deepfakes in multiple files
    
    Args:
        files: List of image/video files to analyze
        explain: Whether to generate explanations
        
    Returns:
        List of detection results for each file
    """
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        file_id = str(uuid.uuid4())
        temp_path = settings.TEMP_DIR / f"{file_id}{file_ext}"
        
        try:
            # Save file
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            # Detect based on file type
            if file_ext in settings.ALLOWED_IMAGE_EXTENSIONS:
                result = await detection_service.detect_image(
                    image_path=temp_path,
                    explain=explain
                )
            elif file_ext in settings.ALLOWED_VIDEO_EXTENSIONS:
                result = await detection_service.detect_video(
                    video_path=temp_path,
                    frame_sampling="uniform",
                    num_frames=16,  # Reduced for batch processing
                    explain=False  # Disabled for batch
                )
            else:
                continue
            
            results.append({
                "filename": file.filename,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    return {"results": results}
