from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DeepGuard API",
        "version": "1.0.0"
    }

@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # TODO: Check if models are loaded
    return {
        "status": "ready",
        "models_loaded": True
    }
