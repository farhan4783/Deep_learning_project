from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from pathlib import Path

from app.api import detection, health
from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="DeepGuard API",
    description="AI-Powered Deepfake Detection System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(detection.router, prefix="/api/v1/detect", tags=["Detection"])

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    print("🚀 Starting DeepGuard API...")
    print(f"📊 Loading models from: {settings.MODEL_PATH}")
    # Models will be loaded lazily on first request to save memory
    print("✅ API ready to accept requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    print("👋 Shutting down DeepGuard API...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DeepGuard API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
