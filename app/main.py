#!/usr/bin/env python3
"""
FastAPI application for ASR transcription.
"""

import os
import time
import sys
import gc
import traceback
import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
from typing import Optional

from app.utils import validate_audio_file, process_audio_file, get_model_path, get_vocab_path
from app.model import get_model, ASRModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "stt_hi_conformer_ctc_medium")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# System metrics
STARTUP_TIME = time.time()
REQUEST_COUNT = 0
FAILURE_COUNT = 0
AVG_PROCESSING_TIME = 0

# Response models
class TranscriptionResponse(BaseModel):
    text: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float
    request_count: int
    failure_rate: float
    avg_processing_time: float
    memory_usage: dict

class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
    suggestion: Optional[str] = None

# Application state
class AppState:
    def __init__(self):
        self.start_time = time.time()
        self.model_loaded = False
        self.request_count = 0
        self.failure_count = 0
        self.total_processing_time = 0

app_state = AppState()

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),
        "vms_mb": memory_info.vms / (1024 * 1024),
        "percent": process.memory_percent()
    }

def cleanup_memory():
    """Perform memory cleanup to prevent leaks."""
    gc.collect()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the model on startup.
    """
    logger.info("Starting up the application")
    try:
        model_path = get_model_path(MODEL_NAME, MODEL_DIR)
        vocab_path = get_vocab_path(MODEL_NAME, MODEL_DIR)
        
        # Initialize the model
        await get_model(model_path, vocab_path)
        app_state.model_loaded = True
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(traceback.format_exc())
        # We don't raise an exception here to allow the app to start
        # The error will be raised when the model is accessed
    
    yield
    
    logger.info("Shutting down the application")
    # Perform cleanup
    cleanup_memory()

# Create FastAPI app
app = FastAPI(
    title="ASR Transcription API",
    description="API for transcribing audio using NVIDIA NeMo ASR model",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time to response headers and handle exceptions.
    """
    start_time = time.time()
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Update metrics for successful requests
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Update app state for non-health requests (to avoid skewing metrics)
        if not request.url.path.endswith("/health"):
            app_state.request_count += 1
            app_state.total_processing_time += process_time
        
        return response
        
    except Exception as e:
        # Log and handle any unhandled exceptions
        process_time = time.time() - start_time
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update metrics for failed requests
        if not request.url.path.endswith("/health"):
            app_state.request_count += 1
            app_state.failure_count += 1
            app_state.total_processing_time += process_time
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal server error: {str(e)}",
                "error_type": type(e).__name__,
                "suggestion": "Please try again later or contact support"
            }
        )

@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint.
    """
    return {"message": "ASR Transcription API is running"}

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    """
    uptime = time.time() - app_state.start_time
    
    # Calculate metrics
    failure_rate = 0
    avg_processing_time = 0
    
    if app_state.request_count > 0:
        failure_rate = (app_state.failure_count / app_state.request_count) * 100
        avg_processing_time = app_state.total_processing_time / app_state.request_count
    
    # Get memory usage
    memory_usage = get_memory_usage()
    
    return HealthResponse(
        status="healthy",
        model_loaded=app_state.model_loaded,
        uptime=uptime,
        request_count=app_state.request_count,
        failure_rate=failure_rate,
        avg_processing_time=avg_processing_time,
        memory_usage=memory_usage
    )

async def get_asr_model():
    """
    Get the ASR model instance.
    """
    try:
        model_path = get_model_path(MODEL_NAME, MODEL_DIR)
        vocab_path = get_vocab_path(MODEL_NAME, MODEL_DIR)
        
        model = await get_model(model_path, vocab_path)
        # Update model loaded status when model is successfully loaded
        app_state.model_loaded = True
        return model
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        logger.error(traceback.format_exc())
        app_state.model_loaded = False
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load ASR model: {str(e)}",
            headers={"X-Error-Type": "ModelLoadError"}
        )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio_file: UploadFile = File(...),
    model: ASRModel = Depends(get_asr_model),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Transcribe audio file.
    
    Args:
        audio_file: Audio file to transcribe
        model: ASR model instance
        background_tasks: Background tasks runner
    
    Returns:
        TranscriptionResponse: Transcription result
    """
    start_time = time.time()
    
    try:
        # Log request information
        logger.info(f"Received transcription request for file: {audio_file.filename}")
        
        # Validate audio file
        await validate_audio_file(audio_file)
        
        # Process audio file
        audio = await process_audio_file(audio_file)
        
        # Transcribe audio
        text = await model.transcribe(audio)
        
        # Schedule memory cleanup
        background_tasks.add_task(cleanup_memory)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Transcription completed in {processing_time:.3f}s")
        
        return TranscriptionResponse(text=text, processing_time=processing_time)
    except HTTPException:
        # Re-raise HTTP exceptions
        app_state.failure_count += 1
        raise
    except Exception as e:
        # Log detailed error
        logger.error(f"Transcription error: {e}")
        logger.error(traceback.format_exc())
        
        # Update failure count
        app_state.failure_count += 1
        
        # Schedule memory cleanup
        background_tasks.add_task(cleanup_memory)
        
        # Provide more helpful error
        error_type = type(e).__name__
        suggestion = "Please check your audio file and try again"
        
        if "memory" in str(e).lower():
            suggestion = "The server is experiencing high memory usage. Please try again later."
        elif "model" in str(e).lower():
            suggestion = "There was an issue with the ASR model. Please try again later."
        
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Transcription error: {str(e)}",
                "error_type": error_type,
                "suggestion": suggestion
            }
        )

# Add exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions.
    """
    if isinstance(exc.detail, dict) and "message" in exc.detail:
        # Handle structured error responses
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail["message"],
                "error_type": exc.detail.get("error_type"),
                "suggestion": exc.detail.get("suggestion")
            }
        )
    else:
        # Handle simple error responses
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc.detail)}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Default configuration
    host = "0.0.0.0"
    port = 8000
    workers = 1
    
    # Get configuration from environment
    host = os.getenv("HOST", host)
    port = int(os.getenv("PORT", port))
    workers = int(os.getenv("WORKERS", workers))
    
    logger.info(f"Starting server with {workers} workers on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, workers=workers) 