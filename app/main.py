#!/usr/bin/env python3
"""
FastAPI application for ASR transcription.
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from app.utils import validate_audio_file, process_audio_file, get_model_path, get_vocab_path
from app.model import get_model, ASRModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ASR Transcription API",
    description="API for transcribing audio using NVIDIA NeMo ASR model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "stt_hi_conformer_ctc_medium")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Response model
class TranscriptionResponse(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup.
    """
    logger.info("Starting up the application")
    try:
        model_path = get_model_path(MODEL_NAME, MODEL_DIR)
        vocab_path = get_vocab_path(MODEL_NAME, MODEL_DIR)
        
        # Initialize the model
        await get_model(model_path, vocab_path)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        # We don't raise an exception here to allow the app to start
        # The error will be raised when the model is accessed

@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": "ASR Transcription API is running"}

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

async def get_asr_model():
    """
    Get the ASR model instance.
    """
    try:
        model_path = get_model_path(MODEL_NAME, MODEL_DIR)
        vocab_path = get_vocab_path(MODEL_NAME, MODEL_DIR)
        
        model = await get_model(model_path, vocab_path)
        return model
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load ASR model: {str(e)}"
        )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio_file: UploadFile = File(...),
    model: ASRModel = Depends(get_asr_model)
):
    """
    Transcribe audio file.
    
    Args:
        audio_file: Audio file to transcribe
        model: ASR model instance
    
    Returns:
        TranscriptionResponse: Transcription result
    """
    try:
        # Validate audio file
        await validate_audio_file(audio_file)
        
        # Process audio file
        audio = await process_audio_file(audio_file)
        
        # Transcribe audio
        text = await model.transcribe(audio)
        
        return TranscriptionResponse(text=text)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 