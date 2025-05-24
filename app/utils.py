#!/usr/bin/env python3
"""
Utility functions for audio processing and validation.
"""

import os
import io
import wave
import numpy as np
import librosa
from fastapi import UploadFile, HTTPException

# Constants
SAMPLE_RATE = 16000  # 16kHz
MIN_DURATION = 5.0  # seconds
MAX_DURATION = 10.0  # seconds
SUPPORTED_FORMATS = ["wav"]

async def validate_audio_file(file: UploadFile) -> None:
    """
    Validate the uploaded audio file.
    
    Args:
        file: The uploaded audio file
    
    Raises:
        HTTPException: If the file is invalid
    """
    # Check file format
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Read file content
    content = await file.read()
    
    try:
        # Check audio duration
        with io.BytesIO(content) as buffer:
            with wave.open(buffer, "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / rate
                
                if duration < MIN_DURATION:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Audio duration too short: {duration:.2f}s. Minimum required: {MIN_DURATION}s"
                    )
                
                if duration > MAX_DURATION:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Audio duration too long: {duration:.2f}s. Maximum allowed: {MAX_DURATION}s"
                    )
                
                # Check sample rate
                if rate != SAMPLE_RATE:
                    # This is just a warning, we'll resample later
                    print(f"Warning: Audio sample rate is {rate}Hz, will be resampled to {SAMPLE_RATE}Hz")
    except wave.Error:
        raise HTTPException(
            status_code=400,
            detail="Invalid WAV file format"
        )
    
    # Reset file position for future reading
    await file.seek(0)

async def process_audio_file(file: UploadFile) -> np.ndarray:
    """
    Process the audio file and convert it to the format required by the model.
    
    Args:
        file: The uploaded audio file
    
    Returns:
        numpy.ndarray: The processed audio data
    """
    # Read file content
    content = await file.read()
    
    # Convert to numpy array
    with io.BytesIO(content) as buffer:
        audio, sample_rate = librosa.load(buffer, sr=None)
    
    # Resample if necessary
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    
    # Reset file position
    await file.seek(0)
    
    return audio

def get_model_path(model_name="stt_hi_conformer_ctc_medium", model_dir="models"):
    """
    Get the path to the ONNX model file.
    
    Args:
        model_name: Name of the model
        model_dir: Directory containing the model
    
    Returns:
        str: Path to the ONNX model file
    
    Raises:
        FileNotFoundError: If the model file does not exist
    """
    model_path = os.path.join(model_dir, f"{model_name}.onnx")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path

def get_vocab_path(model_name="stt_hi_conformer_ctc_medium", model_dir="models"):
    """
    Get the path to the vocabulary file.
    
    Args:
        model_name: Name of the model
        model_dir: Directory containing the model
    
    Returns:
        str: Path to the vocabulary file
    
    Raises:
        FileNotFoundError: If the vocabulary file does not exist
    """
    vocab_path = os.path.join(model_dir, "vocab.txt")
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    return vocab_path 