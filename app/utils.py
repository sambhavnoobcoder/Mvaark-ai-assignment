#!/usr/bin/env python3
"""
Utility functions for audio processing and validation.
"""

import os
import io
import wave
import numpy as np
import librosa
import soundfile as sf
from fastapi import UploadFile, HTTPException
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
                
                logger.info(f"Audio file: {file.filename}, Duration: {duration:.2f}s, Sample rate: {rate}Hz")
                
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
                    logger.warning(f"Audio sample rate is {rate}Hz, will be resampled to {SAMPLE_RATE}Hz")
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
    
    try:
        # Convert to numpy array
        with io.BytesIO(content) as buffer:
            # Try using soundfile first (better quality)
            try:
                audio, sample_rate = sf.read(buffer)
                logger.info(f"Audio loaded with soundfile: shape={audio.shape}, sr={sample_rate}")
            except Exception as e:
                # Fall back to librosa
                logger.warning(f"Soundfile failed: {e}, falling back to librosa")
                buffer.seek(0)
                audio, sample_rate = librosa.load(buffer, sr=None)
                logger.info(f"Audio loaded with librosa: shape={audio.shape}, sr={sample_rate}")
            
            # Make sure audio is mono
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
                logger.info(f"Converted stereo to mono: shape={audio.shape}")
            
            # Resample if necessary
            if sample_rate != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
                logger.info(f"Resampled from {sample_rate}Hz to {SAMPLE_RATE}Hz: shape={audio.shape}")
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Handle length
            if len(audio) > MAX_DURATION * SAMPLE_RATE:
                logger.warning(f"Trimming audio to {MAX_DURATION}s")
                audio = audio[:int(MAX_DURATION * SAMPLE_RATE)]
            
            # Pad if needed
            if len(audio) < MIN_DURATION * SAMPLE_RATE:
                logger.warning(f"Padding audio to {MIN_DURATION}s")
                padding = np.zeros(int(MIN_DURATION * SAMPLE_RATE) - len(audio))
                audio = np.concatenate([audio, padding])
        
        return audio
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )
    finally:
        # Reset file position
        await file.seek(0)

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
        logger.error(f"Model file not found: {model_path}")
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
        logger.error(f"Vocabulary file not found: {vocab_path}")
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    return vocab_path 