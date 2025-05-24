#!/usr/bin/env python3
"""
ASR model implementation using ONNX Runtime.
"""

import os
import numpy as np
import onnxruntime as ort
import logging
import asyncio
import gc
import time
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Singleton model instance
_model_instance = None
_model_lock = asyncio.Lock()

def get_execution_providers() -> List[str]:
    """
    Get available execution providers for ONNX Runtime.
    
    Returns:
        List[str]: List of available execution providers
    """
    available_providers = ort.get_available_providers()
    preferred_providers = []
    
    # Check for CUDA
    if 'CUDAExecutionProvider' in available_providers:
        preferred_providers.append('CUDAExecutionProvider')
        logger.info("Using CUDA for model inference")
    # Check for CoreML (macOS)
    elif 'CoreMLExecutionProvider' in available_providers:
        preferred_providers.append('CoreMLExecutionProvider')
        logger.info("Using CoreML for model inference")
    # Check for DirectML (Windows)
    elif 'DmlExecutionProvider' in available_providers:
        preferred_providers.append('DmlExecutionProvider')
        logger.info("Using DirectML for model inference")
    # Fall back to CPU
    else:
        logger.info("Using CPU for model inference")
    
    # Always add CPU as fallback
    preferred_providers.append('CPUExecutionProvider')
    
    return preferred_providers

class ASRModel:
    """
    ASR model implementation using ONNX Runtime.
    """
    
    def __init__(self, model_path: str, vocab_path: str):
        """
        Initialize the ASR model.
        
        Args:
            model_path: Path to the ONNX model file
            vocab_path: Path to the vocabulary file
        
        Raises:
            FileNotFoundError: If model or vocabulary file does not exist
            RuntimeError: If model initialization fails
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        # Load vocabulary
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = [line.strip() for line in f]
            logger.info(f"Loaded vocabulary with {len(self.vocab)} tokens")
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise
        
        # Initialize ONNX Runtime session with optimized settings
        try:
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
            
            # Set execution providers
            execution_providers = get_execution_providers()
            
            # Initialize session with optimized settings
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=execution_providers
            )
            
            # Get model metadata
            model_inputs = self.session.get_inputs()
            model_outputs = self.session.get_outputs()
            
            logger.info(f"Model loaded successfully. Inputs: {[x.name for x in model_inputs]}, Outputs: {[x.name for x in model_outputs]}")
            
            # Store input and output names
            self.input_name = model_inputs[0].name
            self.output_name = model_outputs[0].name
            
            # Record successful initialization
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            self.initialized = False
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")
    
    def _decode_output(self, output: np.ndarray) -> str:
        """
        Decode model output to text.
        
        Args:
            output: Model output tensor
        
        Returns:
            str: Decoded text
        """
        try:
            # Get the most likely token for each time step
            tokens = np.argmax(output, axis=-1)
            
            # Convert token IDs to characters and join
            chars = []
            previous = -1  # for CTC decoding
            
            for token in tokens[0]:
                # Skip if same as previous (CTC collapsing)
                if token == previous:
                    continue
                # Skip blank token (index 0)
                if token > 0:
                    chars.append(self.vocab[token])
                previous = token
            
            # Join characters to form text
            text = "".join(chars)
            return text
        except Exception as e:
            logger.error(f"Error decoding output: {e}")
            return ""
    
    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array
        
        Returns:
            str: Transcribed text
        
        Raises:
            RuntimeError: If transcription fails
        """
        try:
            if not self.initialized:
                raise RuntimeError("Model not initialized properly")
            
            # Ensure audio is float32 (expected by the model)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Reshape for model input [batch_size, sequence_length]
            audio = audio.reshape(1, -1)
            
            # Run inference with error handling
            try:
                # Create input dictionary
                inputs = {self.input_name: audio}
                
                # Protect against memory issues with large inputs
                if audio.size > 1_000_000:  # If more than ~1M elements
                    logger.warning(f"Large audio input detected: {audio.shape}")
                
                # Run inference
                start_time = time.time()
                output = self.session.run([self.output_name], inputs)[0]
                inference_time = time.time() - start_time
                
                logger.info(f"Inference completed in {inference_time:.3f}s")
                
                # Force garbage collection to prevent memory leaks
                del inputs
                gc.collect()
                
                # Decode output to text
                text = self._decode_output(output)
                
                logger.info(f"Transcription result: {text}")
                return text
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise RuntimeError(f"Inference error: {e}")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

async def get_model(model_path: str, vocab_path: str) -> ASRModel:
    """
    Get the ASR model instance (singleton pattern).
    
    Args:
        model_path: Path to the ONNX model file
        vocab_path: Path to the vocabulary file
    
    Returns:
        ASRModel: ASR model instance
    
    Raises:
        FileNotFoundError: If model or vocabulary file does not exist
        RuntimeError: If model initialization fails
    """
    global _model_instance
    
    # Check if model files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    # If model is already loaded, return it
    if _model_instance is not None:
        return _model_instance
    
    # Acquire lock to prevent multiple initialization
    async with _model_lock:
        # Double-check if model was loaded while waiting for lock
        if _model_instance is not None:
            return _model_instance
        
        try:
            # Clear memory before loading model
            gc.collect()
            
            # Initialize model
            logger.info(f"Initializing model from {model_path}")
            start_time = time.time()
            _model_instance = ASRModel(model_path, vocab_path)
            load_time = time.time() - start_time
            
            logger.info(f"Model initialized in {load_time:.2f}s")
            return _model_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Reset instance on error
            _model_instance = None
            raise 