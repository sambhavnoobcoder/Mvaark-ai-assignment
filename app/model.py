#!/usr/bin/env python3
"""
Model handling code for ONNX inference.
"""

import os
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any

class ASRModel:
    """
    ASR model class for ONNX inference.
    """
    
    def __init__(self, model_path: str, vocab_path: str):
        """
        Initialize the ASR model.
        
        Args:
            model_path: Path to the ONNX model file
            vocab_path: Path to the vocabulary file
        """
        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load the vocabulary
        self.vocab = self._load_vocabulary(vocab_path)
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Vocabulary loaded successfully from {vocab_path}")
    
    def _load_vocabulary(self, vocab_path: str) -> List[str]:
        """
        Load the vocabulary from a file.
        
        Args:
            vocab_path: Path to the vocabulary file
        
        Returns:
            List[str]: The vocabulary
        """
        with open(vocab_path, "r") as f:
            vocab = [line.strip() for line in f]
        
        return vocab
    
    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio using the ASR model.
        
        Args:
            audio: Audio data as a numpy array
        
        Returns:
            str: Transcribed text
        """
        # Reshape audio for model input
        # Expected shape: [batch_size, channels, time]
        audio = audio.reshape(1, 1, -1)
        
        # For demo purposes, we'll just return a fixed Hindi text
        # In a real implementation, we would use the model for inference
        if isinstance(audio, np.ndarray):
            # Run inference
            try:
                outputs = self.session.run(
                    [self.output_name], 
                    {self.input_name: audio.astype(np.float32)}
                )
                
                # Get predictions (dummy implementation)
                # In a real implementation, we would process the model output properly
                return "नमस्ते दुनिया" # "Hello World" in Hindi
            except Exception as e:
                print(f"Inference error: {e}")
                return "मॉडल अनुमान त्रुटि" # "Model inference error" in Hindi
        else:
            return "अमान्य ऑडियो डेटा" # "Invalid audio data" in Hindi

# Singleton model instance
_model_instance = None

async def get_model(model_path: str, vocab_path: str) -> ASRModel:
    """
    Get the ASR model instance.
    
    Args:
        model_path: Path to the ONNX model file
        vocab_path: Path to the vocabulary file
    
    Returns:
        ASRModel: The ASR model instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = ASRModel(model_path, vocab_path)
    
    return _model_instance 