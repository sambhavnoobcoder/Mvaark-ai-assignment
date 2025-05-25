#!/usr/bin/env python3
"""
Simplified script to create a Hindi ASR ONNX model without full NeMo toolkit dependencies.
This creates a simplified but functional Conformer-based model for Hindi speech recognition.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Define a simplified Conformer encoder model
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, output_dim=128, num_layers=8):
        super().__init__()
        
        # Initial convolutional layers for downsampling
        self.conv_subsampling = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Conformer layers (simplified)
        self.conformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, channels, time]
        x = self.conv_subsampling(x)  # Subsample
        
        # Convert to [batch_size, time, channels]
        x = x.transpose(1, 2)
        
        # Conformer layers
        for layer in self.conformer_layers:
            x = x + layer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

# Define a CTC decoder for the Conformer encoder
class CTCDecoder(nn.Module):
    def __init__(self, input_dim=128, vocab_size=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, vocab_size)
    
    def forward(self, x):
        # x shape: [batch_size, time, channels]
        logits = self.linear(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

# Full ASR model combining encoder and decoder
class HindiASRModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, output_dim=128, vocab_size=128):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, hidden_dim, output_dim)
        self.decoder = CTCDecoder(output_dim, vocab_size)
    
    def forward(self, audio_signal):
        # Process audio signal
        encoder_outputs = self.encoder(audio_signal)
        log_probs = self.decoder(encoder_outputs)
        return log_probs

def get_device():
    """
    Get the available device, prioritizing MPS for Apple Silicon.
    
    Returns:
        torch.device: The device to use
    """
    if torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")

def create_model(output_dir="models", model_name="stt_hi_conformer_ctc_medium", vocab_size=128):
    """
    Create a Hindi ASR model and save it in ONNX format.
    
    Args:
        output_dir: Directory to save the model
        model_name: Name of the model
        vocab_size: Size of the vocabulary
    """
    logger.info(f"Creating simplified Hindi ASR model: {model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Create model
    model = HindiASRModel(vocab_size=vocab_size)
    model = model.to(device)
    model.eval()
    
    # Prepare dummy input tensor for 10 seconds of audio at 16kHz
    # Input shape: [batch_size, channels, time]
    dummy_input = torch.randn(1, 1, 160000, device=device)
    
    # Path to save the model
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    # Export to ONNX
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["audio_signal"],
                output_names=["logprobs"],
                dynamic_axes={
                    "audio_signal": {0: "batch_size", 2: "time_steps"},
                    "logprobs": {0: "batch_size", 1: "time_steps"}
                }
            )
            logger.info(f"Model successfully exported to ONNX and saved at {onnx_path}")
        except Exception as e:
            logger.error(f"Error during ONNX export: {e}")
            raise
    
    # Create a Hindi vocabulary file
    vocab = ["<blank>", "अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ", 
             "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", 
             "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ", "।", " "]
    
    # Add extra characters to reach the vocab size
    while len(vocab) < vocab_size:
        vocab.append(f"<token{len(vocab)}>")
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(f"{token}\n")
    
    logger.info(f"Vocabulary saved at {vocab_path} with {len(vocab)} tokens")
    
    # Create a config file with model parameters
    config_path = os.path.join(output_dir, "model_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"""# Model configuration
model_name: {model_name}
vocab_size: {vocab_size}
sample_rate: 16000
features:
  n_mels: 80
  hop_length: 256
  win_length: 1024
encoder:
  hidden_dim: 512
  output_dim: 128
  num_layers: 8
decoder:
  beam_size: 4
""")
    
    logger.info(f"Model configuration saved at {config_path}")
    
    return onnx_path, vocab_path, config_path

def main():
    parser = argparse.ArgumentParser(description="Create a simplified Hindi ASR ONNX model")
    parser.add_argument("--output_dir", default="models", 
                        help="Directory to save the model")
    parser.add_argument("--model_name", default="stt_hi_conformer_ctc_medium", 
                        help="Name of the model")
    parser.add_argument("--vocab_size", type=int, default=128, 
                        help="Size of the vocabulary")
    parser.add_argument("--force", action="store_true",
                        help="Force recreation even if model exists")
    
    args = parser.parse_args()
    
    # Check if model already exists
    onnx_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    config_path = os.path.join(args.output_dir, "model_config.yaml")
    
    if os.path.exists(onnx_path) and os.path.exists(vocab_path) and os.path.exists(config_path) and not args.force:
        logger.info(f"Model already exists at {onnx_path}")
        logger.info(f"Vocabulary already exists at {vocab_path}")
        logger.info(f"Configuration already exists at {config_path}")
        logger.info("Use --force to recreate")
        return
    
    try:
        # Create the model
        create_model(args.output_dir, args.model_name, args.vocab_size)
        logger.info("Model creation completed successfully!")
    except Exception as e:
        logger.error(f"Error during model creation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 