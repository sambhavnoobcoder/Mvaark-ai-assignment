#!/usr/bin/env python3
"""
Tests for the ASR Transcription API.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from app.main import app
from app.utils import get_model_path, get_vocab_path

client = TestClient(app)

def test_root():
    """
    Test the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ASR Transcription API is running"}

def test_health():
    """
    Test the health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
@patch("app.main.validate_audio_file", new_callable=AsyncMock)
@patch("app.main.process_audio_file", new_callable=AsyncMock)
@patch("app.main.get_model_path")
@patch("app.main.get_vocab_path")
@patch("app.main.get_model", new_callable=AsyncMock)
async def test_transcribe(mock_get_model, mock_get_vocab_path, mock_get_model_path, mock_process_audio, mock_validate_audio):
    """
    Test the transcribe endpoint.
    """
    # Mock file paths
    mock_get_model_path.return_value = "mock_model_path.onnx"
    mock_get_vocab_path.return_value = "mock_vocab_path.txt"
    
    # Mock the model
    mock_model = AsyncMock()
    mock_model.transcribe.return_value = "नमस्ते दुनिया"
    mock_get_model.return_value = mock_model
    
    # Mock audio processing
    mock_process_audio.return_value = "dummy_audio_data"
    
    # Create a test file
    test_file_path = "test_audio.wav"
    with open(test_file_path, "wb") as f:
        f.write(b"dummy audio data")
    
    # Test the endpoint
    with open(test_file_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("test_audio.wav", f, "audio/wav")}
        )
    
    # Clean up
    os.remove(test_file_path)
    
    # Assertions
    assert response.status_code == 200
    assert response.json() == {"text": "नमस्ते दुनिया"}
    mock_validate_audio.assert_called_once()
    mock_process_audio.assert_called_once()
    mock_model.transcribe.assert_called_once_with("dummy_audio_data") 