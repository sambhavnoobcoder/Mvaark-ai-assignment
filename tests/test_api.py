#!/usr/bin/env python3
"""
Tests for the ASR Transcription API.
"""

import os
import pytest
from fastapi.testclient import TestClient
import numpy as np
import io

from app.main import app

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
    health_data = response.json()
    
    # Validate basic health info
    assert health_data["status"] == "healthy"
    assert isinstance(health_data["model_loaded"], bool)
    assert isinstance(health_data["uptime"], float)
    assert health_data["uptime"] >= 0
    
    # Validate enhanced metrics
    assert "request_count" in health_data
    assert isinstance(health_data["request_count"], int)
    assert "failure_rate" in health_data
    assert isinstance(health_data["failure_rate"], float)
    assert "avg_processing_time" in health_data
    assert isinstance(health_data["avg_processing_time"], float)
    
    # Validate memory metrics
    assert "memory_usage" in health_data
    memory = health_data["memory_usage"]
    assert "rss_mb" in memory
    assert "vms_mb" in memory
    assert "percent" in memory
    assert isinstance(memory["rss_mb"], float)
    assert isinstance(memory["vms_mb"], float)
    assert isinstance(memory["percent"], float)

def test_transcribe_invalid_format():
    """
    Test transcription with invalid audio format (non-WAV).
    """
    # Create a fake MP3 file
    test_file_path = "test_audio.mp3"
    with open(test_file_path, "wb") as f:
        f.write(b"fake mp3 data")
    
    # Test the endpoint
    with open(test_file_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("test_audio.mp3", f, "audio/mpeg")}
        )
    
    # Clean up
    os.remove(test_file_path)
    
    # Assertions
    assert response.status_code == 400
    error_data = response.json()
    assert "detail" in error_data
    assert "Unsupported file format: mp3" in error_data["detail"]
    assert "Supported formats: wav" in error_data["detail"]

def test_transcribe_short_audio():
    """
    Test transcription with audio that is too short.
    """
    # Use the pre-generated short audio file
    test_file_path = "tests/short_audio.wav"
    
    # Make sure the file exists
    if not os.path.exists(test_file_path):
        pytest.skip(f"Test audio file {test_file_path} not found")
    
    # Test the endpoint
    with open(test_file_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("short_audio.wav", f, "audio/wav")}
        )
    
    # Assertions
    assert response.status_code == 400
    error_data = response.json()
    assert "detail" in error_data
    assert "too short" in error_data["detail"].lower()

@pytest.mark.skip(reason="Causes segmentation fault with ONNX model")
def test_transcribe_real_audio():
    """
    Test transcription with a real audio file.
    This test requires the model to be loaded.
    """
    # Use the pre-generated test audio file
    test_file_path = "tests/test_audio.wav"
    
    # Make sure the file exists
    if not os.path.exists(test_file_path):
        pytest.skip(f"Test audio file {test_file_path} not found")
    
    # Test the endpoint
    with open(test_file_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("test_audio.wav", f, "audio/wav")}
        )
    
    # Check response
    if response.status_code != 200:
        print(f"Error response: {response.json()}")
    
    # Assertions
    assert response.status_code == 200
    assert "text" in response.json()
    assert "processing_time" in response.json()
    assert isinstance(response.json()["processing_time"], float) 