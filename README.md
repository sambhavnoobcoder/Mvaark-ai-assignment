# FastAPI-based ASR Application Using NVIDIA NeMo

This project implements an Automatic Speech Recognition (ASR) service using NVIDIA NeMo's Hindi Conformer CTC model, optimized with ONNX for inference.

## Features

- FastAPI endpoint for audio transcription
- ONNX-optimized ASR model for efficient inference
- Docker containerization with multi-stage build for minimal image size
- Support for Hindi speech recognition
- Async processing for handling concurrent requests
- Apple Silicon (M1/M2/M3/M4) compatibility with MPS acceleration
- Advanced memory management to prevent memory leaks
- Comprehensive error handling with detailed error messages
- Health monitoring with memory and performance metrics
- Cross-platform support (AMD64 and ARM64 architectures)
- Automatic hardware acceleration detection (CUDA, CoreML, DirectML)

## Model Specifications

- **Model**: stt_hi_conformer_ctc_medium (Hindi ASR model)
- **Source**: NVIDIA NeMo Catalog
- **Optimization**: ONNX Runtime
- **Input**: Audio files (.wav) of 5-10 seconds duration at 16kHz
- **Output**: Transcribed Hindi text

## Hardware Requirements

- **Minimum**: 2GB RAM, dual-core CPU
- **Recommended**: 4GB RAM, quad-core CPU
- **GPU Support**: NVIDIA GPU with CUDA or Apple Silicon with MPS (optional but recommended)

## Setup and Installation

### Local Development

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the conversion script to prepare the ONNX model:
   ```
   python scripts/convert_to_onnx.py
   ```

5. Start the FastAPI server:
   ```
   uvicorn app.main:app --reload
   ```

### Using Docker

1. Build the Docker image:
   ```
   docker build -t asr-app .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 asr-app
   ```

## API Usage

### Check Health Status

**Endpoint:** `GET /health`

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 123.45,
  "request_count": 42,
  "failure_rate": 0.5,
  "avg_processing_time": 0.78,
  "memory_usage": {
    "rss_mb": 256.5,
    "vms_mb": 512.3,
    "percent": 1.2
  }
}
```

### Transcribe Audio

**Endpoint:** `POST /transcribe`

**Request:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@sample.wav"
```

**Response:**
```json
{
  "text": "नमस्ते, यह एक परीक्षण वाक्य है।",
  "processing_time": 0.8738718032836914
}
```

### Error Responses

**Invalid File Format:**
```json
{
  "detail": "Unsupported file format: mp3. Supported formats: wav",
  "error_type": "ValidationError",
  "suggestion": "Please upload a WAV file instead"
}
```

**File Too Short/Long:**
```json
{
  "detail": "Audio duration too short: 2.50s. Minimum required: 5.0s",
  "error_type": "ValidationError",
  "suggestion": "Please upload an audio file with duration between 5-10 seconds"
}
```

**Server Error:**
```json
{
  "detail": "Transcription error: Failed to process audio",
  "error_type": "ProcessingError",
  "suggestion": "Please try again with a different audio file"
}
```

## Testing

### Unit Tests

The project includes comprehensive unit tests to ensure the API works correctly. The tests cover:

1. **Basic API Health**: Verifies the API is running and healthy
2. **Format Validation**: Tests that the API correctly rejects unsupported file formats (non-WAV)
3. **Duration Validation**: Ensures audio files shorter than the minimum required duration are rejected
4. **Real Transcription**: Tests the actual transcription functionality (Note: requires model to be loaded)

Run the tests with:

```bash
python -m pytest tests/test_api.py -v
```

### Generate Test Audio

You can generate test audio files using the provided scripts:

```bash
# Generate a simple sine wave audio file
python scripts/generate_test_audio.py --duration 5.0 --output test_audio.wav

# Generate a more realistic speech-like sample
python scripts/generate_speech_sample.py --duration 6.0 --output speech_like_sample.wav
```

### Manual API Testing

After starting the server, you can test the API with curl:

```bash
# Check API health
curl -X GET "http://localhost:8000/health"

# Transcribe an audio file
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@test_audio.wav"
```

Or use the provided test script:

```bash
python scripts/test_api.py --audio_file test_audio.wav
```

## Design Considerations

- **Singleton Pattern**: The model is loaded once and reused for all requests
- **Async Processing**: Implemented to handle multiple requests efficiently
- **Input Validation**: Ensures only compatible audio files are processed
- **Error Handling**: Comprehensive error handling for various edge cases
- **MPS Acceleration**: Support for Apple Silicon devices

## License

[Specify License]

## Monitoring and Performance

### Health Monitoring

The `/health` endpoint provides comprehensive metrics about the application:

- `status`: Current service status ("healthy" or "unhealthy")
- `model_loaded`: Whether the ASR model is loaded successfully
- `uptime`: Time in seconds since application start
- `request_count`: Total number of requests processed
- `failure_rate`: Percentage of failed requests
- `avg_processing_time`: Average processing time in seconds
- `memory_usage`: Memory statistics of the application

### Performance Optimizations

The application includes several performance optimizations:

1. **Hardware Acceleration**: Automatically detects and uses the best available hardware accelerator:
   - NVIDIA GPUs via CUDA
   - Apple Silicon via CoreML
   - Windows DirectX via DirectML
   - Falls back to CPU when no accelerator is available

2. **Memory Management**:
   - Automatic garbage collection to prevent memory leaks
   - Background cleanup tasks after processing
   - Memory usage monitoring

3. **Execution Provider Optimization**:
   - ONNX Runtime execution providers are selected based on available hardware
   - Graph optimization is enabled for maximum performance
   - Multi-threading configuration for optimal CPU utilization

4. **Containerization Optimizations**:
   - Multi-stage Docker build to minimize image size
   - Only runtime dependencies in the final image
   - Cross-platform compatibility with multi-architecture support 