# FastAPI-based ASR Application Using NVIDIA NeMo

This project implements an Automatic Speech Recognition (ASR) service using NVIDIA NeMo's Hindi Conformer CTC model, optimized with ONNX for inference.

## Features

- FastAPI endpoint for audio transcription
- ONNX-optimized ASR model for efficient inference
- Docker containerization for easy deployment
- Support for Hindi speech recognition

## Setup and Installation

### Local Development

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the conversion script to prepare the ONNX model:
   ```
   python scripts/convert_to_onnx.py
   ```

4. Start the FastAPI server:
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
  "text": "transcribed text goes here"
}
```

## Testing

### Generate Test Audio

You can generate a test audio file using the provided script:

```bash
python scripts/generate_test_audio.py --duration 5.0 --output test_audio.wav
```

This will create a 5-second sine wave audio file at 16kHz.

### Test the API

After starting the server, you can test the API using the provided script:

```bash
python scripts/test_api.py --audio_file test_audio.wav
```

### Run Unit Tests

```bash
pytest
```

## Design Considerations

- The model is optimized using ONNX for faster inference
- Async processing is implemented to handle multiple requests efficiently
- Input validation ensures only compatible audio files are processed

## License

[Specify License] 