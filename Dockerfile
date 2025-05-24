# Stage 1: Model conversion
FROM --platform=$BUILDPLATFORM python:3.9-slim AS builder

WORKDIR /build

# Install system dependencies for model conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout=1000 --retries=10 pip setuptools wheel && \
    pip install --no-cache-dir --timeout=1000 --retries=10 -r requirements.txt

# Create models directory and copy conversion scripts
RUN mkdir -p models
COPY scripts/ ./scripts/

# Set environment variables
ENV MODEL_NAME=stt_hi_conformer_ctc_medium
ENV MODEL_DIR=/build/models
ENV PYTHONPATH=/build
ENV PYTHONUNBUFFERED=1

# Download and convert the model
RUN python scripts/convert_to_onnx.py --model_name ${MODEL_NAME} --output_dir ${MODEL_DIR}

# Stage 2: Runtime image
FROM --platform=$TARGETPLATFORM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install runtime dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    --no-deps \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    numpy \
    onnxruntime \
    soundfile \
    librosa \
    pytest \
    pytest-asyncio

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pytest.ini .

# Copy the converted model from the builder stage
COPY --from=builder /build/models/ ./models/

# Set environment variables
ENV MODEL_NAME=stt_hi_conformer_ctc_medium
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create audio samples directory
RUN mkdir -p audio_samples

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 