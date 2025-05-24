FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout=1000 --retries=10 pip setuptools wheel && \
    pip install --no-cache-dir --timeout=1000 --retries=10 -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create model directory
RUN mkdir -p models

# Set environment variables
ENV MODEL_NAME=stt_hi_conformer_ctc_medium
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app

# Download and convert the model
RUN python scripts/convert_to_onnx.py --model_name ${MODEL_NAME} --output_dir ${MODEL_DIR}

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 