#!/bin/bash

# Detect Python command
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Neither 'python' nor 'python3' found in PATH"
        exit 1
    fi
fi

echo "Using Python command: $PYTHON_CMD"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH=$PWD
export MODEL_NAME="stt_hi_conformer_ctc_medium"
export MODEL_DIR="models"

# Check if model exists
if [ ! -f "$MODEL_DIR/$MODEL_NAME.onnx" ]; then
    echo "Model not found! Running conversion script..."
    $PYTHON_CMD scripts/convert_to_onnx.py --model_name $MODEL_NAME --output_dir $MODEL_DIR
fi

# Run tests
echo "Running tests..."
$PYTHON_CMD -m pytest tests/test_api.py -v

# Start the API server
echo "Starting API server..."
$PYTHON_CMD -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Alternative: Use Python module format
# $PYTHON_CMD -m app.main 