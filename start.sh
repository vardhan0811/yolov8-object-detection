#!/bin/bash

# Exit on error
set -e

echo "=========================================================="
echo "Railway Deployment Script for YOLOv8 Object Detection"
echo "=========================================================="

# Hardcoded default port
DEFAULT_PORT=8000

echo "Environment variables:"
env | grep -v PASSWORD | grep -v SECRET

# Check PORT variable
if [ -z "${PORT}" ] || [ "${PORT}" = "\$PORT" ]; then
  export PORT=$DEFAULT_PORT
  echo "WARNING: PORT environment variable not set or invalid. Using default: $PORT"
else
  echo "Using PORT from environment: $PORT"
fi

# Create required directories
mkdir -p static/uploads
mkdir -p static/results

# Print debug information
echo "=========================================================="
echo "System Information:"
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Directory structure:"
find . -maxdepth 2 -type d | sort

# Check if we're using the Dockerfile
if [ -f "Dockerfile" ]; then
  echo "Dockerfile exists - this should be used for the build."
fi

# Check for template files
echo "=========================================================="
echo "Checking for required files:"
if [ -d "templates" ]; then
  echo "✅ Templates directory found"
  ls -la templates
elif [ -d "FlaskTutorial_YOLOv8_Web_PPE/templates" ]; then
  echo "✅ Templates directory found in FlaskTutorial_YOLOv8_Web_PPE"
  ls -la FlaskTutorial_YOLOv8_Web_PPE/templates
else
  echo "❌ Templates directory not found"
fi

# Check for model files
echo "=========================================================="
echo "Checking for model files:"
find . -name "*.pt" -type f | while read -r model; do
  size=$(du -h "$model" | cut -f1)
  echo "✅ Found model: $model ($size)"
done

# Check for required dependencies
echo "=========================================================="
echo "Checking dependencies:"
echo "- Checking OpenCV installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || echo "❌ OpenCV import failed"
echo "- Checking Flask installation..."
python -c "import flask; print(f'Flask version: {flask.__version__}')" || echo "❌ Flask import failed"
echo "- Checking Ultralytics installation..."
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')" || echo "❌ Ultralytics import failed"

echo "=========================================================="
echo "Starting gunicorn with railway_app.py on port $PORT..."
echo "Command: gunicorn railway_app:app --log-file=- --log-level=info --bind=0.0.0.0:$PORT --timeout 120 --workers=1 --threads=4"
echo "=========================================================="

# Execute gunicorn with the guaranteed port - using the simplified railway_app.py
exec gunicorn railway_app:app --log-file=- --log-level=info --bind=0.0.0.0:$PORT --timeout 120 --workers=1 --threads=4 