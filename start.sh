#!/bin/bash

# Default port if not set by Railway
DEFAULT_PORT=8000

# Use default port if PORT is not set or is $PORT literal
if [ -z "${PORT}" ] || [ "${PORT}" = "\$PORT" ]; then
  export PORT=$DEFAULT_PORT
  echo "PORT environment variable not set. Using default: $PORT"
else
  echo "Using PORT from environment: $PORT"
fi

# Print debug information
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Files in directory:"
ls -la

# Check for required dependencies
echo "Checking OpenCV installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || echo "OpenCV import failed"

# Start the application with the resolved port
echo "Starting gunicorn on port $PORT..."
exec gunicorn enhanced:app --log-file=- --log-level=info --bind=0.0.0.0:$PORT --timeout 120 --workers=2 --threads=2 