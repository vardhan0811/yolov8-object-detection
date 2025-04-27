FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    ffmpeg \
    libavcodec-extra \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p static/files

# Set environment variables
ENV RAILWAY_ENVIRONMENT=true \
    YOLO_DEVICE=cpu \
    YOLO_VERBOSE=False \
    FLASK_ENV=production \
    MODEL_CACHE_ENABLED=true \
    DISABLE_WEBCAM=true

# Create an entrypoint script to ensure PORT is set
RUN echo '#!/bin/bash\n\
# Use default port 8000 if PORT is not set or is $PORT literal\n\
if [ "$PORT" = "" ] || [ "$PORT" = "\$PORT" ]; then\n\
  export PORT=8000\n\
  echo "Using default PORT=8000"\n\
fi\n\
\n\
# Execute gunicorn with the resolved port\n\
echo "Starting server on port $PORT"\n\
exec gunicorn enhanced:app --log-file=- --log-level=info --bind=0.0.0.0:$PORT --timeout 120 --workers=2 --threads=2\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"] 