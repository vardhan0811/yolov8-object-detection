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

# Make sure start.sh is executable
RUN chmod +x start.sh

# Set environment variables
ENV RAILWAY_ENVIRONMENT=true \
    YOLO_DEVICE=cpu \
    YOLO_VERBOSE=False \
    FLASK_ENV=production \
    MODEL_CACHE_ENABLED=true \
    DISABLE_WEBCAM=true

# Run the start script directly
CMD ["bash", "./start.sh"] 