#!/usr/bin/env python3
"""
YOLO Model Downloader

This module provides functionality to automatically download YOLOv8 model weights
if they are not found in the local directory. It ensures that required model files
are available for the application to use.
"""

import os
import sys
import logging
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define model URLs and checksums
YOLO_MODELS = {
    'yolov8n.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'size': 6400240,
        'md5': 'f9f3824cb2931e4d8568e47ada07f203',
    },
    'yolov8s.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'size': 22997144,
        'md5': '5f8774abdcf7d69e73af24fcf931f713', 
    },
    'yolov8m.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'size': 52927336,
        'md5': '7be518b68829722aa511236571e950ab',
    },
    'yolov8l.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'size': 87006296,
        'md5': '2f86dd4fbc296f7ec200c4e02eddb3c7',
    },
    'yolov8x.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
        'size': 136432536,
        'md5': '486d9dd1cfd9d06436b97c825e1cffc2',
    }
}

def get_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, file_path):
    """
    Download a file from a URL with a progress bar
    
    Args:
        url (str): URL to download from
        file_path (str): Local path to save the file
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Set up the session
        session = requests.Session()
        response = session.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download the file with progress bar
        logger.info(f"Downloading {url} to {file_path}")
        with open(file_path, 'wb') as f, tqdm(
            desc=os.path.basename(file_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def verify_model(model_path, expected_md5=None, expected_size=None):
    """
    Verify a model file by checking its size and MD5 hash
    
    Args:
        model_path (str): Path to the model file
        expected_md5 (str, optional): Expected MD5 hash
        expected_size (int, optional): Expected file size in bytes
    
    Returns:
        bool: True if the model is valid, False otherwise
    """
    # Check if file exists
    if not os.path.isfile(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return False
    
    # Check file size if expected_size is provided
    if expected_size is not None:
        actual_size = os.path.getsize(model_path)
        if actual_size != expected_size:
            logger.warning(f"Model size mismatch for {model_path}: expected {expected_size}, got {actual_size}")
            return False
    
    # Check MD5 hash if expected_md5 is provided
    if expected_md5 is not None:
        actual_md5 = get_md5(model_path)
        if actual_md5 != expected_md5:
            logger.warning(f"Model MD5 mismatch for {model_path}: expected {expected_md5}, got {actual_md5}")
            return False
    
    return True

def ensure_yolo_weights(models_to_check=None, download_dir='YOLO-Weights'):
    """
    Ensure YOLOv8 weights are available, downloading them if necessary
    
    Args:
        models_to_check (list, optional): List of model names to check. If None, checks all models.
        download_dir (str, optional): Directory to save downloaded models. If a relative path, 
                                    it will be relative to the current working directory.
    
    Returns:
        dict: Dictionary with model names as keys and their paths as values for available models
    """
    # Convert download_dir to absolute path if it's a relative path
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    
    # If no models are specified, check all
    if models_to_check is None:
        models_to_check = list(YOLO_MODELS.keys())
    
    # Dictionary to store available model paths
    available_models = {}
    
    for model_name in models_to_check:
        if model_name not in YOLO_MODELS:
            logger.warning(f"Unknown model: {model_name}")
            continue
            
        model_info = YOLO_MODELS[model_name]
        model_path = os.path.join(download_dir, model_name)
        
        # First check if model exists in current directory
        current_dir_model = os.path.join(os.getcwd(), model_name)
        if os.path.isfile(current_dir_model) and verify_model(
            current_dir_model, 
            model_info.get('md5'), 
            model_info.get('size')
        ):
            logger.info(f"Using existing model in current directory: {current_dir_model}")
            available_models[model_name] = current_dir_model
            continue
            
        # Then check if model exists in download directory
        if os.path.isfile(model_path) and verify_model(
            model_path, 
            model_info.get('md5'), 
            model_info.get('size')
        ):
            logger.info(f"Using existing model in download directory: {model_path}")
            available_models[model_name] = model_path
            continue
            
        # If model doesn't exist or is invalid, download it
        logger.info(f"Model {model_name} not found or invalid, downloading...")
        if download_file(model_info['url'], model_path):
            if verify_model(model_path, model_info.get('md5'), model_info.get('size')):
                logger.info(f"Successfully downloaded and verified {model_name}")
                available_models[model_name] = model_path
            else:
                logger.error(f"Downloaded model {model_name} failed verification")
        else:
            logger.error(f"Failed to download {model_name}")
    
    if not available_models:
        logger.warning("No valid YOLOv8 models available")
    else:
        logger.info(f"Available models: {', '.join(available_models.keys())}")
    
    return available_models

if __name__ == "__main__":
    # When run as a script, download all models
    print("YOLO Model Downloader")
    print("====================")
    
    # Allow specifying custom models and download directory via command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download YOLOv8 models')
    parser.add_argument('--models', nargs='*', help='Specific models to download (default: all)')
    parser.add_argument('--dir', default='YOLO-Weights', help='Directory to save models')
    args = parser.parse_args()
    
    models = ensure_yolo_weights(args.models, args.dir)
    
    if models:
        print("\nDownloaded models:")
        for name, path in models.items():
            print(f"- {name}: {path}")
    else:
        print("\nNo models were downloaded or available.") 