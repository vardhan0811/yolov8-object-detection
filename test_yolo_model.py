#!/usr/bin/env python3
"""
Test script to diagnose YOLO model loading issues
with detailed error reporting
"""
import os
import sys
import traceback
from ultralytics import YOLO

# Configure more verbose error reporting
import logging
logging.basicConfig(level=logging.DEBUG)

def test_model(model_path):
    """Test loading a specific YOLO model with detailed error handling"""
    print(f"Testing model: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    print(f"File size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
    
    try:
        # Try loading with weights_only=True
        print("Attempting to load model with weights_only=True...")
        model = YOLO(model_path, weights_only=True)
        print("SUCCESS: Model loaded with weights_only=True")
        return True
    except Exception as e:
        print(f"ERROR with weights_only=True: {str(e)}")
        traceback.print_exc()
    
    try:
        # Try loading with weights_only=False
        print("Attempting to load model with weights_only=False...")
        model = YOLO(model_path, weights_only=False)
        print("SUCCESS: Model loaded with weights_only=False")
        return True
    except Exception as e:
        print(f"ERROR with weights_only=False: {str(e)}")
        traceback.print_exc()
    
    try:
        # Try loading with default parameters
        print("Attempting to load model with default parameters...")
        model = YOLO(model_path)
        print("SUCCESS: Model loaded with default parameters")
        return True
    except Exception as e:
        print(f"ERROR with default parameters: {str(e)}")
        traceback.print_exc()
    
    return False

def main():
    """Test all available model files"""
    # Try models in the current directory
    current_dir_models = [f for f in os.listdir('.') if f.endswith('.pt')]
    for model in current_dir_models:
        print(f"\n=== Testing model in current directory: {model} ===")
        if test_model(model):
            print(f"Model {model} loaded successfully")
            return
    
    # Try models in the YOLO-Weights directory
    weights_dir = os.path.join('.', 'YOLO-Weights')
    if os.path.exists(weights_dir):
        weights_dir_models = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pt')]
        for model in weights_dir_models:
            print(f"\n=== Testing model in YOLO-Weights directory: {model} ===")
            if test_model(model):
                print(f"Model {model} loaded successfully")
                return
    
    print("\nERROR: All model loading attempts failed")

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Current directory:", os.getcwd())
    main() 