#!/usr/bin/env python3
"""
Test script to diagnose YOLO model loading issues with PyTorch 2.6 compatibility
"""
import os
import sys
import traceback
import glob

def get_model_files():
    """Find all .pt files in the current directory and subdirectories"""
    model_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".pt"):
                model_files.append(os.path.join(root, file))
    return model_files

def test_model_loading():
    """Test loading models with different options"""
    print("=" * 80)
    print("Testing YOLO model loading with different options")
    print("=" * 80)
    
    print("Python version:", sys.version)
    print("Current directory:", os.getcwd())
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"Ultralytics version: {YOLO.__version__}")
    except Exception as e:
        print("Error importing ultralytics:", str(e))
        return
    
    # Find model files
    model_files = get_model_files()
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"  - {model_file} ({size_mb:.2f} MB)")
    
    if not model_files:
        print("No model files found!")
        print("Looking for model files in parent directory...")
        model_files = glob.glob("../*.pt")
        print(f"Found {len(model_files)} model files in parent directory:")
        for model_file in model_files:
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  - {model_file} ({size_mb:.2f} MB)")
    
    # Try loading each model with different weights_only options
    for model_file in model_files:
        print("\n" + "=" * 40)
        print(f"Testing model: {model_file}")
        print("=" * 40)
        
        # Try with weights_only=False
        try:
            print("Loading with weights_only=False...")
            model = YOLO(model_file, weights_only=False)
            print("✅ Successfully loaded model with weights_only=False")
        except Exception as e:
            print(f"❌ Failed to load model with weights_only=False: {str(e)}")
            traceback.print_exc()
        
        # Try with weights_only=True
        try:
            print("\nLoading with weights_only=True...")
            model = YOLO(model_file, weights_only=True)
            print("✅ Successfully loaded model with weights_only=True")
        except Exception as e:
            print(f"❌ Failed to load model with weights_only=True: {str(e)}")
            traceback.print_exc()
        
        # Try with defaults
        try:
            print("\nLoading with default parameters...")
            model = YOLO(model_file)
            print("✅ Successfully loaded model with defaults")
        except Exception as e:
            print(f"❌ Failed to load model with defaults: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    test_model_loading() 