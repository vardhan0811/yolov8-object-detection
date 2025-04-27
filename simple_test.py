import os
import sys

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
    
    # List all .pt files in current directory
    pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    print(f"Found {len(pt_files)} .pt files in current directory:", pt_files)
    
    if pt_files:
        model_path = pt_files[0]
        print(f"Trying to load model: {model_path}")
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model}")
    else:
        print("No .pt files found in current directory")
        
        # Check YOLO-Weights directory
        weights_dir = 'YOLO-Weights'
        if os.path.exists(weights_dir):
            pt_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
            if pt_files:
                model_path = os.path.join(weights_dir, pt_files[0])
                print(f"Trying to load model from weights directory: {model_path}")
                model = YOLO(model_path)
                print(f"Model loaded successfully: {model}")
            else:
                print(f"No .pt files found in {weights_dir} directory")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 