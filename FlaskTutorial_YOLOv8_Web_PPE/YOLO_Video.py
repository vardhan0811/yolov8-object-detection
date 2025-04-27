from ultralytics import YOLO
import cv2
import math
import os
import time
import numpy as np
import threading

# Cache for loaded models to prevent reloading
model_cache = {}
model_cache_lock = threading.Lock()

def get_model(weights_path, model_type):
    """Get model from cache or load it if not present"""
    cache_key = f"{weights_path}_{model_type}"
    
    with model_cache_lock:
        if cache_key in model_cache:
            print(f"Using cached model: {cache_key}")
            return model_cache[cache_key]
        
        print(f"Loading model from: {weights_path}, Model type: {model_type}")
        try:
            # First try loading with weights_only=False (default pre PyTorch 2.6)
            try:
                print("Attempting to load model with weights_only=False...")
                model = YOLO(weights_path, weights_only=False)
                model_cache[cache_key] = model
                print("Model loaded successfully with weights_only=False")
                return model
            except Exception as e1:
                # If that fails, try with weights_only=True (new default in PyTorch 2.6)
                print(f"Loading with weights_only=False failed: {str(e1)}")
                print("Attempting to load model with weights_only=True...")
                model = YOLO(weights_path, weights_only=True)
                model_cache[cache_key] = model
                print("Model loaded successfully with weights_only=True")
                return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If local model fails, try to use the default model from Ultralytics hub
            try:
                if model_type == 'ppe':
                    # Fall back to yolov8n if PPE model fails
                    print("Falling back to general YOLOv8n model")
                    model = YOLO("yolov8n")
                else:
                    # Already trying to use general model, so use yolov8n specifically
                    model = YOLO("yolov8n")
                model_cache[cache_key] = model
                return model
            except Exception as e2:
                print(f"Could not load fallback model either: {str(e2)}")
                raise

def video_detection(path_x, model_type='ppe'):
    video_capture = path_x
    
    # Check if the input is an image file
    if isinstance(video_capture, str) and os.path.exists(video_capture):
        file_ext = os.path.splitext(video_capture)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Process as image file
            try:
                print(f"Processing image file: {video_capture}")
                
                # Read the image
                img = cv2.imread(video_capture)
                
                if img is None:
                    print(f"Error: Could not read image file {video_capture}")
                    error_img = create_error_frame(f"Could not read image file. Check format and permissions.")
                    yield error_img
                    return
                
                # Determine which model to use based on model_type
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                
                if model_type == 'ppe':
                    # Try multiple potential paths for PPE model
                    potential_paths = [
                        os.path.join(project_root, "YOLO-Weights", "ppe.pt"),
                        os.path.join(current_dir, "YOLO-Weights", "ppe.pt"),
                        os.path.join(current_dir, "ppe.pt")
                    ]
                    weights_path = find_existing_file(potential_paths, "PPE model")
                    classNames = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']
                else:
                    # Try multiple potential paths for general model
                    potential_paths = [
                        os.path.join(project_root, "yolov8n.pt"),
                        os.path.join(current_dir, "yolov8n.pt"),
                        "yolov8n"  # This will use the model from Ultralytics hub if local file not found
                    ]
                    weights_path = find_existing_file(potential_paths, "general model")
                    classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                                  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                                  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                                  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                                  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                                  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                
                # Load model and process the image
                try:
                    model = get_model(weights_path, model_type)
                    results = model(img, stream=True)
                    
                    # Process detection results
                    processed_img = img.copy()
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = math.ceil((box.conf[0]*100))/100
                            cls = int(box.cls[0])
                            class_name = classNames[cls]
                            label = f'{class_name} {conf}'
                            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                            c2 = x1 + t_size[0], y1 - t_size[1] - 3
                            
                            # Use different colors for different object types
                            if model_type == 'ppe':
                                if class_name == 'Dust Mask':
                                    color = (0, 204, 255)
                                elif class_name == "Glove":
                                    color = (222, 82, 175)
                                elif class_name == "Protective Helmet":
                                    color = (0, 149, 255)
                                else:
                                    color = (85, 45, 255)
                            else:
                                # For general objects, use a color based on class index
                                color_idx = cls % 10
                                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), 
                                          (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 128), 
                                          (0, 128, 128), (128, 128, 0)]
                                color = colors[color_idx]
                            
                            if conf > 0.5:
                                cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 3)
                                cv2.rectangle(processed_img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                                cv2.putText(processed_img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    
                    # Add a watermark with detection info
                    cv2.putText(processed_img, f"Model: {model_type.upper()}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Keep yielding the same processed image
                    while True:
                        yield processed_img
                        time.sleep(0.1)  # Prevent high CPU usage
                        
                except Exception as e:
                    print(f"Error processing image with model: {str(e)}")
                    error_img = create_error_frame(f"Error processing image: {str(e)[:30]}")
                    yield error_img
                
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
                error_img = create_error_frame(f"Error processing image: {str(e)[:30]}")
                yield error_img
            
            return
    
    # Check if we're in a cloud environment (Railway) - disable webcam in cloud
    is_cloud = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    # Create a Webcam Object only if not in cloud environment
    if is_cloud and (isinstance(video_capture, int) or video_capture == '0'):
        print("Webcam access disabled in cloud environment")
        error_img = create_error_frame("Webcam access is not available in cloud deployment. Please use the video upload feature instead.")
        while True:
            yield error_img
            time.sleep(0.5)
        return
    
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_capture}")
        # For webcam, try alternative camera indices
        if video_capture == 0:
            for i in range(1, 5):
                print(f"Trying alternative camera index: {i}")
                alternative_cap = cv2.VideoCapture(i)
                if alternative_cap.isOpened():
                    print(f"Successfully opened camera with index {i}")
                    cap = alternative_cap
                    break
                alternative_cap.release()
        
        # If still not open, yield error frame
        if not cap.isOpened():
            # Create an error image to display
            error_img = create_error_frame("Camera not accessible. Please check permissions and availability.")
            yield error_img
            return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Determine which model to use based on model_type
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if model_type == 'ppe':
        # PPE detection model
        weights_path = os.path.join(project_root, "YOLO-Weights", "ppe.pt")
        classNames = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']
    else:
        # General object detection model (COCO dataset - 80 classes)
        weights_path = os.path.join(project_root, "yolov8n.pt")
        classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                      'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                      'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                      'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                      'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    # Print model path to confirm it's correct
    print(f"Loading model from: {weights_path}, Model type: {model_type}")
    
    try:
        model = get_model(weights_path, model_type)
        
        # Initialize frame counter for error reporting
        frame_count = 0
        start_time = time.time()
        
        while True:
            success, img = cap.read()
            
            # Check if frame is read correctly
            if not success:
                frame_count += 1
                # Only break after multiple consecutive failures
                if frame_count > 10:
                    print("Failed to read frame from the video source after multiple attempts")
                    # Create an error image to yield
                    error_img = create_error_frame("Video feed unavailable. Please check your camera.")
                    yield error_img
                    break
                
                # If it's been less than 3 seconds, try again
                if time.time() - start_time < 3:
                    # Small delay before retrying
                    time.sleep(0.1)
                    continue
                else:
                    # Reset counter and timer
                    frame_count = 0
                    start_time = time.time()
            else:
                # Reset counter on successful frame
                frame_count = 0
                
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(x1, y1, x2, y2)
                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    label = f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    
                    # Use different colors for different object types
                    if model_type == 'ppe':
                        if class_name == 'Dust Mask':
                            color = (0, 204, 255)
                        elif class_name == "Glove":
                            color = (222, 82, 175)
                        elif class_name == "Protective Helmet":
                            color = (0, 149, 255)
                        else:
                            color = (85, 45, 255)
                    else:
                        # For general objects, use a color based on class index
                        color_idx = cls % 10
                        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), 
                                  (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 128), 
                                  (0, 128, 128), (128, 128, 0)]
                        color = colors[color_idx]
                    
                    if conf > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            yield img
    except Exception as e:
        print(f"Error in video detection: {str(e)}")
        error_img = create_error_frame(f"Error: {str(e)}")
        yield error_img
    finally:
        # Make sure to release the camera
        cap.release()

def find_existing_file(paths, description):
    """Try multiple paths and return the first one that exists"""
    for path in paths:
        if os.path.exists(path):
            print(f"Found {description} at: {path}")
            return path
    
    # If no file exists, use the first path - the model will be downloaded from Ultralytics hub if needed
    print(f"No local {description} found, will try to use model from Ultralytics hub")
    return paths[0]

def create_error_frame(error_message):
    """Create an error frame with text to display when camera fails"""
    # Create a black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add error message
    cv2.putText(img, "Error:", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Break error message into multiple lines if needed
    y_pos = 120
    words = error_message.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if len(test_line) > 35:  # Limit characters per line
            cv2.putText(img, line, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
            line = word + " "
        else:
            line = test_line
    
    # Add the last line
    if line:
        cv2.putText(img, line, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

cv2.destroyAllWindows()