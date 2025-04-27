from ultralytics import YOLO
import cv2
import math
import time
import os
import gc
import numpy as np

# Function to check all available camera backends and find working one
def get_working_camera(camera_id=0, max_attempts=3):
    # List of OpenCV backends to try
    backends = [
        cv2.CAP_ANY,          # Auto-detect
        cv2.CAP_DSHOW,        # DirectShow (Windows)
        cv2.CAP_MSMF,         # Media Foundation (Windows)
        cv2.CAP_V4L2,         # Video4Linux2 (Linux)
        cv2.CAP_AVFOUNDATION, # AVFoundation (macOS)
    ]
    
    # Try each backend
    for backend in backends:
        for attempt in range(max_attempts):
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    # Set properties for better performance
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                    
                    # Read a test frame to make sure it's working
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Successfully opened camera with backend {backend}")
                        return cap
                    else:
                        cap.release()
                        print(f"Failed to read frame with backend {backend}, attempt {attempt+1}")
            except Exception as e:
                print(f"Error with backend {backend}, attempt {attempt+1}: {e}")
                
            # Add delay between attempts
            time.sleep(0.5)
    
    # If all backends failed, try one more time with auto-detect
    print("All backends failed, trying generic approach")
    try:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        return cap
    except Exception as e:
        print(f"Final attempt failed: {e}")
        # Return a dummy camera that will be detected as closed
        return cv2.VideoCapture()

def video_detection(path):
    video_capture = path
    # Load Model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create empty black frames to indicate error
        while True:
            error_frame = create_error_frame(f"Error loading model: {str(e)[:30]}")
            yield error_frame
            time.sleep(1)
    
    # Check if the file is an image
    if isinstance(video_capture, str) and os.path.exists(video_capture):
        file_ext = os.path.splitext(video_capture)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Handle image file
            try:
                # Process the image
                img = cv2.imread(video_capture)
                if img is None:
                    # Image loading error
                    error_frame = create_error_frame(f"Cannot read image file: {video_capture}")
                    yield error_frame
                    return
                
                # Run YOLOv8 inference
                try:
                    results = model(img, conf=0.6, iou=0.5, verbose=False)
                except Exception as e:
                    error_frame = create_error_frame(f"Inference error: {str(e)[:30]}")
                    yield error_frame
                    return
                
                # Process detection results
                detections = []
                annotated_frame = img.copy()
                
                # Class color map
                class_colors = {}
                
                # Visualize the results on the frame
                if hasattr(results, 'boxes') and len(results) > 0:
                    for result in results:
                        boxes = result.boxes  # Boxes object for bbox outputs
                        
                        for box in boxes:
                            # Extract class and confidence
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            # Get the class name
                            cls_name = result.names[cls_id]
                            
                            # Create color for this class if it doesn't exist
                            if cls_id not in class_colors:
                                # Generate a unique color for this class
                                hue = (cls_id * 30) % 180
                                class_colors[cls_id] = tuple(int(c) for c in cv2.cvtColor(
                                    np.array([[[hue, 255, 255]]], dtype=np.uint8), 
                                    cv2.COLOR_HSV2BGR)[0, 0])
                            
                            # Get bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw box with class-specific color
                            color = class_colors[cls_id]
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Create label
                            label = f"{cls_name}: {conf:.2f}"
                            
                            # Draw label background
                            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, y1-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Add to detections list
                            detections.append({
                                'class': cls_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2)
                            })
                
                # Add object count to the frame
                cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Keep yielding the same annotated image (to maintain the generator pattern)
                while True:
                    yield annotated_frame
                    time.sleep(1)  # To prevent high CPU usage
            
            except Exception as e:
                print(f"Error processing image: {e}")
                error_frame = create_error_frame(f"Image processing error: {str(e)[:30]}")
                yield error_frame
                return
            
    # Init variables
    frame_count = 0
    start_time = time.time()
    frames_processed = 0
    current_fps = 0
    skip_frames = 4  # Process every 4th frame
    max_failures = 10  # Maximum consecutive read failures
    consecutive_failures = 0
    error_displayed = False
    
    # Create a class color map
    class_colors = {}
    
    # Initialize camera or video file
    if video_capture == 0:
        # For webcam, use our optimized camera opening function
        cap = get_working_camera(0)
        if not cap.isOpened():
            # If webcam can't be opened, yield error frames
            while True:
                error_frame = create_error_frame("Cannot connect to webcam")
                yield error_frame
                time.sleep(1)
    else:
        # For video files
        if not os.path.exists(video_capture):
            while True:
                error_frame = create_error_frame(f"Video file not found: {video_capture}")
                yield error_frame
                time.sleep(1)
                
        cap = cv2.VideoCapture(video_capture)
        if not cap.isOpened():
            while True:
                error_frame = create_error_frame(f"Cannot open video file: {video_capture}")
                yield error_frame
                time.sleep(1)
    
    # Main processing loop
    try:
        # Read until video is completed or closed
        while cap.isOpened():
            try:
                # Read a frame
                ret, frame = cap.read()
                
                # Check if read was successful
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        print(f"Too many consecutive failures ({consecutive_failures}), stopping")
                        break
                    
                    # Create an error message frame
                    if not error_displayed:
                        error_frame = create_error_frame("Camera initializing...")
                        error_displayed = True
                        yield error_frame
                    
                    # Wait a bit before trying again
                    time.sleep(0.1)
                    continue
                
                # Reset failures counter and error flag if we got a good frame
                consecutive_failures = 0
                error_displayed = False
                
                # Process only every nth frame for performance
                frame_count += 1
                if frame_count % skip_frames != 0 and video_capture == 0:  # Skip frames only for webcam
                    # Just yield the basic frame with minimal processing
                    # Add FPS info
                    if current_fps > 0:
                        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    yield frame
                    continue
                
                # Resizing for better performance
                if video_capture == 0:  # Only resize webcam feed
                    frame = cv2.resize(frame, (640, 360))
                
                # Calculate FPS
                frames_processed += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # Update FPS every second
                    current_fps = frames_processed / elapsed_time
                    frames_processed = 0
                    start_time = time.time()
                
                # Run YOLOv8 inference with optimized settings
                try:
                    # Use lower confidence threshold for speed
                    results = model(frame, conf=0.6, iou=0.5, verbose=False)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    # Add error message to frame
                    cv2.putText(frame, f"Inference error", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if current_fps > 0:
                        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    yield frame
                    continue
                
                # Process detection results
                detections = []
                annotated_frame = frame.copy()
                
                # Visualize the results on the frame
                if hasattr(results, 'boxes') and len(results) > 0:
                    for result in results:
                        boxes = result.boxes  # Boxes object for bbox outputs
                        
                        for box in boxes:
                            # Extract class and confidence
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            # Get the class name
                            cls_name = result.names[cls_id]
                            
                            # Create color for this class if it doesn't exist
                            if cls_id not in class_colors:
                                # Generate a unique color for this class
                                hue = (cls_id * 30) % 180
                                class_colors[cls_id] = tuple(int(c) for c in cv2.cvtColor(
                                    np.array([[[hue, 255, 255]]], dtype=np.uint8), 
                                    cv2.COLOR_HSV2BGR)[0, 0])
                            
                            # Get bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw box with class-specific color
                            color = class_colors[cls_id]
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Create label
                            label = f"{cls_name}: {conf:.2f}"
                            
                            # Draw label background
                            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, y1-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Add to detections list
                            detections.append({
                                'class': cls_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2)
                            })
                
                # Add FPS and detection count to the frame
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                # Force garbage collection occasionally to prevent memory leaks
                if frame_count % 30 == 0:
                    gc.collect()
                
                # Return the annotated frame
                yield annotated_frame
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Create an error frame but keep going
                error_frame = create_error_frame(f"Processing error: {str(e)[:30]}")
                yield error_frame
                time.sleep(0.1)
    
    # Catch any exceptions in the main loop
    except Exception as e:
        print(f"Fatal error in video processing: {e}")
        # Yield error frames if the main loop breaks
        while True:
            error_frame = create_error_frame(f"Fatal error: {str(e)[:30]}")
            yield error_frame
            time.sleep(1)
    
    finally:
        # Always release the video capture object
        if 'cap' in locals() and cap is not None:
            cap.release()
        
        # Force garbage collection
        gc.collect()

def create_error_frame(message):
    """Create a frame with an error message"""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw error message
    cv2.putText(frame, message, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press Refresh button to try again", (50, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame