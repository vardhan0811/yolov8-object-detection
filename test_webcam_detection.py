from ultralytics import YOLO
import cv2
import time
import os

def test_webcam_detection():
    # Path to the models
    ppe_model_path = "YOLO-Weights/ppe.pt"
    general_model_path = "yolov8n.pt"
    
    # Choose which model to use
    model_path = general_model_path  # Change to ppe_model_path for PPE detection
    
    # Define class names based on model type
    if model_path == ppe_model_path:
        class_names = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']
    else:  # General COCO model
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                      'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                      'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                      'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                      'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        
        # Check if webcam opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            # Try alternative cameras
            for i in range(1, 5):
                print(f"Trying alternative camera index: {i}")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Successfully opened camera with index {i}")
                    break
            
            if not cap.isOpened():
                print("Could not open any camera.")
                return
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return
    
    print("Webcam initialized. Press 'q' to quit.")
    
    # Create output directory for saving frames
    output_dir = "webcam_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # For FPS calculation
        prev_time = time.time()
        frame_count = 0
        
        while True:
            # Read a frame from webcam
            success, img = cap.read()
            
            if not success:
                print("Failed to read frame from webcam")
                break
            
            # Run detection
            results = model(img, stream=True)
            
            # Process results
            for r in results:
                # Draw boxes directly
                img_with_boxes = r.plot()
                
                # Calculate FPS
                frame_count += 1
                curr_time = time.time()
                elapsed = curr_time - prev_time
                
                if elapsed > 1:  # Update FPS every second
                    fps = frame_count / elapsed
                    prev_time = curr_time
                    frame_count = 0
                    # Add FPS display
                    cv2.putText(img_with_boxes, f"FPS: {fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the processed frame
                cv2.imshow("Webcam Detection", img_with_boxes)
            
            # Save a frame every 5 seconds
            if int(time.time()) % 5 == 0:
                output_path = os.path.join(output_dir, f"frame_{int(time.time())}.jpg")
                cv2.imwrite(output_path, img_with_boxes)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

if __name__ == "__main__":
    test_webcam_detection() 