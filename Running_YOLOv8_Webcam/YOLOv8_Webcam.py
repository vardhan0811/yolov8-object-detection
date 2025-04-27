from ultralytics import YOLO
import cv2
import os
import math

# Get correct path to model relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, "YOLO-Weights", "yolov8n.pt")

# Define output path directly in the project root
output_path = os.path.join(project_root, "output.avi")

print(f"Loading model from: {model_path}")
# Check if model exists
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Downloading...")
    model = YOLO("yolov8n.pt")  # This will download the model
    # Save it to the specified path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
else:
    model = YOLO(model_path)

# Initialize webcam
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        exit(1)
        
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Set up video writer to save in the main project folder
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
    
    # Check if the video writer was initialized correctly
    if not out.isOpened():
        print(f"Warning: Could not initialize video writer. Output will not be saved to {output_path}.")
    else:
        print(f"Video will be saved to: {output_path}")
    
    # COCO class names
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    
    # Define colors for better visualization
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    print("Webcam detection started. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            print("Failed to get frame from webcam. Exiting...")
            break
        
        # Run YOLOv8 inference on the frame
        results = model(img, stream=True)
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                
                # Skip if confidence is too low (optional)
                if conf < 0.4:  # You can adjust this threshold
                    continue
                
                # Get class name
                class_name = classNames[cls]
                
                # Choose color based on class index (cycling through our colors)
                color = colors[cls % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name and confidence
                label = f"{class_name} {conf:.2f}"
                
                # Add label to image with background
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1 - t_size[1] - 5), c2, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 5), 0, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        
        # Add FPS or timestamp info (optional)
        cv2.putText(img, "Press 'q' to quit", (10, frame_height - 10), 0, 0.6, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # Write the frame to video file if writer is available
        if out.isOpened():
            out.write(img)
        
        # Show the frame
        cv2.imshow("YOLOv8 Webcam Detection", img)
        
        # Check for key press - 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
    
except KeyboardInterrupt:
    print("Process interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up resources
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    if 'out' in locals() and out.isOpened():
        out.release()
    cv2.destroyAllWindows()
    print("Resources released.")
    print(f"Output saved to: {output_path}")