from ultralytics import YOLO
import cv2
import os
import math
import time

# Get the script and project directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set up model path
model_path = os.path.join(project_root, "YOLO-Weights", "yolov8n.pt")

# Set up input video path
video_path = os.path.join(project_root, "Videos", "bikes.mp4")

# Define output path directly in the project root
output_path = os.path.join(project_root, "output.avi")

# Check if input video exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    default_samples = os.path.join(os.path.dirname(os.path.dirname(cv2.__file__)), 'samples', 'data')
    if os.path.exists(os.path.join(default_samples, 'vtest.avi')):
        print(f"Using default video: vtest.avi")
        video_path = os.path.join(default_samples, 'vtest.avi')
    else:
        print("No default video found. Please provide a valid video file.")
        exit(1)

# Load the YOLOv8 model
try:
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
        
    # Open the video file
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit(1)
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer to save in the main project folder
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
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
    
    # Process the video
    print("Starting video processing...")
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read a frame from the video
        success, img = cap.read()
        if not success:
            print("End of video reached.")
            break
        
        frame_count += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames}), Processing rate: {fps_processing:.1f} FPS")
        
        # Perform object detection
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
                
                # Skip if confidence is too low
                if conf < 0.4:  # Confidence threshold, adjust as needed
                    continue
                
                # Get class name and assign color
                class_name = classNames[cls]
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
        
        # Add progress information to the frame
        progress_text = f"Frame: {frame_count}/{total_frames} ({(frame_count/total_frames*100):.1f}%)"
        cv2.putText(img, progress_text, (10, 30), 0, 0.7, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # Write the processed frame to the output video
        if out.isOpened():
            out.write(img)
        
        # Display the processed frame
        cv2.imshow("YOLOv8 Video Processing", img)
        
        # Check for key press - 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user.")
            break
    
    # Calculate and display final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"Video processing complete!")
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
    print(f"Output saved to: {output_path}")
    
except KeyboardInterrupt:
    print("Processing interrupted by user.")
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