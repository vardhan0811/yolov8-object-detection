from ultralytics import YOLO
import cv2
import os
import time

def test_video_detection():
    # Path to the test video
    video_path = "Videos/bikes.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Path to the models
    ppe_model_path = "YOLO-Weights/ppe.pt"
    general_model_path = "yolov8n.pt"
    
    # Choose which model to use
    model_path = general_model_path  # Change to ppe_model_path for PPE detection
    
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize video capture
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    except Exception as e:
        print(f"Error initializing video: {e}")
        return
    
    # Create output directory for saving frames and video
    output_dir = "video_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create video writer
    output_video_path = os.path.join(output_dir, "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Read a frame from video
            success, img = cap.read()
            
            if not success:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            
            # Process every 2nd frame to speed up for preview
            if frame_count % 2 != 0:
                continue
                
            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames}), Elapsed: {elapsed:.1f}s")
            
            # Run detection
            results = model(img, stream=True)
            
            # Process results
            for r in results:
                # Draw boxes on the frame
                img_with_boxes = r.plot()
                
                # Write to output video
                out.write(img_with_boxes)
                
                # Save key frames
                if frame_count % 60 == 0:  # Save a frame every ~2 seconds
                    frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, img_with_boxes)
                
                # Display the frame
                cv2.imshow("Video Detection", img_with_boxes)
                
                # Display progress on frame
                progress = (frame_count / total_frames) * 100
                cv2.putText(img_with_boxes, f"Progress: {progress:.1f}%", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detection stopped by user")
                break
                
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"Detection completed in {elapsed:.2f} seconds")
        print(f"Processed {frame_count} frames")
        print(f"Output video saved to: {output_video_path}")
        print(f"Output frames saved to: {output_dir}")

if __name__ == "__main__":
    test_video_detection() 