import os
import cv2
from flask import Flask, render_template, Response, jsonify
import time
import json
import numpy as np
from YOLO_Video import video_detection, get_working_camera

app = Flask(__name__)

# Global variables to track webcam state and stats
webcam = None
webcam_id = 0
detection_stats = {
    "fps": 0,
    "object_count": 0,
    "confidence": 0,
    "width": 0,
    "height": 0,
    "last_update": time.time()
}

# Frame processing metrics
frame_count = 0
start_time = time.time()
processing_times = []
detected_objects = []

def gen_frames():  
    global webcam, frame_count, start_time, processing_times, detected_objects, detection_stats
    
    if webcam is None:
        webcam = get_working_camera(camera_id=webcam_id)
    
    if webcam is None or not webcam.isOpened():
        # Return error frame
        error_frame = create_error_frame("Camera not available. Please restart.")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    for result_frame, detections, processing_time in video_detection(webcam):
        # Update stats
        frame_count += 1
        processing_times.append(processing_time)
        
        # Calculate FPS (limit history to last 30 frames)
        if len(processing_times) > 30:
            processing_times.pop(0)
        
        if processing_times:
            current_fps = 1.0 / (sum(processing_times) / len(processing_times))
        else:
            current_fps = 0
            
        # Update detection objects
        detected_objects = detections
        
        # Update global stats
        current_time = time.time()
        if current_time - detection_stats["last_update"] >= 0.5:  # Update stats every 0.5 seconds
            detection_stats["fps"] = current_fps
            detection_stats["object_count"] = len(detections)
            detection_stats["last_update"] = current_time
            
            # Calculate average confidence
            if detections:
                avg_conf = sum(det['confidence'] for det in detections) / len(detections)
                detection_stats["confidence"] = avg_conf
            else:
                detection_stats["confidence"] = 0
                
            # Get frame dimensions
            if result_frame is not None:
                height, width = result_frame.shape[:2]
                detection_stats["width"] = width
                detection_stats["height"] = height
        
        # Encode the frame
        if result_frame is not None:
            ret, buffer = cv2.imencode('.jpg', result_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Return error frame if result_frame is None
            error_frame = create_error_frame("Processing error. Please restart.")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def create_error_frame(message):
    # Create a black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add error message
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/api/restart-webcam')
def restart_webcam():
    global webcam, webcam_id
    
    # Release current webcam if it exists
    if webcam is not None:
        webcam.release()
        webcam = None
    
    try:
        # Try to open the webcam with a new instance
        webcam = get_working_camera(camera_id=webcam_id)
        
        if webcam is not None and webcam.isOpened():
            return jsonify({'status': 'success', 'message': 'Camera restarted successfully'})
        else:
            # Try different camera ID
            webcam_id = 0 if webcam_id > 0 else 1
            webcam = get_working_camera(camera_id=webcam_id)
            
            if webcam is not None and webcam.isOpened():
                return jsonify({'status': 'success', 'message': 'Camera restarted with different ID'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to restart camera'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'status': 'success',
        'fps': detection_stats["fps"],
        'object_count': detection_stats["object_count"],
        'confidence': detection_stats["confidence"],
        'width': detection_stats["width"],
        'height': detection_stats["height"]
    })

if __name__ == "__main__":
    app.run(debug=True) 