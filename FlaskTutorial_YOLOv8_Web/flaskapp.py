from flask import Flask, render_template, Response, jsonify, request, session
import threading
import time
import queue
import cv2
import os
import numpy as np
import gc  # Garbage collection

# FlaskForm--> it is required to receive input from the user
# Whether uploading a video file to our object detection model
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange

# YOLO_Video is the python file which contains the code for our object detection model
# Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'cyberbot'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Global variables for webcam frame caching
last_frame = None
last_frame_time = 0
frame_cache_age = 0.033  # Maximum age of frame cache in seconds (30 FPS)
is_webcam_active = False
frame_buffer = queue.Queue(maxsize=5)  # Reduced buffer size to save memory
webcam_lock = threading.Lock()
webcam_thread = None
webcam_generator = None
stats = {
    'fps': 0,
    'object_count': 0,
    'confidence': 0,
    'buffer_size': 0,
    'resolution': '640x480'
}

# Use FlaskForm to get input video file from user
class UploadFileForm(FlaskForm):
    # We store the uploaded video file path in the FileField in the variable file
    # We have added validators to make sure the user inputs the video in the valid format and user does upload the
    # video when prompted to do so
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

# Safely stop the webcam thread if it's running
def stop_webcam_thread():
    global is_webcam_active, webcam_thread, webcam_generator, last_frame
    
    # Signal thread to stop
    is_webcam_active = False
    
    # Wait for thread to finish if it exists
    if webcam_thread and webcam_thread.is_alive():
        webcam_thread.join(timeout=1.0)
        webcam_thread = None
    
    # Clear the frame buffer
    while not frame_buffer.empty():
        try:
            frame_buffer.get_nowait()
        except queue.Empty:
            break
    
    # Reset last frame
    with webcam_lock:
        last_frame = None
    
    # Clean up generator
    webcam_generator = None
    
    # Force garbage collection
    gc.collect()

# Background thread for continuous frame processing
def process_webcam_frames():
    global last_frame, last_frame_time, is_webcam_active, webcam_generator, stats
    
    # Set webcam active flag
    is_webcam_active = True
    
    # Create generator
    try:
        webcam_generator = video_detection(0)
        
        # Counter for calculating FPS
        frame_count = 0
        start_time = time.time()
        current_fps = 0
        object_count = 0
        confidence = 0.0
        
        while is_webcam_active:
            try:
                # Get next frame
                frame = next(webcam_generator)
                
                # Calculate FPS every second
                frame_count += 1
                current_time = time.time()
                time_diff = current_time - start_time
                
                if time_diff >= 1.0:  # Update FPS every second
                    current_fps = frame_count / time_diff
                    frame_count = 0
                    start_time = current_time
                
                # Extract object count and confidence from the frame (simple detection)
                # This is just a crude approximation - could be improved with actual data
                # Look for text indicating object count
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for y in range(100, 150):  # Search in the area where we draw stats
                    for x in range(100, 300):
                        if gray[y, x] > 200:  # Bright text
                            # Check if the pixel is part of "Objects: N" text
                            if y == 110 and x > 150 and x < 180:
                                # Extract number from the likely position
                                try:
                                    # Use the fact we know where object count is drawn
                                    object_count_str = ""
                                    for i in range(180, 200):
                                        if gray[110, i] > 200:
                                            object_count_str += "1"
                                        else:
                                            object_count_str += "0"
                                    
                                    # Simple heuristic to estimate object count
                                    ones_count = object_count_str.count("1")
                                    if ones_count > 5:
                                        object_count = 1 + ones_count // 10
                                except:
                                    pass
                                    
                            # Check for confidence value
                            if y == 150 and x > 150 and x < 180:
                                try:
                                    # Use the fact we know confidence is drawn
                                    confidence_str = ""
                                    for i in range(180, 220):
                                        if gray[150, i] > 200:
                                            confidence_str += "1"
                                        else:
                                            confidence_str += "0"
                                    
                                    # Simple heuristic for confidence
                                    ones_count = confidence_str.count("1") 
                                    if ones_count > 0:
                                        confidence = 0.5 + (ones_count / 40)  # Scale to 0.5-1.0
                                except:
                                    pass
                
                # Update global statistics
                with webcam_lock:
                    stats['fps'] = current_fps
                    stats['object_count'] = object_count
                    stats['confidence'] = confidence
                    stats['buffer_size'] = frame_buffer.qsize()
                    h, w = frame.shape[:2]
                    stats['resolution'] = f"{w}x{h}"
                
                # Update the global frame with lock
                with webcam_lock:
                    last_frame = frame
                    last_frame_time = time.time()
                
                # Add to buffer, discard oldest if full
                if frame_buffer.full():
                    try:
                        frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    frame_buffer.put_nowait(frame)
                except queue.Full:
                    pass
                    
            except StopIteration:
                print("Webcam generator stopped")
                break
            except Exception as e:
                print(f"Error in frame processing: {e}")
                time.sleep(0.1)  # Avoid CPU spinning on errors
                
    except Exception as e:
        print(f"Error starting webcam: {e}")
        
    finally:
        # Clean up
        is_webcam_active = False
        print("Webcam thread stopped")

def generate_frames(path_x=''):
    try:
        yolo_output = video_detection(path_x)
        for detection_ in yolo_output:
            try:
                # Check if detection is None or empty
                if detection_ is None:
                    # Create an error frame instead of trying to encode None
                    error_frame = create_error_frame("No valid detection data")
                    ref, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    continue
                
                # Handle various object types from video_detection
                if isinstance(detection_, np.ndarray):  # Frames/images are numpy arrays
                    # Check if the array is empty or invalid
                    if detection_.size == 0 or detection_.shape[0] == 0 or detection_.shape[1] == 0:
                        error_frame = create_error_frame("Empty detection frame")
                        ref, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    else:
                        # Use lower JPEG quality for better performance
                        ref, buffer = cv2.imencode('.jpg', detection_, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                elif isinstance(detection_, tuple) and len(detection_) == 3:
                    # If result is a tuple with (frame, detections, processing_time)
                    frame_data, _, _ = detection_
                    
                    # Check if the frame is empty or invalid
                    if frame_data is None or frame_data.size == 0 or frame_data.shape[0] == 0 or frame_data.shape[1] == 0:
                        error_frame = create_error_frame("Empty detection frame from tuple")
                        ref, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    else:
                        ref, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    # For any unexpected format, create an error message
                    error_frame = create_error_frame(f"Unknown detection format: {type(detection_)}")
                    ref, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                # Create error frame
                error_frame = create_error_frame(f"Error: {str(e)[:30]}")
                ref, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                # Add a small delay to prevent CPU spinning on errors
                time.sleep(0.1)
    except Exception as e:
        # Handle errors in the video_detection generator itself
        print(f"Error in video detection: {e}")
        while True:
            error_frame = create_error_frame(f"Detection error: {str(e)[:30]}")
            ref, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0)  # Slow loop to avoid high CPU usage

def create_error_frame(message):
    """Create a frame with an error message"""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw error message
    cv2.putText(frame, message, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Please try again with a different file", (50, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def generate_frames_web(path_x):
    global last_frame, last_frame_time, is_webcam_active, frame_buffer, webcam_thread
    
    # Start the background frame processing if not already active
    if not is_webcam_active:
        # Stop any existing thread
        stop_webcam_thread()
        
        # Start new thread
        webcam_thread = threading.Thread(target=process_webcam_frames)
        webcam_thread.daemon = True
        webcam_thread.start()
    
    while True:
        try:
            # Try to get a frame from the buffer first
            try:
                frame = frame_buffer.get(timeout=0.5)  # Wait up to 0.5 seconds
            except queue.Empty:
                # If buffer is empty, use the last frame if it's fresh enough
                with webcam_lock:
                    current_time = time.time()
                    if last_frame is not None and (current_time - last_frame_time) < frame_cache_age:
                        frame = last_frame
                    else:
                        # If no frame available or too old, use placeholder
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Loading...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Convert the frame to JPEG with lower quality for better performance
            ref, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Send the frame
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Add a small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS max
            
        except Exception as e:
            print(f"Error in generate_frames_web: {e}")
            # Return an error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Error: {str(e)[:30]}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ref, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Add a longer delay on error
            time.sleep(0.5)

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

# Rendering the Webcam Page
@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    # Stop any existing webcam thread when loading the page
    stop_webcam_thread()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to restart the webcam
@app.route('/api/restart-webcam', methods=['POST'])
def restart_webcam():
    stop_webcam_thread()
    
    # Add a small delay before responding
    time.sleep(0.2)
    
    return jsonify({'status': 'success', 'message': 'Webcam restarted'})

# API endpoint to get detection stats
@app.route('/api/stats')
def get_stats():
    global last_frame, last_frame_time, stats
    
    # Get frame metrics and return stats
    if is_webcam_active:
        return jsonify({
            'active': is_webcam_active,
            'fps': round(stats['fps'], 1),
            'object_count': stats['object_count'],
            'confidence': round(stats['confidence'] * 100),
            'buffer_size': stats['buffer_size'],
            'resolution': stats['resolution'],
            'last_update': last_frame_time
        })
    else:
        return jsonify({'active': is_webcam_active, 'status': 'Webcam not active'})

# Clean up resources when the app is shutting down
@app.teardown_appcontext
def shutdown_session(exception=None):
    stop_webcam_thread()

if __name__ == "__main__":
    # For better performance in development
    app.run(debug=True, threaded=True, host='0.0.0.0')