from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField, SelectField, RadioField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os
import sys
import cv2
import numpy as np
import time
import platform

# Print debug info at startup
print("=" * 50)
print("STARTUP DEBUG INFO")
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Working directory: {os.getcwd()}")
print("=" * 50)

# Add the current directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Required to run the YOLOv8 model
import cv2

# Import the module but don't run the functions yet - this allows the app to start faster
print("Importing video_detection module...")
from YOLO_Video import video_detection
print("Module imported successfully")

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cyberbot')
app.config['UPLOAD_FOLDER'] = 'static/files'

# Ensure upload folder exists
os.makedirs(os.path.join(current_dir, app.config['UPLOAD_FOLDER']), exist_ok=True)

# Multiple health check routes for Railway deployment
@app.route('/health')
def health_check():
    print("Health check endpoint called")
    return jsonify({"status": "healthy"}), 200

@app.route('/healthz')
def healthz():
    print("Healthz endpoint called")
    return jsonify({"status": "healthy"}), 200

@app.route('/ping')
def ping():
    print("Ping endpoint called")
    return "pong", 200

@app.route('/debug')
def debug():
    """Endpoint for debugging deployment issues"""
    debug_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "environment_variables": {k: v for k, v in os.environ.items() if not k.startswith("AWS")},
        "app_config": {
            "SECRET_KEY_SET": bool(app.config.get('SECRET_KEY')),
            "UPLOAD_FOLDER": app.config.get('UPLOAD_FOLDER'),
        }
    }
    return jsonify(debug_info), 200

#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File", validators=[InputRequired()])
    model_type = RadioField("Detection Model", 
                        choices=[('ppe', 'PPE Detection (Protective Equipment)'), 
                                ('general', 'General Object Detection (People, Cars, etc.)')],
                        default='ppe')
    submit = SubmitField("Run Detection")

class WebcamSelectForm(FlaskForm):
    # Get available camera options
    def get_camera_options():
        options = [(0, "Default Camera")]
        # Try to detect additional cameras
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                options.append((i, f"Camera {i}"))
                cap.release()
        return options
    
    camera = SelectField("Select Camera", choices=get_camera_options(), coerce=int)
    model_type = RadioField("Detection Model", 
                          choices=[('ppe', 'PPE Detection (Protective Equipment)'), 
                                  ('general', 'General Object Detection (People, Cars, etc.)')],
                          default='ppe')
    submit = SubmitField("Use This Camera")

def generate_frames(path_x = '', model_type='ppe'):
    try:
        # Don't access session here
        yolo_output = video_detection(path_x, model_type)
        for detection_ in yolo_output:
            try:
                # Check if detection is None or empty
                if detection_ is None:
                    # Create a fallback frame
                    error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "No valid detection data", (50, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    ref, buffer = cv2.imencode('.jpg', error_frame)
                else:
                    # Make sure the detection is a valid image/frame
                    if isinstance(detection_, np.ndarray) and detection_.size > 0 and len(detection_.shape) == 3:
                        ref, buffer = cv2.imencode('.jpg', detection_)
                    else:
                        # Create a fallback frame for invalid type
                        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, f"Invalid detection format", (50, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        ref, buffer = cv2.imencode('.jpg', error_frame)

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in frame generation: {e}")
                # Create error frame
                error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)[:30]}", (50, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                ref, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)  # Brief pause on error
    except Exception as e:
        print(f"Fatal error in generate_frames: {e}")
        while True:
            # Create error frame for catastrophic failure
            error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Fatal error: {str(e)[:30]}", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(error_frame, "Please try again with different file/settings", (30, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            ref, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0)  # Slow down the loop on fatal error

def generate_frames_web(path_x, model_type='ppe'):
    try:
        # Don't access session here
        yolo_output = video_detection(path_x, model_type)
        for detection_ in yolo_output:
            try:
                # Check if detection is None or empty
                if detection_ is None:
                    # Create a fallback frame
                    error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "No valid detection data", (50, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    ref, buffer = cv2.imencode('.jpg', error_frame)
                else:
                    # Make sure the detection is a valid image/frame
                    if isinstance(detection_, np.ndarray) and detection_.size > 0 and len(detection_.shape) == 3:
                        ref, buffer = cv2.imencode('.jpg', detection_)
                    else:
                        # Create a fallback frame for invalid type
                        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, f"Invalid detection format", (50, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        ref, buffer = cv2.imencode('.jpg', error_frame)

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in webcam frame generation: {e}")
                # Create error frame
                error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)[:30]}", (50, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                ref, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)  # Brief pause on error
    except Exception as e:
        print(f"Fatal error in generate_frames_web: {e}")
        while True:
            # Create error frame for catastrophic failure
            error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Camera error: {str(e)[:30]}", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(error_frame, "Please restart the camera", (50, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            ref, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0)  # Slow down the loop on fatal error

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

# Rendering the Webcam page
@app.route("/webcam", methods=['GET','POST'])
def webcam():
    form = WebcamSelectForm()
    # Default to camera 0 if not specified
    camera_id = session.get('camera_id', 0)
    # Default to PPE model
    model_type = session.get('model_type', 'ppe')
    
    if form.validate_on_submit():
        camera_id = form.camera.data
        model_type = form.model_type.data
        session['camera_id'] = camera_id
        session['model_type'] = model_type
        # Force reload by redirecting to same page
        return render_template('ui.html', form=form, camera_id=camera_id, model_type=model_type, reload=True)
    
    return render_template('ui.html', form=form, camera_id=camera_id, model_type=model_type, reload=False)

@app.route('/FrontPage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    
    # Get the current model type from session (for display purposes)
    current_model = session.get('model_type', 'ppe')
    
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        
        # Use session storage to save video file path and model type
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
        session['model_type'] = form.model_type.data
        
        # Redirect to the video page which will use the session values
        return redirect(url_for('video'))
    
    # Pass the current model type to the template
    return render_template('videoprojectnew.html', form=form, model_type=current_model)

@app.route('/video')
def video():
    # Get model type within request context
    model_type = session.get('model_type', 'ppe')
    video_path = session.get('video_path', None)
    
    # Get model type from query param if provided
    model_override = request.args.get('model_type')
    if model_override:
        model_type = model_override
        session['model_type'] = model_override  # Update session for consistency
    
    return Response(generate_frames(path_x=video_path, model_type=model_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    # Get parameters within request context
    camera_id = session.get('camera_id', 0)
    model_type = session.get('model_type', 'ppe')
    
    # Get model type from query param if provided (helps with refresh)
    model_override = request.args.get('model_type')
    if model_override:
        model_type = model_override
    
    # Pass the model_type to generate_frames_web
    return Response(generate_frames_web(path_x=camera_id, model_type=model_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Add a route to test camera
@app.route('/test_cameras')
def test_cameras():
    results = []
    for i in range(5):  # Test cameras 0-4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            results.append({
                "camera_id": i,
                "status": "Available" if ret else "Error reading frame",
                "resolution": f"{int(cap.get(3))}x{int(cap.get(4))}" if ret else "Unknown"
            })
        else:
            results.append({
                "camera_id": i,
                "status": "Not available",
                "resolution": "N/A"
            })
    return jsonify(results)

if __name__ == "__main__":
    # Development mode
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True)
    # Production mode
    else:
        # Get port from environment variable or use 5000 as default
        port = int(os.environ.get('PORT', 5000))
        # Host 0.0.0.0 makes the server externally visible
        app.run(host='0.0.0.0', port=port, debug=False)