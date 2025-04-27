#!/usr/bin/env python3
"""
Real-time Object Detection Web Application

This application uses Flask to create a web interface for object detection
using YOLOv8 models. It supports uploading images and videos for detection,
as well as real-time detection through webcam streaming.
"""

import os
import sys
import logging
import uuid
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename

# Import our custom modules
try:
    from download_models import ensure_yolo_weights
    from YOLO_Video import YOLO_Video
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all required modules are in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max file size

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables
model = None
camera = None
output_frame = None
frame_lock = None
detection_running = False

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_path=None, confidence=0.25, device='cpu'):
    """Load YOLOv8 model"""
    global model
    
    try:
        # Try to import Ultralytics YOLO
        from ultralytics import YOLO
        
        # If model_path is None, use default YOLOv8n model
        if model_path is None or not os.path.exists(model_path):
            # Get available models
            models = ensure_yolo_weights(['yolov8n.pt'])
            if not models:
                logger.error("Failed to load YOLOv8 model: No models available")
                return None
            model_path = models.get('yolov8n.pt')
        
        # Load model
        logger.info(f"Loading YOLOv8 model from {model_path}")
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully: {model}")
        return model
        
    except ImportError:
        logger.warning("Ultralytics package not found. Trying to use PyTorch directly.")
        try:
            import torch
            
            if model_path is None or not os.path.exists(model_path):
                # Get available models
                models = ensure_yolo_weights(['yolov8n.pt'])
                if not models:
                    logger.error("Failed to load YOLOv8 model: No models available")
                    return None
                model_path = models.get('yolov8n.pt')
            
            # Load model using PyTorch
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'model'):
                model = model.model  # Unwrap the model if it's in a wrapper class
            
            # Set model to evaluation mode
            model.eval()
            logger.info(f"Model loaded with PyTorch: {model}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

def process_image(file_path, confidence=0.25, device='cpu'):
    """Process an image file for object detection"""
    global model
    
    try:
        # Ensure model is loaded
        if model is None:
            model = load_model(confidence=confidence, device=device)
            if model is None:
                return None, "Failed to load model"

        # Create a unique output filename
        filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}_result.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Perform detection with the model
        logger.info(f"Processing image: {file_path}")
        try:
            # Try with Ultralytics API
            results = model(file_path, conf=confidence)
            # Save the results
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                cv2.imwrite(output_path, im_array)
        except AttributeError:
            # If using PyTorch directly
            image = cv2.imread(file_path)
            # Preprocess for PyTorch model
            # This is a simplified version; actual preprocessing depends on model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (640, 640))
            image_tensor = np.transpose(image_resized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, 0).astype(np.float32) / 255.0
            
            import torch
            with torch.no_grad():
                # Convert to tensor and move to device
                input_tensor = torch.from_numpy(image_tensor).to(device)
                # Forward pass
                output = model(input_tensor)
                
                # Process results (simplified)
                # This needs to be adapted to match the model's output format
                # This is a placeholder for demonstration
                boxes = output[0]
                
                # Draw boxes on image
                result_image = image.copy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if conf > confidence:
                        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(result_image, f"Class {int(cls)}: {conf:.2f}", 
                                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imwrite(output_path, result_image)
        
        logger.info(f"Image processed successfully: {output_path}")
        return output_filename, None
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, str(e)

def process_video(file_path, confidence=0.25, device='cpu'):
    """Process a video file for object detection"""
    global model
    
    try:
        # Ensure model is loaded
        if model is None:
            model = load_model(confidence=confidence, device=device)
            if model is None:
                return None, "Failed to load model"

        # Create a unique output filename
        filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}_result.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Process video
        logger.info(f"Processing video: {file_path}")
        
        # Get model path
        model_path = None
        if hasattr(model, 'pt_path'):
            model_path = model.pt_path
        elif hasattr(model, 'model_path'):
            model_path = model.model_path
        
        # Use our YOLO_Video function
        success = YOLO_Video(
            video_path=file_path,
            model_path=model_path,
            output_path=output_path,
            conf_threshold=confidence,
            device=device,
            weights_only=(model_path is None)  # If model_path is None, we're using the loaded model
        )
        
        if success:
            logger.info(f"Video processed successfully: {output_path}")
            return output_filename, None
        else:
            return None, "Failed to process video"
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None, str(e)

@app.route('/')
def index():
    """Render the home page"""
    # Get the list of available models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YOLO-Weights')
    available_models = []
    
    # Check for models in YOLO-Weights directory
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pt'):
                available_models.append(os.path.join(models_dir, file))
    
    # Check for models in current directory
    for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if file.endswith('.pt'):
            available_models.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), file))
    
    return render_template('index.html', models=available_models)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
    
    # Get parameters
    confidence = float(request.form.get('confidence', 0.25))
    device = request.form.get('device', 'cpu')
    model_path = request.form.get('model', None)
    
    if model_path == 'default':
        model_path = None
    
    # Create a unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save the file
    file.save(file_path)
    logger.info(f"File saved: {file_path}")
    
    # Process the file based on its type
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    if file_extension in {'png', 'jpg', 'jpeg'}:
        # Process image
        output_filename, error = process_image(file_path, confidence, device)
        if error:
            return jsonify({'error': error}), 500
        
        # Return the result
        result_url = url_for('static', filename=f'results/{output_filename}')
        return jsonify({
            'success': True,
            'result_type': 'image',
            'result_url': result_url
        })
    
    elif file_extension in {'mp4', 'avi', 'mov', 'mkv'}:
        # Process video
        output_filename, error = process_video(file_path, confidence, device)
        if error:
            return jsonify({'error': error}), 500
        
        # Return the result
        result_url = url_for('static', filename=f'results/{output_filename}')
        return jsonify({
            'success': True,
            'result_type': 'video',
            'result_url': result_url
        })
    
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/webcam')
def webcam():
    """Render the webcam page"""
    return render_template('webcam.html')

def generate_frames():
    """Generator function for webcam streaming"""
    global camera, output_frame, detection_running
    
    # Initialize the camera if not already initialized
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Failed to open webcam")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   b'Failed to open webcam' + b'\r\n')
            return
    
    # Ensure model is loaded
    if model is None:
        load_model()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # If detection is running, use the processed frame
        if detection_running and output_frame is not None:
            encoded_frame = cv2.imencode('.jpg', output_frame)[1].tobytes()
        else:
            # Otherwise, use the original frame
            encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start object detection on webcam feed"""
    global detection_running, model, camera, output_frame, frame_lock
    
    # Get parameters
    confidence = float(request.form.get('confidence', 0.25))
    
    # Ensure model is loaded
    if model is None:
        model = load_model(confidence=confidence)
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
    
    # Start detection
    detection_running = True
    
    # Return success
    return jsonify({'success': True})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop object detection on webcam feed"""
    global detection_running
    
    # Stop detection
    detection_running = False
    
    # Return success
    return jsonify({'success': True})

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/download_model', methods=['POST'])
def download_model():
    """Download a YOLOv8 model"""
    model_name = request.form.get('model_name', 'yolov8n.pt')
    
    # Ensure model is downloaded
    models = ensure_yolo_weights([model_name])
    
    if not models:
        return jsonify({'error': f'Failed to download model: {model_name}'}), 500
    
    return jsonify({
        'success': True,
        'model_path': models[model_name]
    })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure YOLO weights are available
    ensure_yolo_weights()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug) 