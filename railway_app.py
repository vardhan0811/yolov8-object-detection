#!/usr/bin/env python3
"""
Simplified Railway Deployment Version of YOLOv8 Object Detection App
"""

import os
import sys
import logging
import traceback
import time
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Log system information
logger.info("=" * 50)
logger.info("Starting Railway Deployment Application")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Files in current directory: {os.listdir('.')}")

# Try to import required modules with error handling
try:
    import cv2
    logger.info(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    logger.error(f"Failed to import OpenCV: {str(e)}")
    logger.error(traceback.format_exc())

try:
    import numpy as np
    logger.info(f"NumPy imported successfully")
except ImportError as e:
    logger.error(f"Failed to import NumPy: {str(e)}")
    logger.error(traceback.format_exc())

try:
    from ultralytics import YOLO
    logger.info("Ultralytics YOLO imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Ultralytics YOLO: {str(e)}")
    logger.error(traceback.format_exc())

# Find template and static directories
current_dir = os.path.dirname(os.path.abspath(__file__))
template_paths = [
    os.path.join(current_dir, 'templates'),
    os.path.join(current_dir, 'FlaskTutorial_YOLOv8_Web_PPE', 'templates'),
]

# Find the first valid template directory
template_folder = None
for path in template_paths:
    if os.path.exists(path):
        template_folder = path
        logger.info(f"Using template folder: {template_folder}")
        break

if template_folder is None:
    logger.error("No valid template folder found!")
    template_folder = os.path.join(current_dir, 'templates')
    os.makedirs(template_folder, exist_ok=True)
    
# Find static folder
static_paths = [
    os.path.join(current_dir, 'static'),
    os.path.join(current_dir, 'FlaskTutorial_YOLOv8_Web_PPE', 'static'),
]

# Find the first valid static directory
static_folder = None
for path in static_paths:
    if os.path.exists(path):
        static_folder = path
        logger.info(f"Using static folder: {static_folder}")
        break

if static_folder is None:
    logger.error("No valid static folder found!")
    static_folder = os.path.join(current_dir, 'static')
    os.makedirs(static_folder, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=static_folder)

# App configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = os.path.join(static_folder, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(static_folder, 'results')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables
model = None
model_loaded = False
model_loading_error = None

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def find_model_file():
    """Find available YOLO model files"""
    model_paths = []
    
    # Check current directory
    for file in os.listdir(current_dir):
        if file.endswith('.pt'):
            model_paths.append(os.path.join(current_dir, file))
            
    # Check YOLO-Weights directory if it exists
    weights_dir = os.path.join(current_dir, 'YOLO-Weights')
    if os.path.exists(weights_dir):
        for file in os.listdir(weights_dir):
            if file.endswith('.pt'):
                model_paths.append(os.path.join(weights_dir, file))
    
    if model_paths:
        logger.info(f"Found model files: {model_paths}")
        return model_paths[0]  # Return the first model file found
    else:
        logger.error("No model files found!")
        return None

def load_model_safe():
    """Load YOLOv8 model with extensive error handling"""
    global model, model_loaded, model_loading_error
    
    try:
        logger.info("Attempting to load YOLO model...")
        model_path = find_model_file()
        
        if model_path is None:
            model_loading_error = "No model file found"
            logger.error(model_loading_error)
            return None
            
        if not os.path.exists(model_path):
            model_loading_error = f"Model file does not exist: {model_path}"
            logger.error(model_loading_error)
            return None
            
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"File size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
        
        # Fix for PyTorch 2.6 weights_only issue
        try:
            import torch
            logger.info("Setting up safe globals for PyTorch model loading")
            # Import required modules for patching
            try:
                import ultralytics.nn.tasks
                # Add safe globals for ultralytics models
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                logger.info("Added ultralytics.nn.tasks.DetectionModel to safe globals")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not patch ultralytics safe globals: {str(e)}")
            
            try:
                # Set the environment variable to allow unsafe loading
                logger.info("Setting TORCH_ALLOW_WEIGHTS_ONLY_LOAD_DETECTION=1")
                os.environ["TORCH_ALLOW_WEIGHTS_ONLY_LOAD_DETECTION"] = "1"
            except Exception as e:
                logger.warning(f"Could not set environment variable: {str(e)}")
            
        except ImportError as e:
            logger.warning(f"Could not import torch for patching: {str(e)}")
        
        # Load model with different parameters
        try:
            # Try with weights_only=False first (safer approach)
            try:
                import torch
                logger.info("Trying to load with torch.serialization.safe_globals context")
                # Use context manager for safest approach
                with torch.serialization.safe_globals(['ultralytics.nn.tasks.DetectionModel']):
                    model = YOLO(model_path, weights_only=False)
                logger.info("Model loaded successfully with safe_globals context")
                model_loaded = True
                return model
            except (ImportError, AttributeError) as e:
                logger.warning(f"Context manager approach failed: {str(e)}")
                # Try without context if not available
                model = YOLO(model_path, weights_only=False)
                logger.info("Model loaded successfully with weights_only=False")
                model_loaded = True
                return model
        except Exception as e1:
            logger.warning(f"Failed to load with weights_only=False: {str(e1)}")
            
        try:
            # Force task='detect' to improve compatibility
            logger.info("Trying with task='detect' explicitly")
            model = YOLO(model_path, task='detect')
            logger.info("Model loaded successfully with task='detect'")
            model_loaded = True
            return model
        except Exception as e2:
            logger.warning(f"Failed to load with task='detect': {str(e2)}")
            
        try:
            # Try with default parameters
            logger.info("Trying with default parameters")
            model = YOLO(model_path)
            logger.info("Model loaded successfully with default parameters")
            model_loaded = True
            return model
        except Exception as e3:
            model_loading_error = f"All model loading attempts failed: {str(e3)}"
            logger.error(model_loading_error)
            logger.error(traceback.format_exc())
            return None
    
    except Exception as e:
        model_loading_error = f"Error loading model: {str(e)}"
        logger.error(model_loading_error)
        logger.error(traceback.format_exc())
        return None

def process_image(file_path, confidence=0.25):
    """Process an image file for object detection with extensive error handling"""
    global model, model_loaded
    
    try:
        # Ensure model is loaded
        if model is None:
            logger.info("Model not loaded yet, attempting to load...")
            model = load_model_safe()
            
            if model is None:
                # Fallback: process the image without detection (just return the original)
                logger.warning("Model could not be loaded, returning original image as fallback")
                
                # Read the original image
                try:
                    import cv2
                    logger.info(f"Reading original image: {file_path}")
                    image = cv2.imread(file_path)
                    
                    if image is None:
                        return None, "Failed to read image file"
                    
                    # Add a text overlay explaining the situation
                    height, width, _ = image.shape
                    cv2.putText(image, "Model loading failed - original image shown", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(image, "See model error for details", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Save the image with the message
                    filename = os.path.basename(file_path)
                    output_filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}_original.jpg"
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    
                    cv2.imwrite(output_path, image)
                    logger.info(f"Saved original image with message: {output_path}")
                    
                    return output_filename, None
                except Exception as e:
                    logger.error(f"Error in fallback image processing: {str(e)}")
                    return None, "Failed to load model and process image"
        
        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"Image file not found: {file_path}"
            logger.error(error_msg)
            return None, error_msg
            
        # Create output filename
        filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}_result.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Run detection
        logger.info(f"Processing image: {file_path}")
        logger.info(f"Confidence threshold: {confidence}")
        
        # Perform detection
        try:
            results = model(file_path, conf=confidence)
            
            # Check if results are valid
            if results is None or len(results) == 0:
                logger.warning("No detection results returned, using original image with message")
                
                # Read the original image
                image = cv2.imread(file_path)
                # Add a text overlay explaining the situation
                cv2.putText(image, "No objects detected", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Save the image with the message
                cv2.imwrite(output_path, image)
                
                return output_filename, None
                
            # Save result image
            for r in results:
                im_array = r.plot()
                cv2.imwrite(output_path, im_array)
                
            logger.info(f"Image processed successfully: {output_path}")
            return output_filename, None
            
        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Fallback to returning the original image with an error message
            try:
                image = cv2.imread(file_path)
                # Add a text overlay explaining the error
                cv2.putText(image, "Error during detection", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(image, str(e)[:50], 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Save the image with the message
                cv2.imwrite(output_path, image)
                
                return output_filename, None
            except:
                return None, error_msg
    
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

# Health check endpoint (required for Railway)
@app.route('/health')
def health():
    return "healthy", 200

@app.route('/healthz')
def healthz():
    return "healthy", 200

@app.route('/ping')
def ping():
    return "pong", 200

# Basic error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Main routes
@app.route('/')
def index():
    """Render home page"""
    try:
        # Get list of available models
        model_files = []
        
        # Check current directory
        for file in os.listdir(current_dir):
            if file.endswith('.pt'):
                model_files.append(file)
                
        # Check YOLO-Weights directory if it exists
        weights_dir = os.path.join(current_dir, 'YOLO-Weights')
        if os.path.exists(weights_dir):
            for file in os.listdir(weights_dir):
                if file.endswith('.pt'):
                    model_files.append(file)
        
        # Don't try to render a template that might reference webcam, just send directly to upload page
        return redirect(url_for('upload_file'))
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        logger.error(traceback.format_exc())
        # Send directly to upload page as fallback
        return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload for processing"""
    if request.method == 'GET':
        return render_template('upload.html', model_loaded=model_loaded)
        
    try:
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
                
            file = request.files['file']
            
            # If user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
                
            # Create a secure filename and save the file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            # Get parameters
            confidence = float(request.form.get('confidence', 0.25))
            
            # Process the file based on its type
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension in {'png', 'jpg', 'jpeg'}:
                # Process image
                output_filename, error = process_image(file_path, confidence)
                
                if error:
                    return jsonify({'error': error}), 500
                    
                # Return the result URL
                result_url = url_for('static', filename=f'results/{output_filename}')
                return jsonify({
                    'success': True,
                    'result_type': 'image',
                    'result_url': result_url
                })
            else:
                return jsonify({'error': 'Only image processing is supported in this deployment'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/debug')
def debug():
    """Debug information page"""
    debug_info = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'template_folder': template_folder,
        'static_folder': static_folder,
        'model_loaded': model_loaded,
        'model_error': model_loading_error,
        'environment_variables': {k: v for k, v in os.environ.items() if not k.lower() in ['secret', 'key', 'password', 'token']},
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'output_folder': app.config['OUTPUT_FOLDER'],
    }
    
    return jsonify(debug_info)

@app.route('/reload_model')
def reload_model():
    """Manually attempt to reload the model"""
    global model, model_loaded, model_loading_error
    
    # Clear current model
    model = None
    model_loaded = False
    model_loading_error = None
    
    # Set environment variables to help with loading
    os.environ["TORCH_ALLOW_WEIGHTS_ONLY_LOAD_DETECTION"] = "1"
    
    # Try to import and patch torch
    try:
        import torch
        try:
            import ultralytics.nn.tasks
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            logger.info("Added ultralytics.nn.tasks.DetectionModel to safe globals")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not patch ultralytics safe globals: {str(e)}")
    except ImportError:
        logger.warning("Could not import torch for patching")
    
    # Attempt to reload
    model = load_model_safe()
    
    # Return status
    return jsonify({
        'success': model_loaded,
        'model_loaded': model_loaded,
        'error': model_loading_error
    })

# Initialize the application
logger.info("Initializing application...")
try:
    # Pre-load model at startup
    logger.info("Pre-loading model...")
    model = load_model_safe()
    if model is None:
        logger.warning("Model pre-loading failed, will try again during first request")
    else:
        logger.info("Model pre-loaded successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())

# For local testing
if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 