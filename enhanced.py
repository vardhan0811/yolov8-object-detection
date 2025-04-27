from flask import Flask, render_template, jsonify, redirect, url_for, session, request, send_from_directory, Response, flash
import os
import sys
import glob
import logging
import threading
import time
import json
from werkzeug.utils import secure_filename
import numpy as np
import base64
from logging import FileHandler, WARNING
from os.path import isfile, join
import shutil
import platform
from pathlib import Path
import traceback

# Import model downloader functionality 
try:
    from download_models import ensure_yolo_weights
except ImportError:
    # Define a fallback function if the module is not available
    def ensure_yolo_weights():
        print("WARNING: download_models module not found, skipping model check")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure YOLO models are available
try:
    logger.info("Checking for required YOLO model files...")
    ensure_yolo_weights()
    logger.info("YOLO model check completed")
except Exception as e:
    logger.error(f"Error checking YOLO weights: {str(e)}")
    logger.error(traceback.format_exc())

# Attempt OpenCV import - some deployment environments may have issues
try:
    import cv2
    logger.info(f"Successfully imported OpenCV version: {cv2.__version__}")
    HAS_OPENCV = True
except ImportError as e:
    logger.error(f"Failed to import OpenCV: {str(e)}")
    logger.error(traceback.format_exc())
    HAS_OPENCV = False

# Log startup information
logger.info("=" * 50)
logger.info("Starting enhanced application")
logger.info(f"Current working directory: {os.getcwd()}")

# Add the Flask application directory to path
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlaskTutorial_YOLOv8_Web_PPE")
logger.info(f"Application directory: {app_dir}")

if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Define static and template folders
template_folder = os.path.join(app_dir, 'templates')
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app_static_folder = os.path.join(app_dir, 'static')
github_static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'github_upload', 'static')

logger.info(f"Template folder: {template_folder}")
logger.info(f"Static folder: {static_folder}")
logger.info(f"App static folder: {app_static_folder}")
logger.info(f"GitHub static folder: {github_static_folder}")

# Check if template and static folders exist
logger.info(f"Templates directory exists: {os.path.exists(template_folder)}")
logger.info(f"Static directory exists: {os.path.exists(static_folder)}")
logger.info(f"App static directory exists: {os.path.exists(app_static_folder)}")
logger.info(f"GitHub static directory exists: {os.path.exists(github_static_folder)}")

# List static files for debugging
if os.path.exists(static_folder):
    logger.info("Static folder contents:")
    for root, dirs, files in os.walk(static_folder):
        for file in files:
            logger.info(f"  {os.path.join(root, file)}")

# Create Flask app
app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=static_folder)

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cyberbot')
app.config['UPLOAD_FOLDER'] = os.path.join(static_folder, 'files')
app.config['APP_STATIC_FOLDER'] = app_static_folder
app.config['GITHUB_STATIC_FOLDER'] = github_static_folder

# Global variables for model loading
model_loading_thread = None
models_loaded = False
model_loading_error = None

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Basic health check endpoint
@app.route('/health')
def health():
    return "healthy", 200

@app.route('/healthz')
def healthz():
    return "healthy", 200

@app.route('/ping')
def ping():
    return "pong", 200

# Landing page
@app.route('/')
@app.route('/home')
def home():
    try:
        logger.info("Rendering home page template")
        # Add a version parameter to force reload of static assets
        version = int(time.time())  # Use current timestamp as version
        return render_template('indexproject.html', models_loaded=models_loaded, loading_error=model_loading_error, version=version)
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return f"""
        <h1>Enhanced App Loading</h1>
        <p>The application is starting up. Template error: {str(e)}</p>
        <p>Working directory: {os.getcwd()}</p>
        <p>App directory: {app_dir}</p>
        <p>Template folder: {app.template_folder}</p>
        """

# Serve static files directly from any path if needed
@app.route('/static/<path:filename>')
def custom_static(filename):
    logger.info(f"Request for static file: {filename}")
    # First try our static folder
    if os.path.exists(os.path.join(static_folder, filename)):
        return send_from_directory(static_folder, filename)
    # Then try the github static folder
    elif os.path.exists(os.path.join(github_static_folder, filename)):
        return send_from_directory(github_static_folder, filename)
    # If not found, try the app's static folder
    elif os.path.exists(os.path.join(app_static_folder, filename)):
        return send_from_directory(app_static_folder, filename)
    else:
        logger.error(f"Static file not found: {filename}")
        return f"File not found: {filename}", 404

# Serve static files from nested 'images' folder if needed
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    logger.info(f"Request for image file: {filename}")
    # First check our static/images folder
    if os.path.exists(os.path.join(static_folder, 'images', filename)):
        return send_from_directory(os.path.join(static_folder, 'images'), filename)
    # Then check the github static/images folder
    elif os.path.exists(os.path.join(github_static_folder, 'images', filename)):
        return send_from_directory(os.path.join(github_static_folder, 'images'), filename)
    # If not found, check the app's static/images folder
    elif os.path.exists(os.path.join(app_static_folder, 'images', filename)):
        return send_from_directory(os.path.join(app_static_folder, 'images'), filename)
    else:
        logger.error(f"Image file not found: {filename}")
        return f"Image not found: {filename}", 404

# Lazy load YOLO models
def load_models_in_background():
    global models_loaded, model_loading_error
    
    try:
        logger.info("Starting background model loading...")
        time.sleep(5)  # Delay to allow app to start up fully
        
        # Import YOLO-related modules
        try:
            import ultralytics
            from ultralytics import YOLO
            logger.info(f"Successfully imported ultralytics version: {ultralytics.__version__}")
            
            # Define paths to models
            project_root = os.path.dirname(os.path.abspath(__file__))
            ppe_model_paths = [
                os.path.join(project_root, "YOLO-Weights", "ppe.pt"),
                os.path.join(app_dir, "YOLO-Weights", "ppe.pt")
            ]
            
            general_model_paths = [
                os.path.join(project_root, "yolov8n.pt"),
                os.path.join(app_dir, "yolov8n.pt")
            ]
            
            # Log memory usage for debugging
            try:
                import psutil
                process = psutil.Process(os.getpid())
                logger.info(f"Memory usage before model load: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            except ImportError:
                logger.info("psutil not available for memory debugging")
            
            # Railway-specific optimizations
            is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
            if is_railway:
                logger.info("Running on Railway - applying optimization settings")
                # Use lower precision models on Railway to reduce memory usage
                os.environ['YOLO_VERBOSE'] = 'False'  # Reduce logging
                YOLO_DEVICE = os.environ.get('YOLO_DEVICE', 'cpu')
                logger.info(f"Setting YOLO to use device: {YOLO_DEVICE}")
            
            # Try to load PPE model
            ppe_model_loaded = False
            
            # First check for local model files
            for path in ppe_model_paths:
                if os.path.exists(path):
                    logger.info(f"Found PPE model at: {path}, size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
                    try:
                        # Set a timeout for model loading to prevent hanging
                        import threading
                        import queue

                        def load_model(path, result_queue):
                            try:
                                # First try with weights_only=False (pre PyTorch 2.6 default)
                                try:
                                    logger.info(f"Attempting to load model with weights_only=False...")
                                    model = YOLO(path, weights_only=False)
                                    result_queue.put(("success", model))
                                except Exception as e_inner:
                                    logger.warning(f"Loading with weights_only=False failed: {str(e_inner)}")
                                    logger.info(f"Attempting to load model with weights_only=True...")
                                    # Try with weights_only=True (PyTorch 2.6+ default)
                                    model = YOLO(path, weights_only=True)
                                    result_queue.put(("success", model))
                            except Exception as e:
                                result_queue.put(("error", str(e)))

                        result_queue = queue.Queue()
                        load_thread = threading.Thread(target=load_model, args=(path, result_queue))
                        load_thread.daemon = True
                        load_thread.start()
                        
                        # Wait for the model to load with a timeout
                        try:
                            status, result = result_queue.get(timeout=60)  # 60 second timeout
                            if status == "success":
                                ppe_model = result
                                logger.info("PPE model loaded successfully from local file")
                                # Run a small inference to ensure model is working
                                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                                _ = ppe_model(dummy_img, verbose=False)
                                logger.info("PPE model inference test successful")
                                ppe_model_loaded = True
                                break
                            else:
                                logger.error(f"Error in thread loading PPE model: {result}")
                        except queue.Empty:
                            logger.error(f"Timeout while loading PPE model from {path}")
                    except Exception as e:
                        logger.error(f"Failed to load PPE model from {path}: {str(e)}")
            
            # If local models failed, try loading from hub
            if not ppe_model_loaded and not is_railway:  # Skip downloading on Railway to save bandwidth/time
                try:
                    logger.info("Attempting to download model from Ultralytics hub...")
                    # Use yolov8n as fallback since we don't have a PPE model in the hub
                    ppe_model = YOLO("yolov8n")
                    logger.info("Downloaded and loaded model from hub successfully")
                    ppe_model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load model from hub: {str(e)}")
            
            # Try to load general model
            general_model_loaded = False
            
            # First check local files
            for path in general_model_paths:
                if os.path.exists(path):
                    logger.info(f"Found general model at: {path}, size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
                    try:
                        # Set a timeout for model loading
                        import threading
                        import queue

                        def load_model(path, result_queue):
                            try:
                                # First try with weights_only=False (pre PyTorch 2.6 default)
                                try:
                                    logger.info(f"Attempting to load model with weights_only=False...")
                                    model = YOLO(path, weights_only=False)
                                    result_queue.put(("success", model))
                                except Exception as e_inner:
                                    logger.warning(f"Loading with weights_only=False failed: {str(e_inner)}")
                                    logger.info(f"Attempting to load model with weights_only=True...")
                                    # Try with weights_only=True (PyTorch 2.6+ default)
                                    model = YOLO(path, weights_only=True)
                                    result_queue.put(("success", model))
                            except Exception as e:
                                result_queue.put(("error", str(e)))

                        result_queue = queue.Queue()
                        load_thread = threading.Thread(target=load_model, args=(path, result_queue))
                        load_thread.daemon = True
                        load_thread.start()
                        
                        # Wait for the model to load with a timeout
                        try:
                            status, result = result_queue.get(timeout=60)  # 60 second timeout
                            if status == "success":
                                general_model = result
                                logger.info("General model loaded successfully from local file")
                                # Run a small inference to ensure model is working
                                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                                _ = general_model(dummy_img, verbose=False)
                                logger.info("General model inference test successful")
                                general_model_loaded = True
                                break
                            else:
                                logger.error(f"Error in thread loading general model: {result}")
                        except queue.Empty:
                            logger.error(f"Timeout while loading general model from {path}")
                    except Exception as e:
                        logger.error(f"Failed to load general model from {path}: {str(e)}")
            
            # If local models failed and not on Railway, try loading from hub
            if not general_model_loaded and not is_railway:
                try:
                    logger.info("Attempting to download general model from Ultralytics hub...")
                    general_model = YOLO("yolov8n")
                    logger.info("Downloaded and loaded general model from hub successfully")
                    general_model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load general model from hub: {str(e)}")
            
            # Log memory after model load
            try:
                import psutil
                process = psutil.Process(os.getpid())
                logger.info(f"Memory usage after model load: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            except ImportError:
                pass
            
            # If at least one model loaded, consider it a success
            if ppe_model_loaded or general_model_loaded:
                models_loaded = True
                logger.info("At least one model loaded successfully")
            else:
                model_loading_error = "Failed to load any models from local files or hub"
                logger.error(model_loading_error)
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            model_loading_error = f"Failed to import required modules: {str(e)}"
        
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        model_loading_error = str(e)
        models_loaded = False
        
    # Force models_loaded to True for testing on Railway
    if os.environ.get('RAILWAY_ENVIRONMENT') is not None:
        logger.info("Setting models_loaded flag for Railway environment")
        # Only set to true if there was no critical error
        if model_loading_error is None or "Failed to import" not in model_loading_error:
            logger.info("Forcing models_loaded to True for Railway testing")
            models_loaded = True
        else:
            logger.error(f"Critical error prevented model loading: {model_loading_error}")

# Video upload route
@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    # Check if forcing models_loaded for testing on Railway
    railway_env = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    # Print debugging info
    print(f"=== Front Page Request ===")
    print(f"Railway environment: {railway_env}")
    print(f"Request method: {request.method}")
    if request.method == 'POST':
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
    
    if not models_loaded and not railway_env:
        if model_loading_error:
            return f"""
            <h1>Model Loading Error</h1>
            <p>The YOLO models failed to load: {model_loading_error}</p>
            <p><a href="/">Return to home</a></p>
            <p><a href="/debug">View debug information</a></p>
            """
        else:
            return """
            <h1>Models Loading</h1>
            <p>The YOLO models are still loading. Please wait a moment and refresh the page.</p>
            <p><a href="/">Return to home</a></p>
            """
    
    # Import form classes from the main app
    try:
        # Only import these if models are loaded or we're on Railway
        logger.info("Importing required modules for video upload...")
        
        # First check if OpenCV is available
        try:
            import cv2
            logger.info(f"Successfully imported OpenCV version: {cv2.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import OpenCV: {str(e)}")
            
            # More specific error handling for Railway
            if "libGL.so.1" in str(e):
                # This is the common OpenCV error on Railway
                return f"""
                <h1>System Dependency Error</h1>
                <p>The application is missing required system libraries for OpenCV: {str(e)}</p>
                <p>This is a common issue on cloud deployments. The system needs additional libraries.</p>
                
                <h2>Troubleshooting Steps for Railway:</h2>
                <ol>
                    <li>Check your Dockerfile or railway.json configuration</li>
                    <li>Ensure you've installed all required system packages:
                        <pre>libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libxcb-shm0 libxcb-xfixes0 ffmpeg</pre>
                    </li>
                    <li>Try using the Dockerfile deployment method which has more explicit dependency handling</li>
                    <li>Visit <a href="/test_opencv">the OpenCV test page</a> for additional diagnostics</li>
                </ol>
                
                <p><a href="/">Return to home</a></p>
                <p><a href="/debug">View debug information</a></p>
                """
            else:
                return f"""
                <h1>System Dependency Error</h1>
                <p>The application is missing required system libraries for OpenCV: {str(e)}</p>
                <p>This is a common issue on cloud deployments. The administrator needs to install the following packages:</p>
                <pre>libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6</pre>
                <p><a href="/">Return to home</a></p>
                <p><a href="/debug">View debug information</a></p>
                """
        
        # Try to import the rest of our dependencies
        try:
            # Try to use a direct import approach for Railway 
            if railway_env:
                # For Railway deployment - direct approach without relying on package structure
                from flask_wtf import FlaskForm
                from flask_wtf.file import FileField, FileRequired, FileAllowed
                from wtforms import RadioField, SubmitField, SelectField
                from werkzeug.utils import secure_filename
                
                class UploadFileForm(FlaskForm):
                    file = FileField("File", validators=[
                        FileRequired(),
                        FileAllowed(['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'], 'Images and videos only!')
                    ])
                    model_type = RadioField("Detection Model", 
                                choices=[('ppe', 'PPE Detection (Protective Equipment)'), 
                                        ('general', 'General Object Detection (People, Cars, etc.)')],
                                default='ppe')
                    submit = SubmitField("Run Detection")
                
                # Use the local YOLO_Video module
                from YOLO_Video import video_detection
                logger.info("Using direct imports for Railway environment")
            else:
                # Regular import method for local development
                from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import UploadFileForm
                from FlaskTutorial_YOLOv8_Web_PPE.YOLO_Video import video_detection
                logger.info("Using package imports for local environment")
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return f"""
            <h1>Module Import Error</h1>
            <p>Failed to import required modules: {str(e)}</p>
            <p><a href="/">Return to home</a></p>
            <p><a href="/debug">View debug information</a></p>
            """
        
        # Create form instance
        logger.info("Creating upload form instance...")
        form = UploadFileForm()
        
        if request.method == 'POST':
            logger.info(f"POST request received. Form data: {request.form}")
            logger.info(f"Files: {request.files}")
            
            # Check if it's a regular form submit or if we need to handle direct file uploads
            has_file = False
            file = None
            model_type = 'general'  # Default model type
            
            # Check if file exists in request.files
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename != '':
                    has_file = True
                    logger.info(f"Found file in request.files: {file.filename}")
                    # Get model type from form
                    if 'model_type' in request.form:
                        model_type = request.form['model_type']
                    logger.info(f"Model type from form data: {model_type}")
            
            # Try to validate form normally
            if form.validate_on_submit():
                logger.info("Form validated successfully")
                file = form.file.data
                model_type = form.model_type.data
                has_file = True
            elif has_file:
                # We found a file but the form didn't validate - proceed with direct handling
                logger.info("Form did not validate but we have a file - using direct handling")
            else:
                logger.info(f"Form validation failed. Errors: {form.errors}")
                # If there are specific CSRF errors, let's handle them specially for Railway
                if form.errors and 'csrf_token' in form.errors:
                    logger.warning("CSRF token validation failed - this can happen on Railway")
                    flash("CSRF validation failed. This can happen in cloud environments. Try again.", "warning")
            
            # Process file if we have one
            if has_file and file and file.filename != '':
                # Process file upload similar to the original app
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logger.info(f"Saving uploaded file to {file_path}")
                
                # Ensure upload folder exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                try:
                    file.save(file_path)
                    logger.info(f"File saved successfully to {file_path}")
                    
                    # Save the file information in session for the video route
                    session['video_path'] = file_path
                    session['model_type'] = model_type
                    
                    # Redirect to the video page
                    logger.info("Redirecting to video page")
                    return redirect(url_for('video'))
                except Exception as e:
                    logger.error(f"Error saving file: {str(e)}")
                    return f"""
                    <h1>File Upload Error</h1>
                    <p>Error saving file: {str(e)}</p>
                    <p><a href="{url_for('front')}">Try again</a></p>
                    """
        
        # Render the template with the form
        logger.info("Rendering video upload form template")
        try:
            # Try to use the normal template
            return render_template('videoprojectnew.html', form=form)
        except Exception as template_error:
            logger.error(f"Error rendering template: {str(template_error)}")
            
            # Fallback to a simpler template we'll create on the fly
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Upload</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ font-family: Arial, sans-serif; background-color: #1a2a3a; color: white; margin: 0; padding: 20px; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }}
                    h1 {{ color: #3498db; }}
                    .form-group {{ margin-bottom: 15px; }}
                    label {{ display: block; margin-bottom: 5px; }}
                    input[type="file"], select {{ width: 100%; padding: 8px; background: rgba(255,255,255,0.8); color: #333; border: none; border-radius: 4px; }}
                    button {{ background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }}
                    button:hover {{ background: #2980b9; }}
                    .radio-group {{ display: flex; gap: 20px; }}
                    .radio-option {{ display: flex; align-items: center; }}
                    .radio-option input {{ margin-right: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Upload Video for Object Detection</h1>
                    
                    <form method="POST" enctype="multipart/form-data">
                        <!-- CSRF token -->
                        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                        
                        <div class="form-group">
                            <label for="file">Choose File:</label>
                            <input type="file" id="file" name="file" accept=".jpg,.jpeg,.png,.mp4,.avi,.mov,.mkv">
                        </div>
                        
                        <div class="form-group">
                            <label>Select Detection Model:</label>
                            <div class="radio-group">
                                <div class="radio-option">
                                    <input type="radio" id="ppe" name="model_type" value="ppe" checked>
                                    <label for="ppe">PPE Detection (Protective Equipment)</label>
                                </div>
                                <div class="radio-option">
                                    <input type="radio" id="general" name="model_type" value="general">
                                    <label for="general">General Object Detection (People, Cars, etc.)</label>
                                </div>
                            </div>
                        </div>
                        
                        <button type="submit">Process File</button>
                    </form>
                    
                    <div style="margin-top: 20px;">
                        <p>Supported formats: MP4, AVI, MOV, MKV, JPG, PNG, JPEG</p>
                        <p><a href="/" style="color: #3498db;">Return to home</a></p>
                    </div>
                </div>
            </body>
            </html>
            """
    
    except Exception as e:
        logger.error(f"Error in front route: {str(e)}")
        return f"""
        <h1>Video Upload Feature Error</h1>
        <p>An error occurred: {str(e)}</p>
        <p>This may be due to missing dependencies or configuration issues.</p>
        <p><a href="/">Return to home</a></p>
        <p><a href="/debug">View debug information</a></p>
        """

# Video stream route
@app.route('/video')
def video():
    try:
        # Debug info
        print(f"=== Video Route Request ===")
        print(f"Session data: {session}")
        
        # Check if forcing models_loaded for testing on Railway
        railway_env = os.environ.get('RAILWAY_ENVIRONMENT') is not None
        print(f"Railway environment: {railway_env}")
        
        if not models_loaded and not railway_env:
            return """
            <h1>Models Loading</h1>
            <p>The YOLO models are still loading. Please wait a moment and refresh the page.</p>
            <p><a href="/">Return to home</a></p>
            """
        
        # Get path and model type from session
        video_path = session.get('video_path', '')
        model_type = session.get('model_type', 'ppe')
        
        logger.info(f"Video route called with video_path: {video_path}, model_type: {model_type}")
        print(f"Video path: {video_path}")
        print(f"Model type: {model_type}")
        
        if not video_path:
            logger.warning("No video path in session, redirecting to upload form")
            return redirect(url_for('front'))
        
        # Check if the file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            # Try to look for the file in a different location (in case of Railway deployment path issues)
            potential_paths = [
                video_path,
                os.path.join(os.getcwd(), os.path.basename(video_path)),
                os.path.join('uploads', os.path.basename(video_path))
            ]
            
            found_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    logger.info(f"Found video file at alternative path: {path}")
                    found_path = path
                    break
            
            if found_path:
                video_path = found_path
                # Update session with new path
                session['video_path'] = video_path
            else:
                return f"""
                <h1>Video File Not Found</h1>
                <p>The video file could not be found. It may have been deleted or moved.</p>
                <p>Looked in these locations: {', '.join(potential_paths)}</p>
                <p><a href="{url_for('front')}">Upload another video</a></p>
                <p><a href="/">Return to home</a></p>
                """
        
        # Import the generate_frames function
        try:
            if railway_env:
                logger.info("Using direct import of generate_frames for Railway")
                # Define a simple generate_frames function that uses our video_detection function
                from YOLO_Video import video_detection
                
                def generate_frames(path_x, model_type='ppe'):
                    try:
                        logger.info(f"Starting video detection with path: {path_x}, model: {model_type}")
                        # Process frames using our video_detection function
                        for frame in video_detection(path_x, model_type):
                            try:
                                # Convert the frame to JPEG
                                ret, buffer = cv2.imencode('.jpg', frame)
                                if not ret:
                                    logger.error("Failed to encode frame")
                                    continue
                                    
                                # Convert to bytes and yield
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            except Exception as frame_error:
                                logger.error(f"Error processing frame: {str(frame_error)}")
                                # Create error frame
                                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(error_frame, "Error processing frame", (50, 240), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                ret, buffer = cv2.imencode('.jpg', error_frame)
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    except Exception as e:
                        logger.error(f"Error in generate_frames: {str(e)}")
                        # Create error frame
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, f"Error: {str(e)[:50]}", (20, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.info("Using FlaskTutorial_YOLOv8_Web_PPE.flaskapp.generate_frames")
                from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import generate_frames
            
            # Return the video feed
            logger.info("Starting video processing")
            return Response(generate_frames(video_path, model_type),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
                          
        except ImportError as e:
            logger.error(f"Failed to import generate_frames: {str(e)}")
            return f"""
            <h1>Module Import Error</h1>
            <p>Failed to import video processing module: {str(e)}</p>
            <p><a href="/">Return to home</a></p>
            """
            
    except Exception as e:
        logger.error(f"Unexpected error in video route: {str(e)}")
        return f"""
        <h1>Error Processing Video</h1>
        <p>An unexpected error occurred: {str(e)}</p>
        <p><a href="{url_for('front')}">Try uploading another video</a></p>
        <p><a href="/">Return to home</a></p>
        """

# Webcam route
@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    if not models_loaded:
        if model_loading_error:
            return f"""
            <h1>Model Loading Error</h1>
            <p>The YOLO models failed to load: {model_loading_error}</p>
            <p><a href="/">Return to home</a></p>
            """
        else:
            return """
            <h1>Models Loading</h1>
            <p>The YOLO models are still loading. Please wait a moment and refresh the page.</p>
            <p><a href="/">Return to home</a></p>
            """
    
    # Import form classes from the main app
    try:
        from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import WebcamSelectForm
        
        # Check if in cloud environment
        is_cloud = os.environ.get('RAILWAY_ENVIRONMENT') is not None
        
        if is_cloud:
            return """
            <h1>Webcam Feature Restricted</h1>
            <p>Webcam functionality is not available in cloud deployments like Railway.</p>
            <p>Please use the video upload feature instead.</p>
            <p><a href="/">Return to home</a></p>
            """
        
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
    
    except Exception as e:
        logger.error(f"Error in webcam route: {str(e)}")
        return f"""
        <h1>Webcam Feature Error</h1>
        <p>An error occurred: {str(e)}</p>
        <p><a href="/">Return to home</a></p>
        """

# Webcam feed
@app.route('/webapp')
def webapp():
    try:
        # Get parameters
        camera_id = session.get('camera_id', 0)
        model_type = session.get('model_type', 'ppe')
        
        from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import generate_frames_web
        
        return Response(generate_frames_web(camera_id, model_type),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        logger.error(f"Error in webapp route: {str(e)}")
        return f"Error streaming webcam: {str(e)}"

# Debug information
@app.route('/debug')
def debug():
    # Memory information
    memory_info = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "system_total_mb": psutil.virtual_memory().total / 1024 / 1024,
            "system_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "system_percent": psutil.virtual_memory().percent
        }
    except ImportError:
        memory_info = {"error": "psutil not installed"}
    
    # Check static files
    static_files = []
    if os.path.exists(static_folder):
        for root, dirs, files in os.walk(static_folder):
            rel_dir = os.path.relpath(root, static_folder)
            for file in files:
                static_files.append(os.path.join(rel_dir, file))
    
    # Check template files
    template_files = []
    if os.path.exists(template_folder):
        for root, dirs, files in os.walk(template_folder):
            rel_dir = os.path.relpath(root, template_folder)
            for file in files:
                template_files.append(os.path.join(rel_dir, file))
    
    # Check GitHub static files
    github_static_files = []
    if os.path.exists(github_static_folder):
        for root, dirs, files in os.walk(github_static_folder):
            rel_dir = os.path.relpath(root, github_static_folder)
            for file in files:
                github_static_files.append(os.path.join(rel_dir, file))
    
    # Get YOLO model paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    ppe_model_paths = [
        os.path.join(project_root, "YOLO-Weights", "ppe.pt"),
        os.path.join(app_dir, "YOLO-Weights", "ppe.pt")
    ]
    
    general_model_paths = [
        os.path.join(project_root, "yolov8n.pt"),
        os.path.join(app_dir, "yolov8n.pt")
    ]
    
    model_paths = {
        "ppe_model_candidates": ppe_model_paths,
        "general_model_candidates": general_model_paths,
        "ppe_model_exists": any(os.path.exists(p) for p in ppe_model_paths),
        "general_model_exists": any(os.path.exists(p) for p in general_model_paths),
    }
    
    # Add file sizes if they exist
    for path_type, paths in [("ppe_model_sizes", ppe_model_paths), ("general_model_sizes", general_model_paths)]:
        model_paths[path_type] = {}
        for path in paths:
            if os.path.exists(path):
                model_paths[path_type][path] = os.path.getsize(path) / 1024 / 1024  # Size in MB
    
    # Python package versions
    package_versions = {}
    try:
        import pip
        for package in ['flask', 'werkzeug', 'ultralytics', 'opencv-python-headless', 'numpy']:
            try:
                pkg = __import__(package.replace('-', '_'))
                package_versions[package] = getattr(pkg, '__version__', 'unknown')
            except ImportError:
                package_versions[package] = "not installed"
    except ImportError:
        package_versions = {"error": "pip not available"}
    
    # Railway specific info
    railway_info = {}
    for key, value in os.environ.items():
        if key.startswith('RAILWAY_'):
            railway_info[key] = value
    
    debug_info = {
        "app_config": {
            "template_folder": app.template_folder,
            "static_folder": app.static_folder,
            "app_static_folder": app.config.get('APP_STATIC_FOLDER'),
            "github_static_folder": app.config.get('GITHUB_STATIC_FOLDER'),
            "upload_folder": app.config.get('UPLOAD_FOLDER'),
            "secret_key_set": bool(app.config.get('SECRET_KEY')),
        },
        "memory_info": memory_info,
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd(),
            "python_path": sys.path,
            "package_versions": package_versions,
            "environment_variables": {k: v for k, v in os.environ.items() 
                                    if not k.startswith(('AWS', 'RAILWAY_', 'SECRET'))}
        },
        "railway_environment": railway_info,
        "directory_structure": {
            "templates_exist": os.path.exists(app.template_folder),
            "static_exists": os.path.exists(app.static_folder),
            "app_static_exists": os.path.exists(app.config.get('APP_STATIC_FOLDER', '')),
            "github_static_exists": os.path.exists(app.config.get('GITHUB_STATIC_FOLDER', '')),
            "upload_folder_exists": os.path.exists(app.config.get('UPLOAD_FOLDER', '')),
            "sample_static_files": static_files[:20] if len(static_files) > 20 else static_files,
            "sample_template_files": template_files,
            "sample_github_static_files": github_static_files[:20] if len(github_static_files) > 20 else github_static_files,
        },
        "yolo_status": {
            "models_loaded": models_loaded,
            "model_loading_error": model_loading_error,
            "model_paths": model_paths,
            "railway_environment": os.environ.get('RAILWAY_ENVIRONMENT') is not None
        }
    }
    
    # Return a simple HTML page with the debug info if the request is from a browser
    if request.headers.get('Accept', '').find('text/html') >= 0:
        html = f"""
        <html>
        <head>
            <title>Debug Information</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .back {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Debug Information</h1>
            <pre>{json.dumps(debug_info, indent=2)}</pre>
            <div class="back">
                <a href="/">Return to home</a>
            </div>
        </body>
        </html>
        """
        return html
    
    return jsonify(debug_info)

# A route to help verify OpenCV installation 
@app.route('/test_opencv')
def test_opencv():
    logger.info("Testing OpenCV installation")
    try:
        import cv2
        
        # Test that OpenCV is functioning with the key libraries
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', img)
        
        # Return success if everything works
        opencv_info = {
            "version": cv2.__version__,
            "build_info": cv2.getBuildInformation()
        }
        
        # Extract key build information for troubleshooting
        build_summary = {}
        for line in opencv_info["build_info"].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                build_summary[key.strip()] = value.strip()
        
        # Create an HTML response
        html = f"""
        <html>
        <head>
            <title>OpenCV Test</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: green; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .back {{ margin-top: 20px; }}
                img {{ border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>OpenCV is working correctly!</h1>
            <p>Version: {opencv_info["version"]}</p>
            
            <h2>Test Image:</h2>
            <img src="data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" alt="Test Image">
            
            <h2>Key Build Information:</h2>
            <pre>{json.dumps(build_summary, indent=2)}</pre>
            
            <div class="back">
                <a href="/">Return to home</a>
            </div>
        </body>
        </html>
        """
        return html
        
    except ImportError as e:
        logger.error(f"OpenCV import error: {str(e)}")
        return f"""
        <h1>OpenCV Import Error</h1>
        <p>Failed to import OpenCV: {str(e)}</p>
        <p>This usually indicates missing system dependencies.</p>
        
        <h2>Troubleshooting Steps:</h2>
        <ol>
            <li>Check that all required system packages are installed:
                <pre>libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libxcb-shm0 libxcb-xfixes0</pre>
            </li>
            <li>Verify that opencv-python-headless is installed:
                <pre>pip install opencv-python-headless==4.8.1.78</pre>
            </li>
            <li>Review Railway deployment logs for any errors during installation</li>
        </ol>
        
        <p><a href="/">Return to home</a></p>
        <p><a href="/debug">View debug information</a></p>
        """
    except Exception as e:
        logger.error(f"OpenCV test error: {str(e)}")
        return f"""
        <h1>OpenCV Error</h1>
        <p>Error testing OpenCV functionality: {str(e)}</p>
        <p><a href="/">Return to home</a></p>
        <p><a href="/debug">View debug information</a></p>
        """

@app.route('/diagnose_models')
def diagnose_models():
    """Route to diagnose model loading issues"""
    logger.info("Diagnosing model loading...")
    
    model_results = []
    
    try:
        from ultralytics import YOLO
        import torch
        import torchvision
        
        torch_info = {
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__
        }
        
        # Find all model files
        model_files = []
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".pt"):
                    model_path = os.path.join(root, file)
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    model_files.append({
                        "path": model_path,
                        "size_mb": round(size_mb, 2)
                    })
        
        # Test each model
        for model_file in model_files:
            test_results = {"path": model_file["path"], "size_mb": model_file["size_mb"], "tests": []}
            
            # Test with weights_only=False
            try:
                logger.info(f"Testing model {model_file['path']} with weights_only=False")
                model = YOLO(model_file["path"], weights_only=False)
                test_results["tests"].append({
                    "option": "weights_only=False",
                    "success": True,
                    "error": None
                })
            except Exception as e:
                logger.error(f"Error loading model with weights_only=False: {str(e)}")
                test_results["tests"].append({
                    "option": "weights_only=False",
                    "success": False,
                    "error": str(e)
                })
            
            # Test with weights_only=True
            try:
                logger.info(f"Testing model {model_file['path']} with weights_only=True")
                model = YOLO(model_file["path"], weights_only=True)
                test_results["tests"].append({
                    "option": "weights_only=True",
                    "success": True,
                    "error": None
                })
            except Exception as e:
                logger.error(f"Error loading model with weights_only=True: {str(e)}")
                test_results["tests"].append({
                    "option": "weights_only=True",
                    "success": False,
                    "error": str(e)
                })
            
            # Test with defaults
            try:
                logger.info(f"Testing model {model_file['path']} with defaults")
                model = YOLO(model_file["path"])
                test_results["tests"].append({
                    "option": "defaults",
                    "success": True,
                    "error": None
                })
            except Exception as e:
                logger.error(f"Error loading model with defaults: {str(e)}")
                test_results["tests"].append({
                    "option": "defaults",
                    "success": False,
                    "error": str(e)
                })
            
            model_results.append(test_results)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to run model diagnostics"
        }), 500
    
    # Return HTML response
    return render_template_string("""
    <html>
    <head>
        <title>Model Diagnostics</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .model { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .success { color: green; }
            .failure { color: red; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .back { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Model Diagnostics</h1>
        
        <h2>PyTorch Information</h2>
        <p>PyTorch Version: {{ torch_info.pytorch_version }}</p>
        <p>Torchvision Version: {{ torch_info.torchvision_version }}</p>
        
        <h2>Model Files ({{ model_results|length }})</h2>
        {% if model_results %}
            {% for model in model_results %}
                <div class="model">
                    <h3>{{ model.path }} ({{ model.size_mb }} MB)</h3>
                    <ul>
                        {% for test in model.tests %}
                            <li class="{{ 'success' if test.success else 'failure' }}">
                                <strong>{{ test.option }}:</strong> 
                                {% if test.success %}
                                    Success
                                {% else %}
                                    Failed - {{ test.error }}
                                {% endif %}
                            </li>
                        {% endfor %}
                    </ul>
                </div>
            {% endfor %}
        {% else %}
            <p>No model files found in the deployment.</p>
        {% endif %}
        
        <div class="back">
            <a href="/">Return to home</a>
        </div>
    </body>
    </html>
    """, torch_info=torch_info, model_results=model_results)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    logger.info("Upload image endpoint called")
    
    # Check if a file was provided
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If user submits an empty form
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'})
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        logger.warning(f"File {file.filename} has invalid extension")
        return jsonify({'error': 'File type not allowed'})
    
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        logger.info(f"Creating upload folder: {app.config['UPLOAD_FOLDER']}")
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    try:
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving uploaded file to {filepath}")
        file.save(filepath)
        
        try:
            detected_image, _ = detect_objects_on_image(filepath)
            detected_image_filename = "detected_" + filename
            detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_image_filename)
            logger.info(f"Saving detected image to {detected_image_path}")
            cv2.imwrite(detected_image_path, detected_image)
            
            return jsonify({
                'success': True,
                'original_image': filename,
                'detected_image': detected_image_filename
            })
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f"Error during detection: {str(e)}",
                'traceback': traceback.format_exc()
            })
            
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Error handling file: {str(e)}"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    logger.info("Upload video endpoint called")
    
    # Check if a file was provided
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If user submits an empty form
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'})
    
    # Check if the file is allowed
    if not allowed_video_file(file.filename):
        logger.warning(f"File {file.filename} has invalid extension")
        return jsonify({'error': 'Video file type not allowed'})
    
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        logger.info(f"Creating upload folder: {app.config['UPLOAD_FOLDER']}")
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    try:
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving uploaded video to {filepath}")
        file.save(filepath)
        
        # Process video (your video processing function)
        try:
            # Find suitable model for detection
            model_paths = find_yolo_model_paths()
            if not model_paths:
                logger.error("No YOLO model files found")
                return jsonify({'error': 'No YOLO model files available'})
                
            logger.info(f"Found model paths: {model_paths}")
            
            # Select first available model
            model_path = model_paths[0]
            logger.info(f"Using model: {model_path}")
            
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "detected_" + filename)
            logger.info(f"Output video will be saved to {output_path}")
            
            # Call the YOLO video processing function
            from YOLO_Video import YOLO_Video
            YOLO_Video(video_path=filepath, 
                       model_path=model_path, 
                       output_path=output_path,
                       device="cpu",  # Use CPU for compatibility
                       weights_only=True)  # Try weights_only mode
            
            return jsonify({
                'success': True,
                'original_video': filename,
                'detected_video': "detected_" + filename
            })
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f"Error processing video: {str(e)}",
                'traceback': traceback.format_exc()
            })
    except Exception as e:
        logger.error(f"Error handling video upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Error handling video: {str(e)}"})

def find_yolo_model_paths():
    """Find available YOLO model files"""
    model_paths = []
    
    # Check common locations
    possible_models = ["yolov8n.pt", "ppe.pt"]
    search_locations = [
        ".",  # Current directory
        "./YOLO-Weights",  # YOLO-Weights directory
        "../YOLO-Weights",  # Up one level
        "/app/YOLO-Weights",  # Docker container location
    ]
    
    for location in search_locations:
        for model_name in possible_models:
            model_path = os.path.join(location, model_name)
            if os.path.isfile(model_path):
                logger.info(f"Found model at {model_path}")
                model_paths.append(model_path)
    
    return model_paths

@app.route('/test_model')
def test_model():
    """Test if the YOLO models can be loaded correctly"""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        # Get some system info
        import platform
        import torch
        
        # Create a response with detailed information
        info = {
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "ultralytics_version": getattr(YOLO, "__version__", "unknown"),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "railway_environment": os.environ.get('RAILWAY_ENVIRONMENT') is not None
        }
        
        # Try to load a model
        model_results = []
        
        # Test the standard path first
        try:
            model_path = os.path.join(os.getcwd(), "yolov8n.pt")
            if os.path.exists(model_path):
                start_time = time.time()
                model = YOLO(model_path)
                load_time = time.time() - start_time
                model_results.append({
                    "path": model_path,
                    "status": "success",
                    "message": f"Model loaded in {load_time:.2f} seconds",
                    "load_time_seconds": load_time
                })
            else:
                model_results.append({
                    "path": model_path,
                    "status": "not_found",
                    "message": "Model file does not exist"
                })
        except Exception as e:
            model_results.append({
                "path": model_path,
                "status": "error",
                "message": str(e)
            })
        
        # Try with direct hub loading
        try:
            start_time = time.time()
            model = YOLO("yolov8n")
            load_time = time.time() - start_time
            model_results.append({
                "path": "yolov8n (hub)",
                "status": "success",
                "message": f"Model loaded from hub in {load_time:.2f} seconds",
                "load_time_seconds": load_time
            })
        except Exception as e:
            model_results.append({
                "path": "yolov8n (hub)",
                "status": "error",
                "message": str(e)
            })
        
        # Check for all available model files
        model_files = []
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.pt'):
                    model_files.append(os.path.join(root, file))
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Model Test</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #1a2a3a; color: white; margin: 0; padding: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }}
                h1 {{ color: #3498db; }}
                h2 {{ color: #2ecc71; margin-top: 30px; }}
                pre {{ background: rgba(0,0,0,0.3); padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .code {{ font-family: monospace; }}
                .success {{ color: #2ecc71; }}
                .error {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: rgba(52, 152, 219, 0.3); }}
                tr {{ border-bottom: 1px solid rgba(255,255,255,0.1); }}
                tr:nth-child(even) {{ background-color: rgba(255,255,255,0.05); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>YOLO Model Test Results</h1>
                
                <h2>System Information</h2>
                <table>
                    <tr><th>Python Version</th><td>{info['python_version']}</td></tr>
                    <tr><th>Operating System</th><td>{info['operating_system']}</td></tr>
                    <tr><th>OpenCV Version</th><td>{info['opencv_version']}</td></tr>
                    <tr><th>NumPy Version</th><td>{info['numpy_version']}</td></tr>
                    <tr><th>Ultralytics Version</th><td>{info['ultralytics_version']}</td></tr>
                    <tr><th>PyTorch Version</th><td>{info['pytorch_version']}</td></tr>
                    <tr><th>CUDA Available</th><td>{'Yes' if info['cuda_available'] else 'No'}</td></tr>
                    <tr><th>Railway Environment</th><td>{'Yes' if info['railway_environment'] else 'No'}</td></tr>
                </table>
                
                <h2>Model Loading Tests</h2>
                <table>
                    <tr>
                        <th>Model Path</th>
                        <th>Status</th>
                        <th>Message</th>
                    </tr>
        """
        
        for result in model_results:
            status_class = "success" if result["status"] == "success" else "error"
            html_content += f"""
                    <tr>
                        <td>{result['path']}</td>
                        <td class="{status_class}">{result['status']}</td>
                        <td>{result['message']}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>Available Model Files</h2>
                <pre>
        """
        
        if model_files:
            for model_file in model_files:
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                html_content += f"{model_file} ({size_mb:.2f} MB)\n"
        else:
            html_content += "No model files found in the current directory structure."
        
        html_content += """
                </pre>
                
                <h2>Environment Variables</h2>
                <pre>
        """
        
        # Add environment variables (excluding any sensitive ones)
        for key, value in os.environ.items():
            if not any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token', 'auth']):
                html_content += f"{key}={value}\n"
        
        html_content += """
                </pre>
                
                <div style="margin-top: 30px; text-align: center;">
                    <a href="/" style="color: #3498db; text-decoration: none; margin-right: 20px;">Return to Home</a>
                    <a href="/FrontPage" style="color: #3498db; text-decoration: none;">Go to Video Upload</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Model Test Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #1a2a3a; color: white; margin: 0; padding: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }}
                h1 {{ color: #e74c3c; }}
                pre {{ background: rgba(0,0,0,0.3); padding: 15px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Error Testing YOLO Model</h1>
                <p>An error occurred while testing the YOLO model:</p>
                <pre>{str(e)}</pre>
                <p><a href="/" style="color: #3498db;">Return to Home</a></p>
            </div>
        </body>
        </html>
        """
        return error_html

if __name__ == '__main__':
    # Start background model loading
    model_loading_thread = threading.Thread(target=load_models_in_background, daemon=True)
    model_loading_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 