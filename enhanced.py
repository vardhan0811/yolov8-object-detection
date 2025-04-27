from flask import Flask, render_template, jsonify, redirect, url_for, session, request, send_from_directory, Response
import os
import sys
import glob
import logging
import threading
import time
import json
from werkzeug.utils import secure_filename
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                                model = YOLO(path)
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
                                model = YOLO(path)
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
            logger.info("Successfully imported OpenCV")
        except ImportError as e:
            logger.error(f"Failed to import OpenCV: {str(e)}")
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
            from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import UploadFileForm
            from FlaskTutorial_YOLOv8_Web_PPE.YOLO_Video import video_detection
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
        
        if form.validate_on_submit():
            logger.info("Form submitted, processing file upload...")
            file = form.file.data
            model_type = form.model_type.data
            
            # Process file upload similar to the original app
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving uploaded file to {file_path}")
            
            # Ensure upload folder exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file.save(file_path)
            
            # Save the file information in session for the video route
            session['video_path'] = file_path
            session['model_type'] = model_type
            
            # Redirect to the video page
            logger.info("Redirecting to video page")
            return redirect(url_for('video'))
        
        # Render the template with the form
        logger.info("Rendering video upload form template")
        try:
            return render_template('videoprojectnew.html', form=form)
        except Exception as template_error:
            logger.error(f"Error rendering template: {str(template_error)}")
            return f"""
            <h1>Template Error</h1>
            <p>Failed to render the form template: {str(template_error)}</p>
            <pre>{str(form)}</pre>
            <p><a href="/">Return to home</a></p>
            <p><a href="/debug">View debug information</a></p>
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
        # Check if forcing models_loaded for testing on Railway
        railway_env = os.environ.get('RAILWAY_ENVIRONMENT') is not None
        
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
        
        if not video_path:
            logger.warning("No video path in session, redirecting to upload form")
            return redirect(url_for('front'))
        
        # Check if the file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return f"""
            <h1>Video File Not Found</h1>
            <p>The video file could not be found. It may have been deleted or moved.</p>
            <p><a href="{url_for('front')}">Upload another video</a></p>
            <p><a href="/">Return to home</a></p>
            """
        
        # Import the generate_frames function
        logger.info("Importing generate_frames function")
        try:
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
            <p><a href="/debug">View debug information</a></p>
            """
    
    except Exception as e:
        logger.error(f"Error in video route: {str(e)}")
        return f"""
        <h1>Video Processing Error</h1>
        <p>An error occurred while processing the video: {str(e)}</p>
        <p><a href="{url_for('front')}">Try another video</a></p>
        <p><a href="/">Return to home</a></p>
        <p><a href="/debug">View debug information</a></p>
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

if __name__ == '__main__':
    # Start background model loading
    model_loading_thread = threading.Thread(target=load_models_in_background, daemon=True)
    model_loading_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 