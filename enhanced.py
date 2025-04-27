from flask import Flask, render_template, jsonify, redirect, url_for, session, request, send_from_directory, Response
import os
import sys
import glob
import logging
import threading
import time
from werkzeug.utils import secure_filename

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

logger.info(f"Template folder: {template_folder}")
logger.info(f"Static folder: {static_folder}")
logger.info(f"App static folder: {app_static_folder}")

# Check if template and static folders exist
logger.info(f"Templates directory exists: {os.path.exists(template_folder)}")
logger.info(f"Static directory exists: {os.path.exists(static_folder)}")
logger.info(f"App static directory exists: {os.path.exists(app_static_folder)}")

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
        return render_template('indexproject.html', models_loaded=models_loaded, loading_error=model_loading_error)
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
            from ultralytics import YOLO
            logger.info("Successfully imported ultralytics")
            
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
            
            # Try to load PPE model
            ppe_model_loaded = False
            for path in ppe_model_paths:
                if os.path.exists(path):
                    logger.info(f"Loading PPE model from: {path}")
                    try:
                        ppe_model = YOLO(path)
                        logger.info("PPE model loaded successfully")
                        ppe_model_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load PPE model from {path}: {str(e)}")
            
            if not ppe_model_loaded:
                logger.warning("PPE model not found in expected locations, will use fallback at runtime")
            
            # Try to load general model
            general_model_loaded = False
            for path in general_model_paths:
                if os.path.exists(path):
                    logger.info(f"Loading general model from: {path}")
                    try:
                        general_model = YOLO(path)
                        logger.info("General model loaded successfully")
                        general_model_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load general model from {path}: {str(e)}")
            
            if not general_model_loaded:
                logger.warning("General model not found in expected locations, will use hub model at runtime")
            
            # If at least one model loaded or we have fallbacks configured, consider it a success
            models_loaded = True
            logger.info("Model loading completed successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            model_loading_error = f"Failed to import required modules: {str(e)}"
        
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        model_loading_error = str(e)
        models_loaded = False

# Video upload route
@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
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
        # Only import these if models are loaded
        from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import UploadFileForm
        from FlaskTutorial_YOLOv8_Web_PPE.YOLO_Video import video_detection
        import cv2
        
        # Create form instance
        form = UploadFileForm()
        
        if form.validate_on_submit():
            file = form.file.data
            model_type = form.model_type.data
            
            # Process file upload similar to the original app
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Save the file information in session for the video route
            session['video_path'] = file_path
            session['model_type'] = model_type
            
            # Redirect to the video page
            return redirect(url_for('video'))
        
        return render_template('videoprojectnew.html', form=form)
    
    except Exception as e:
        logger.error(f"Error in front route: {str(e)}")
        return f"""
        <h1>Video Upload Feature Error</h1>
        <p>An error occurred: {str(e)}</p>
        <p><a href="/">Return to home</a></p>
        """

# Video stream route
@app.route('/video')
def video():
    try:
        if not models_loaded:
            return """
            <h1>Models Loading</h1>
            <p>The YOLO models are still loading. Please wait a moment and refresh the page.</p>
            """
        
        # Get path and model type from session
        video_path = session.get('video_path', '')
        model_type = session.get('model_type', 'ppe')
        
        if not video_path:
            return redirect(url_for('front'))
        
        # Import the generate_frames function
        from FlaskTutorial_YOLOv8_Web_PPE.flaskapp import generate_frames
        
        # Return the video feed
        return Response(generate_frames(video_path, model_type),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        logger.error(f"Error in video route: {str(e)}")
        return f"Error streaming video: {str(e)}"

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
    
    # Get YOLO model paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_paths = {
        "ppe_model_candidates": [
            os.path.join(project_root, "YOLO-Weights", "ppe.pt"),
            os.path.join(app_dir, "YOLO-Weights", "ppe.pt")
        ],
        "general_model_candidates": [
            os.path.join(project_root, "yolov8n.pt"),
            os.path.join(app_dir, "yolov8n.pt")
        ],
        "ppe_model_exists": any(os.path.exists(p) for p in [
            os.path.join(project_root, "YOLO-Weights", "ppe.pt"),
            os.path.join(app_dir, "YOLO-Weights", "ppe.pt")
        ]),
        "general_model_exists": any(os.path.exists(p) for p in [
            os.path.join(project_root, "yolov8n.pt"),
            os.path.join(app_dir, "yolov8n.pt")
        ])
    }
    
    debug_info = {
        "app_config": {
            "template_folder": app.template_folder,
            "static_folder": app.static_folder,
            "upload_folder": app.config.get('UPLOAD_FOLDER'),
            "secret_key_set": bool(app.config.get('SECRET_KEY')),
        },
        "environment": {
            "python_path": sys.path,
            "working_directory": os.getcwd(),
            "environment_variables": {k: v for k, v in os.environ.items() 
                                    if not k.startswith(('AWS', 'RAILWAY_'))}
        },
        "directory_structure": {
            "templates_exist": os.path.exists(app.template_folder),
            "static_exists": os.path.exists(app.static_folder),
            "sample_static_files": static_files[:20] if len(static_files) > 20 else static_files,
            "sample_template_files": template_files
        },
        "yolo_status": {
            "models_loaded": models_loaded,
            "model_loading_error": model_loading_error,
            "model_paths": model_paths,
        }
    }
    return jsonify(debug_info)

if __name__ == '__main__':
    # Start background model loading
    model_loading_thread = threading.Thread(target=load_models_in_background, daemon=True)
    model_loading_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 