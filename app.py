from flask import Flask, jsonify, request, Response
import os
import sys
import time
import threading
import importlib

app = Flask(__name__)

# Keep track of app info
startup_time = time.time()
app_ready = False
main_app = None
main_app_lock = threading.Lock()

# Simple status endpoint
@app.route('/')
def index():
    """Root endpoint that shows app status"""
    uptime = int(time.time() - startup_time)
    return jsonify({
        "status": "running",
        "uptime": uptime,
        "ready": app_ready,
        "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'local')
    })

# Health check endpoints
@app.route('/health')
def health():
    """Health check endpoint that always returns 200"""
    print(f"Health check called, app_ready={app_ready}")
    return jsonify({"status": "healthy"}), 200

@app.route('/healthz')
def healthz():
    """Alternative health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/ping')
def ping():
    """Simple ping endpoint"""
    return "pong", 200

@app.route('/debug')
def debug():
    """Debug endpoint to help diagnose issues"""
    debug_info = {
        "app_ready": app_ready,
        "uptime": int(time.time() - startup_time),
        "sys_path": sys.path,
        "working_directory": os.getcwd(),
        "environment": dict(os.environ),
        "python_version": sys.version
    }
    return jsonify(debug_info)

# Proxy to main app for all other routes
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    """Route all other requests to the main app once it's loaded"""
    global main_app, app_ready
    
    # If main app is not ready yet, show a loading page
    if not app_ready or main_app is None:
        return jsonify({
            "status": "loading",
            "message": "Application is starting, please wait a moment and refresh...",
            "uptime": int(time.time() - startup_time)
        }), 503
    
    # Forward the request to the main app
    try:
        return main_app.full_dispatch_request()
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error dispatching to main app: {str(e)}"
        }), 500

def load_main_app():
    """Background task to load the main app"""
    global app_ready, main_app
    
    try:
        print("Starting background load of the main application...")
        
        # Add the real app directory to the path
        app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlaskTutorial_YOLOv8_Web_PPE")
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
            
        # Wait a moment to allow health checks to pass first
        time.sleep(10)
        
        with main_app_lock:
            # Check if we're already loaded 
            if main_app is not None:
                return
                
            # Import the main Flask application
            spec = importlib.util.spec_from_file_location(
                "flaskapp", 
                os.path.join(app_dir, "flaskapp.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the Flask app instance
            main_app = module.app
            app_ready = True
            
            print("Main application loaded successfully!")
    except Exception as e:
        print(f"Error loading main application: {str(e)}")
        # You could restart the loading process here if needed

if __name__ == "__main__":
    # Start background loading
    threading.Thread(target=load_main_app, daemon=True).start()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 