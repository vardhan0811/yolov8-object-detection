import os
import threading
import time
import importlib.util
from flask import Flask, jsonify, request

# Create a minimal Flask app that will respond to health checks immediately
app = Flask(__name__)

# Global flag to track if the main app is ready
MAIN_APP_READY = False
MAIN_APP = None

@app.route('/health')
def health_check():
    """Health check endpoint that responds immediately"""
    return jsonify({"status": "healthy"}), 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Catch-all route that forwards to the main app once it's ready"""
    if not MAIN_APP_READY:
        return jsonify({"status": "Application is starting, please wait..."}), 503
    
    # Forward the request to the main app
    return MAIN_APP.full_dispatch_request()

def load_main_app():
    """Load the main Flask application in a background thread"""
    global MAIN_APP, MAIN_APP_READY
    
    # Give some time for the health check to respond first
    time.sleep(2)
    
    try:
        print("Starting to load the main application...")
        
        # Import the main app
        import flaskapp
        MAIN_APP = flaskapp.app
        
        # Register all routes from the main app to this app
        print("Main application loaded successfully!")
        MAIN_APP_READY = True
    except Exception as e:
        print(f"Error loading main application: {str(e)}")
        # Keep trying to load the main app
        time.sleep(10)
        load_main_app()

if __name__ == "__main__":
    # Start loading the main app in a background thread
    threading.Thread(target=load_main_app, daemon=True).start()
    
    # Start the Flask app with the health check
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 