import os
import threading
import time
import importlib.util
from flask import Flask, jsonify, request, redirect, url_for

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
        # Instead of JSON, show a nice loading page
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv8 Object Detection - Starting</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background: linear-gradient(135deg, #1a2a3a 0%, #0f2027 100%);
                    color: white;
                    text-align: center;
                }
                .loader {
                    border: 16px solid #f3f3f3;
                    border-top: 16px solid #3498db;
                    border-radius: 50%;
                    width: 80px;
                    height: 80px;
                    animation: spin 2s linear infinite;
                    margin: 0 auto 20px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div>
                <div class="loader"></div>
                <h1>YOLOv8 Object Detection</h1>
                <p>Application is starting, please wait...</p>
                <p>This may take up to 1-2 minutes while the AI models load.</p>
                <p>The page will refresh automatically.</p>
            </div>
        </body>
        </html>
        """, 503
    
    # Enable cloud mode for the main app
    os.environ['CLOUD_MODE'] = 'true'
    
    # Once ready, forward the request to the main app
    return MAIN_APP.full_dispatch_request()

def load_main_app():
    """Load the main Flask application in a background thread"""
    global MAIN_APP, MAIN_APP_READY
    
    # Give some time for the health check to respond first
    time.sleep(2)
    
    try:
        print("Starting to load the main application...")
        
        # Set cloud mode environment variable before importing
        os.environ['CLOUD_MODE'] = 'true'
        print("CLOUD_MODE environment variable set to 'true'")
        
        # Import the main app
        import flaskapp
        MAIN_APP = flaskapp.app
        
        # Ensure the app knows it's in cloud mode
        with MAIN_APP.app_context():
            print("Application imported, setting up paths...")
            # Ensure upload folder exists
            os.makedirs(os.path.join(os.getcwd(), 'static', 'files'), exist_ok=True)
        
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