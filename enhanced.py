from flask import Flask, render_template, jsonify, redirect, url_for, session, request
import os
import sys

# Add the Flask application directory to path
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlaskTutorial_YOLOv8_Web_PPE")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Create Flask app
app = Flask(__name__, 
            template_folder=os.path.join(app_dir, 'templates'),
            static_folder=os.path.join(app_dir, 'static'))

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cyberbot')
app.config['UPLOAD_FOLDER'] = os.path.join(app_dir, 'static/files')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Basic health check endpoint
@app.route('/health')
def health():
    return "healthy", 200

# Landing page
@app.route('/')
@app.route('/home')
def home():
    try:
        return render_template('indexproject.html')
    except Exception as e:
        return f"""
        <h1>Enhanced App Loading</h1>
        <p>The application is starting up. Template error: {str(e)}</p>
        <p>Working directory: {os.getcwd()}</p>
        <p>App directory: {app_dir}</p>
        <p>Template folder: {app.template_folder}</p>
        """

# Placeholder for the webcam route
@app.route('/webcam')
def webcam():
    return """
    <h1>Webcam Feature</h1>
    <p>The webcam feature will be available in the next phase of deployment.</p>
    <p><a href="/">Return to home</a></p>
    """

# Placeholder for the front page
@app.route('/FrontPage')
def front():
    return """
    <h1>Upload Video Feature</h1>
    <p>The video upload feature will be available in the next phase of deployment.</p>
    <p><a href="/">Return to home</a></p>
    """

# Debug information
@app.route('/debug')
def debug():
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
        "templates_exist": os.path.exists(app.template_folder),
        "static_exists": os.path.exists(app.static_folder)
    }
    return jsonify(debug_info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 