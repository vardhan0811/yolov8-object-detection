from flask import Flask
import os

# Create the absolute simplest Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from minimal app!"

@app.route('/health')
def health():
    return "healthy", 200

# Only include this if running directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 