from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import the Blueprints
from routes.data_routes import data_bp
from routes.clean_routes import clean_bp

# Create Flask app
app = Flask(__name__)

# Configure for large file uploads - 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dataforge-secret-key-2024')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Enable CORS for all origins
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}}, supports_credentials=True)

# Register the Routes with /api prefix
app.register_blueprint(data_bp, url_prefix='/api')
app.register_blueprint(clean_bp, url_prefix='/api')

@app.route('/')
def index():
    return {"status": "DataForge API is Live", "version": "1.0.0"}

@app.route('/health')
def health():
    return {"status": "healthy"}

@app.route('/api/test-post', methods=['POST', 'GET'])
def test_post():
    """Test endpoint to verify POST method works"""
    print(f"TEST-POST HIT with method: {request.method}")
    return {"status": "ok", "method": request.method, "message": "POST is working!"}

@app.errorhandler(413)
def too_large(e):
    return {"error": "File too large. Max 500MB allowed."}, 413

# Debug: Print all registered routes on startup
def print_routes():
    print("\n📋 REGISTERED ROUTES:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        print(f"  {rule.endpoint:30s} {methods:20s} {rule.rule}")
    print("")

# For local development and production deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"\n🚀 DataForge API starting on http://localhost:{port}")
    print(f"📊 Health check: http://localhost:{port}/health")
    print(f"📁 Upload endpoint: http://localhost:{port}/api/upload\n")
    print_routes()
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
else:
    # For gunicorn/production - print routes on import
    print_routes()
