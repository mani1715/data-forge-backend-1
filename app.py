from flask import Flask
from flask_cors import CORS
from config import Config
import os

# Import the Blueprints
from routes.data_routes import data_bp
from routes.clean_routes import clean_bp

app = Flask(__name__)
app.config.from_object(Config)

# Configure for large file uploads - 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Enable CORS for all origins (for Railway deployment)
CORS(app, resources={r"/*": {"origins": "*"}})

# Register the Routes with /api prefix
app.register_blueprint(data_bp, url_prefix='/api')
app.register_blueprint(clean_bp, url_prefix='/api')

@app.route('/')
def index():
    return {"status": "DataForge API is Live", "version": "1.0.0"}

@app.route('/health')
def health():
    return {"status": "healthy"}

@app.errorhandler(413)
def too_large(e):
    return {"error": "File too large. Max 500MB allowed."}, 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
