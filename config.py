import os

class Config:
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dataforge-secret-key-2024'
    
    # Max file upload size (500MB)
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024
    
    # Allowed extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}
