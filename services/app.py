from flask import Flask
from config import Config

# Initialize App
app = Flask(__name__)
app.config.from_object(Config)

# Test Route
@app.route('/')
def index():
    return "DataForge Backend is Running."

if __name__ == '__main__':
    app.run(debug=True, port=5000)