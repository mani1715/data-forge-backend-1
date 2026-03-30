from flask import Blueprint, request, jsonify, send_file, current_app
import pandas as pd
import io
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

# Import our logic from Stage 1
from services.profiler import DataProfiler

# Create a Blueprint
data_bp = Blueprint('data', __name__)

# We will store the dataframe in memory
CURRENT_DF = None
CURRENT_FILE_PATH = None

# Helper functions for other routes
def get_current_df():
    return CURRENT_DF

def set_current_df(df):
    global CURRENT_DF
    CURRENT_DF = df

@data_bp.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DF, CURRENT_FILE_PATH
    
    print(f"Upload request size: {request.content_length} bytes")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    print(f"File content_length: {file.content_length} bytes")
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Get upload folder from app config (Windows-safe)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1]
        unique_filename = f"{timestamp}_{unique_id}{file_ext}"
        file_path = os.path.join(upload_folder, unique_filename)
        
        # STEP 1: Save file to disk first (this prevents memory issues)
        print(f"Saving file to disk: {file_path}")
        file.save(file_path)
        saved_size = os.path.getsize(file_path)
        print(f"File saved successfully: {saved_size} bytes")
        
        # STEP 2: Read from disk with chunking for large files
        print("Reading file from disk...")
        if filename.endswith('.csv'):
            # Read CSV in chunks for large files
            file_size = saved_size
            if file_size > 50 * 1024 * 1024:  # If larger than 50MB
                print(f"Large file detected ({file_size} bytes), using chunked reading")
                chunks = []
                chunk_size = 50000  # Process 50k rows at a time
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                print(f"Processed {len(chunks)} chunks")
            else:
                df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            # Clean up file
            os.remove(file_path)
            return jsonify({'error': 'Unsupported file format'}), 400

        print(f"DataFrame loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # STEP 3: Profile the data BEFORE cleaning (to show original issues)
        print("Profiling original data...")
        summary = DataProfiler.get_summary(df)
        score = DataProfiler.calculate_quality_score(df)
        chart_data = DataProfiler.get_chart_data(df)
        
        # STEP 4: Apply custom formatting rules AFTER profiling
        print("Applying custom formatting rules...")
        from services.ai_engine import AIEngine
        df = AIEngine.apply_custom_rules(df)
        
        # STEP 5: Save cleaned data to memory
        CURRENT_DF = df
        CURRENT_FILE_PATH = file_path
        
        # STEP 6: Prepare Response with original stats but cleaned preview
        print("Preparing response...")
        response = {
            'message': 'File uploaded successfully',
            'filename': filename,
            'quality_score': score,  # Original quality score (before cleaning)
            'summary': summary,  # Original summary (shows issues found)
            'chart_data': chart_data,  # Original charts (shows missing values)
            'preview': df.head(20).to_dict(orient='records')  # Cleaned preview!
        }
        
        print("Upload complete!")
        return jsonify(response), 200
        
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The uploaded file is empty'}), 400
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        # Clean up file on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@data_bp.route('/download', methods=['GET'])
def download_file():
    global CURRENT_DF
    
    if CURRENT_DF is None:
        return jsonify({
            'error': 'No dataset available for download',
            'help': 'Please upload a file first, then optionally clean it, then download.'
        }), 400
    
    try:
        # Create a CSV in memory
        output = io.BytesIO()
        CURRENT_DF.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cleaned_data.csv'
        )
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@data_bp.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up uploaded files"""
    global CURRENT_FILE_PATH
    
    try:
        if CURRENT_FILE_PATH and os.path.exists(CURRENT_FILE_PATH):
            os.remove(CURRENT_FILE_PATH)
            CURRENT_FILE_PATH = None
            return jsonify({'message': 'Cleanup successful'}), 200
        return jsonify({'message': 'No file to clean up'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
