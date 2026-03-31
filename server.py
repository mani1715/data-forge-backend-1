"""
DataForge Backend - FastAPI Server
Wraps Flask app for ASGI compatibility with uvicorn
"""
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our services
from services.profiler import DataProfiler
from services.ai_engine import AIEngine

# Create FastAPI app
app = FastAPI(title="DataForge API", version="1.0.0")

# Configure CORS - Allow all origins for Railway deployment
origins = [
    "https://data-forge-frontend-production-06e0.up.railway.app",
    "https://web-production-169b9.up.railway.app",
    "http://localhost:3000",
    "http://localhost:8001",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# In-memory storage
CURRENT_DF = None
CURRENT_FILE_PATH = None
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_current_df():
    return CURRENT_DF

def set_current_df(df):
    global CURRENT_DF
    CURRENT_DF = df


@app.get("/")
async def index():
    return {"status": "DataForge API is Live", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "healthy", "api": True}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global CURRENT_DF, CURRENT_FILE_PATH
    
    print("UPLOAD API HIT")
    print(f"File: {file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{timestamp}_{unique_id}{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save file to disk
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        saved_size = os.path.getsize(file_path)
        print(f"File saved: {saved_size} bytes")
        
        # Read file into DataFrame
        if file.filename.lower().endswith('.csv'):
            if saved_size > 50 * 1024 * 1024:  # > 50MB
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=50000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        print(f"DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Profile the data BEFORE cleaning
        summary = DataProfiler.get_summary(df)
        score = DataProfiler.calculate_quality_score(df)
        chart_data = DataProfiler.get_chart_data(df)
        
        # Apply custom formatting rules
        df = AIEngine.apply_custom_rules(df)
        
        # Save to memory
        CURRENT_DF = df
        CURRENT_FILE_PATH = file_path
        
        return {
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'quality_score': score,
            'summary': summary,
            'chart_data': chart_data,
            'preview': df.head(20).replace({np.nan: None}).to_dict(orient='records')
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/download")
async def download_file():
    print("DOWNLOAD API HIT")
    
    if CURRENT_DF is None:
        raise HTTPException(status_code=400, detail="No dataset available. Upload a file first.")
    
    try:
        output = io.BytesIO()
        CURRENT_DF.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type='text/csv',
            headers={'Content-Disposition': 'attachment; filename=cleaned_data.csv'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.post("/api/action")
async def perform_action(request: Request):
    global CURRENT_DF
    
    print("ACTION API HIT")
    
    if CURRENT_DF is None:
        return JSONResponse({
            'error': 'No data uploaded',
            'message': 'Please upload a file first',
            'success': False
        })
    
    try:
        data = await request.json()
    except:
        data = {}
    
    action = data.get('action')
    if not action:
        return JSONResponse({
            'error': 'Missing action',
            'message': 'Please specify an action',
            'valid_actions': ['remove_duplicates', 'fill_missing', 'remove_outliers', 'clean_text'],
            'success': False
        })
    
    strategy = data.get('strategy', 'ai')
    fill_value = data.get('fill_value')
    
    df = CURRENT_DF.copy()
    message = "No action performed"
    
    try:
        initial_rows = len(df)
        initial_missing = df.isnull().sum().sum()
        
        if action == 'remove_duplicates':
            df, message = AIEngine.remove_duplicates(df)
        elif action == 'fill_missing':
            df, message = AIEngine.clean_missing_values(df, strategy=strategy, fill_value=fill_value)
        elif action == 'remove_outliers':
            df, message = AIEngine.remove_outliers(df)
        elif action == 'clean_text':
            df, message = AIEngine.clean_categorical_data(df, strategy=strategy)
        else:
            return JSONResponse({
                'error': 'Invalid action',
                'message': f"Unknown action: '{action}'",
                'valid_actions': ['remove_duplicates', 'fill_missing', 'remove_outliers', 'clean_text'],
                'success': False
            })
        
        # Update stored DataFrame
        CURRENT_DF = df
        
        # Calculate new stats
        new_score = DataProfiler.calculate_quality_score(df)
        new_summary = DataProfiler.get_summary(df)
        new_chart_data = DataProfiler.get_chart_data(df)
        
        return JSONResponse({
            'success': True,
            'message': message,
            'new_score': new_score,
            'summary': new_summary,
            'chart_data': new_chart_data,
            'preview': df.head(20).replace({np.nan: None}).to_dict(orient='records'),
            'changes': {
                'rows_removed': initial_rows - len(df),
                'missing_filled': int(initial_missing - df.isnull().sum().sum())
            }
        })
        
    except Exception as e:
        print(f"Action error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            'error': str(e),
            'message': f"Failed to perform '{action}'",
            'success': False
        })


@app.get("/api/stats")
async def get_stats():
    print("STATS API HIT")
    
    if CURRENT_DF is None:
        return JSONResponse({
            'error': 'No data uploaded',
            'success': False
        })
    
    try:
        return JSONResponse({
            'success': True,
            'score': DataProfiler.calculate_quality_score(CURRENT_DF),
            'summary': DataProfiler.get_summary(CURRENT_DF),
            'chart_data': DataProfiler.get_chart_data(CURRENT_DF)
        })
    except Exception as e:
        return JSONResponse({
            'error': str(e),
            'success': False
        })


@app.post("/api/cleanup")
async def cleanup():
    global CURRENT_FILE_PATH
    
    print("CLEANUP API HIT")
    
    try:
        if CURRENT_FILE_PATH and os.path.exists(CURRENT_FILE_PATH):
            os.remove(CURRENT_FILE_PATH)
            CURRENT_FILE_PATH = None
            return {'message': 'Cleanup successful'}
        return {'message': 'No file to clean up'}
    except Exception as e:
        return {'error': str(e)}
