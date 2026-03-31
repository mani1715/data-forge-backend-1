"""
DataForge Backend - FastAPI Server
Production-ready with proper CORS and error handling
"""
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd
import numpy as np
import io
import os
import uuid
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Create FastAPI app
app = FastAPI(title="DataForge API", version="1.0.0")

# Custom CORS middleware to ensure headers are always sent
class CORSMiddlewareCustom(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Handle preflight OPTIONS request
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "3600"
            return response
        
        # Process actual request
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )
        
        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

# Add custom CORS middleware
app.add_middleware(CORSMiddlewareCustom)

# Also add standard CORS middleware as backup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
CURRENT_DF = None
CURRENT_FILE_PATH = None
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================
# DATA PROFILER - Inline to avoid import issues
# ============================================

class DataProfiler:
    @staticmethod
    def calculate_quality_score(df):
        if df.empty:
            return 0.0
        total_cells = df.size
        total_rows = len(df)
        missing_cells = df.isnull().sum().sum()
        missing_penalty = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        duplicate_rows = df.duplicated().sum()
        duplicate_penalty = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        missing_penalty = min(missing_penalty, 50)
        duplicate_penalty = min(duplicate_penalty, 30)
        score = 100 - (missing_penalty + duplicate_penalty)
        return round(max(0, min(100, score)), 2)

    @staticmethod
    def get_summary(df):
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'column_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        }

    @staticmethod
    def get_chart_data(df):
        missing_counts = df.isnull().sum().sort_values(ascending=False).head(10)
        missing_data = []
        for col, count in missing_counts.items():
            if count > 0:
                missing_data.append({
                    'name': str(col)[:15] if len(str(col)) > 15 else str(col),
                    'missing': int(count)
                })
        
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        
        type_data = []
        if numeric_count > 0:
            type_data.append({'name': 'Numeric', 'value': numeric_count})
        if categorical_count > 0:
            type_data.append({'name': 'Categorical', 'value': categorical_count})
        
        return {'missing_data': missing_data, 'type_data': type_data}


# ============================================
# AI ENGINE - Inline with safe imports
# ============================================

class AIEngine:
    @staticmethod
    def apply_custom_rules(df):
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna("Unknown")
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'None', ''], "Unknown")
        return df_clean

    @staticmethod
    def clean_missing_values(df, strategy='mean', fill_value=None):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        message = ""
        filled_count = 0

        if strategy == 'drop_rows':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            message = f"Dropped {initial_rows - len(df_clean)} rows with missing values."
            return df_clean, message

        if len(numeric_cols) > 0:
            cols_with_missing = [col for col in numeric_cols if df_clean[col].isnull().any()]
            
            if cols_with_missing:
                if strategy == 'ai' or strategy == 'mean':
                    for col in cols_with_missing:
                        col_mean = df_clean[col].mean()
                        missing_count = df_clean[col].isnull().sum()
                        df_clean[col] = df_clean[col].fillna(col_mean)
                        filled_count += missing_count
                    message = f"Filled {filled_count} missing values with column means."
                    
                elif strategy == 'median':
                    for col in cols_with_missing:
                        col_median = df_clean[col].median()
                        missing_count = df_clean[col].isnull().sum()
                        df_clean[col] = df_clean[col].fillna(col_median)
                        filled_count += missing_count
                    message = f"Filled {filled_count} missing values with column medians."
                
                elif strategy == 'mode':
                    for col in cols_with_missing:
                        mode_val = df_clean[col].mode()
                        missing_count = df_clean[col].isnull().sum()
                        if not mode_val.empty:
                            df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                            filled_count += missing_count
                    message = f"Filled {filled_count} missing values with column modes."

                elif strategy == 'constant':
                    val = fill_value if fill_value is not None else 0
                    for col in cols_with_missing:
                        missing_count = df_clean[col].isnull().sum()
                        df_clean[col] = df_clean[col].fillna(val)
                        filled_count += missing_count
                    message = f"Filled {filled_count} missing values with {val}."
            else:
                message = "No missing numeric values found."
        else:
            message = "No numeric columns found."
        
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, message

    @staticmethod
    def remove_outliers(df):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        initial_rows = len(df_clean)
        
        if len(df_clean) > 10:
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        rows_removed = initial_rows - len(df_clean)
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"Removed {rows_removed} outliers using IQR method."

    @staticmethod
    def clean_categorical_data(df, strategy='unknown'):
        df_clean = df.copy()
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        
        if len(cat_cols) == 0:
            return df_clean, "No text columns found."

        filled_count = 0
        for col in cat_cols:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                df_clean[col] = df_clean[col].fillna('Unknown')
                filled_count += missing_before
        
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"Cleaned {filled_count} missing values in text columns."
    
    @staticmethod
    def remove_duplicates(df):
        df_clean = df.copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        rows_removed = initial_rows - len(df_clean)
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"Removed {rows_removed} duplicate rows."


# ============================================
# API ENDPOINTS
# ============================================

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
    
    print(f"UPLOAD API HIT - File: {file.filename}")
    
    if not file.filename:
        return JSONResponse(status_code=400, content={"detail": "No file selected"})
    
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
            df = pd.read_csv(file_path)
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            os.remove(file_path)
            return JSONResponse(status_code=400, content={"detail": "Unsupported format. Use CSV or XLSX."})
        
        print(f"DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Profile and clean
        summary = DataProfiler.get_summary(df)
        score = DataProfiler.calculate_quality_score(df)
        chart_data = DataProfiler.get_chart_data(df)
        df = AIEngine.apply_custom_rules(df)
        
        # Save to memory
        CURRENT_DF = df
        CURRENT_FILE_PATH = file_path
        
        # Convert preview to JSON-safe format
        preview = df.head(20).fillna("").to_dict(orient='records')
        
        return {
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'quality_score': score,
            'summary': summary,
            'chart_data': chart_data,
            'preview': preview
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(status_code=500, content={"detail": f"Error: {str(e)}"})


@app.get("/api/download")
async def download_file():
    if CURRENT_DF is None:
        return JSONResponse(status_code=400, content={"detail": "No dataset. Upload first."})
    
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
        return JSONResponse(status_code=500, content={"detail": f"Download failed: {str(e)}"})


@app.post("/api/action")
async def perform_action(request: Request):
    global CURRENT_DF
    
    if CURRENT_DF is None:
        return JSONResponse(content={'error': 'No data uploaded', 'success': False})
    
    try:
        data = await request.json()
    except:
        data = {}
    
    action = data.get('action')
    if not action:
        return JSONResponse(content={
            'error': 'Missing action',
            'valid_actions': ['remove_duplicates', 'fill_missing', 'remove_outliers', 'clean_text'],
            'success': False
        })
    
    strategy = data.get('strategy', 'mean')
    fill_value = data.get('fill_value')
    
    df = CURRENT_DF.copy()
    
    try:
        if action == 'remove_duplicates':
            df, message = AIEngine.remove_duplicates(df)
        elif action == 'fill_missing':
            df, message = AIEngine.clean_missing_values(df, strategy=strategy, fill_value=fill_value)
        elif action == 'remove_outliers':
            df, message = AIEngine.remove_outliers(df)
        elif action == 'clean_text':
            df, message = AIEngine.clean_categorical_data(df, strategy=strategy)
        else:
            return JSONResponse(content={'error': f"Unknown action: '{action}'", 'success': False})
        
        CURRENT_DF = df
        
        return JSONResponse(content={
            'success': True,
            'message': message,
            'new_score': DataProfiler.calculate_quality_score(df),
            'summary': DataProfiler.get_summary(df),
            'chart_data': DataProfiler.get_chart_data(df),
            'preview': df.head(20).fillna("").to_dict(orient='records')
        })
        
    except Exception as e:
        print(f"Action error: {str(e)}")
        return JSONResponse(content={'error': str(e), 'success': False})


@app.get("/api/stats")
async def get_stats():
    if CURRENT_DF is None:
        return JSONResponse(content={'error': 'No data uploaded', 'success': False})
    
    return JSONResponse(content={
        'success': True,
        'score': DataProfiler.calculate_quality_score(CURRENT_DF),
        'summary': DataProfiler.get_summary(CURRENT_DF),
        'chart_data': DataProfiler.get_chart_data(CURRENT_DF)
    })


@app.post("/api/cleanup")
async def cleanup():
    global CURRENT_FILE_PATH
    
    try:
        if CURRENT_FILE_PATH and os.path.exists(CURRENT_FILE_PATH):
            os.remove(CURRENT_FILE_PATH)
            CURRENT_FILE_PATH = None
            return {'message': 'Cleanup successful'}
        return {'message': 'No file to clean up'}
    except Exception as e:
        return {'error': str(e)}
