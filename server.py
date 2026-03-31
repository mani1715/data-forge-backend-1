"""
DataForge Backend - Minimal FastAPI Server for Railway
"""
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import os
import uuid
from datetime import datetime
import re

# Create FastAPI app
app = FastAPI()

# CORS - Must be first middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
CURRENT_DF = None
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_order_id(value):
    """Format order ID as 'ORD X' or return 0 if missing"""
    if pd.isna(value) or value is None:
        return 0
    
    str_val = str(value).strip()
    
    # Check for missing indicators
    if str_val.lower() in ['', 'nan', 'none', 'null', 'unknown', '??', '?', 'missing']:
        return 0
    
    # Already formatted as ORD X
    if str_val.upper().startswith('ORD'):
        # Extract number and reformat
        num_part = re.sub(r'[^0-9]', '', str_val)
        if num_part and int(num_part) > 0:
            return f"ORD {int(num_part)}"
        return 0
    
    # Try to get numeric value
    try:
        num = int(float(str_val))
        if num > 0:
            return f"ORD {num}"
        return 0
    except:
        return 0


def clean_numeric(value, allow_negative=False):
    """Convert to integer, return 0 if missing/invalid"""
    if pd.isna(value) or value is None:
        return 0
    
    str_val = str(value).strip()
    
    if str_val.lower() in ['', 'nan', 'none', 'null', 'unknown', '??', '?', 'missing']:
        return 0
    
    try:
        num = float(str_val)
        if not allow_negative and num < 0:
            return 0
        # Return as integer if whole number, else keep decimal
        if num == int(num):
            return int(num)
        return round(num, 2)
    except:
        return 0


def clean_date(value):
    """Validate date, return 00-00-0000 if invalid"""
    if pd.isna(value) or value is None:
        return "00-00-0000"
    
    str_val = str(value).strip()
    
    if str_val.lower() in ['', 'nan', 'none', 'null', 'unknown', '??', '?', 'missing']:
        return "00-00-0000"
    
    # Already placeholder
    if str_val in ['00-00-0000', '0000-00-00', '00/00/0000']:
        return "00-00-0000"
    
    # Try to parse and validate date
    try:
        # Replace / with - for consistency
        normalized = str_val.replace('/', '-')
        parts = normalized.split('-')
        
        if len(parts) == 3:
            # Determine format
            if len(parts[0]) == 4:  # YYYY-MM-DD
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            else:  # DD-MM-YYYY or MM-DD-YYYY (assume DD-MM-YYYY)
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Validate month and day
            if month < 1 or month > 12:
                return "00-00-0000"
            if day < 1 or day > 31:
                return "00-00-0000"
            
            # Return original format if valid
            return str_val
        
        return "00-00-0000"
    except:
        return "00-00-0000"


def clean_text(value):
    """Clean text values, replace ?? and missing with Unknown"""
    if pd.isna(value) or value is None:
        return "Unknown"
    
    str_val = str(value).strip()
    
    if str_val.lower() in ['', 'nan', 'none', 'null', '??', '?', 'missing']:
        return "Unknown"
    
    return str_val


def process_dataframe(df):
    """Process entire dataframe with proper cleaning rules"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        col_lower = col.lower()
        
        # Order ID columns
        if 'order' in col_lower or (col_lower == 'id' and 'order' in df_clean.columns[0].lower()):
            df_clean[col] = df_clean[col].apply(clean_order_id)
        
        # Date columns
        elif 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
            df_clean[col] = df_clean[col].apply(clean_date)
        
        # Price columns (allow decimals)
        elif 'price' in col_lower or 'cost' in col_lower or 'amount' in col_lower or 'total' in col_lower:
            df_clean[col] = df_clean[col].apply(lambda x: clean_numeric(x, allow_negative=False))
        
        # Quantity columns (integers only, no negatives)
        elif 'quantity' in col_lower or 'qty' in col_lower or 'count' in col_lower:
            df_clean[col] = df_clean[col].apply(lambda x: clean_numeric(x, allow_negative=False))
        
        # Other numeric columns
        elif df_clean[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            df_clean[col] = df_clean[col].apply(lambda x: clean_numeric(x, allow_negative=True))
        
        # Text columns
        elif df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(clean_text)
    
    return df_clean


@app.get("/")
def root():
    return {"status": "ok", "message": "DataForge API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/health")
def api_health():
    return {"status": "healthy"}


@app.options("/api/upload")
def upload_options():
    return JSONResponse(content={}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    })


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global CURRENT_DF
    
    try:
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.csv")
        contents = await file.read()
        
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Read DataFrame
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return JSONResponse(status_code=400, content={"detail": "Use CSV or XLSX"})
        
        # Clean up temp file
        os.remove(file_path)
        
        # Process and clean the dataframe
        df = process_dataframe(df)
        
        CURRENT_DF = df
        
        # Calculate stats
        total_cells = df.size
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        
        missing_pct = (missing / total_cells * 100) if total_cells > 0 else 0
        dup_pct = (duplicates / len(df) * 100) if len(df) > 0 else 0
        score = round(max(0, 100 - missing_pct - dup_pct), 2)
        
        # Missing data per column
        missing_data = []
        for col in df.columns:
            cnt = int(df[col].isnull().sum())
            if cnt > 0:
                missing_data.append({"name": str(col)[:12], "missing": cnt})
        
        # Type distribution
        num_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object']).columns)
        type_data = []
        if num_cols > 0:
            type_data.append({"name": "Numeric", "value": num_cols})
        if cat_cols > 0:
            type_data.append({"name": "Categorical", "value": cat_cols})
        
        # Build preview
        preview = build_preview(df)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "quality_score": score,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": missing,
                "duplicates": duplicates
            },
            "chart_data": {
                "missing_data": missing_data[:10],
                "type_data": type_data
            },
            "preview": preview
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


def build_preview(df):
    """Convert dataframe to JSON-safe preview"""
    preview = []
    for _, row in df.head(20).iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_dict[col] = None
            elif isinstance(val, (np.integer)):
                row_dict[col] = int(val)
            elif isinstance(val, (np.floating)):
                # Convert to int if whole number
                if val == int(val):
                    row_dict[col] = int(val)
                else:
                    row_dict[col] = round(float(val), 2)
            else:
                row_dict[col] = str(val)
        preview.append(row_dict)
    return preview


@app.options("/api/action")
def action_options():
    return JSONResponse(content={}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    })


@app.post("/api/action")
async def perform_action(request: Request):
    global CURRENT_DF
    
    if CURRENT_DF is None:
        return {"success": False, "error": "No data uploaded"}
    
    try:
        data = await request.json()
    except:
        data = {}
    
    action = data.get("action", "")
    strategy = data.get("strategy", "mean")
    df = CURRENT_DF.copy()
    message = ""
    
    try:
        if action == "remove_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            message = f"AI Analysis: Identified and removed {removed} duplicate rows. Data integrity preserved."
            
        elif action == "fill_missing":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            filled = 0
            
            for col in numeric_cols:
                missing = df[col].isnull().sum()
                if missing > 0:
                    if strategy == "mean":
                        fill_val = df[col].mean()
                        df[col] = df[col].fillna(fill_val)
                    elif strategy == "median":
                        fill_val = df[col].median()
                        df[col] = df[col].fillna(fill_val)
                    elif strategy == "mode":
                        mode = df[col].mode()
                        if len(mode) > 0:
                            df[col] = df[col].fillna(mode[0])
                    else:  # constant or ai
                        df[col] = df[col].fillna(0)
                    filled += missing
            
            # Convert to integers where appropriate
            for col in numeric_cols:
                if df[col].apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                    df[col] = df[col].astype(int)
            
            message = f"AI Analysis: Filled {filled} missing values using {strategy} strategy."
            
        elif action == "remove_outliers":
            before = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
            
            removed = before - len(df)
            message = f"AI Analysis: Detected and removed {removed} statistical outliers using IQR method."
            
        elif action == "clean_text":
            cleaned = 0
            for col in df.select_dtypes(include=['object']).columns:
                # Replace ??, ?, and other invalid values
                mask = df[col].isin(['??', '?', '', 'nan', 'NaN', 'None', 'null'])
                cleaned += mask.sum()
                df[col] = df[col].replace(['??', '?', '', 'nan', 'NaN', 'None', 'null'], 'Unknown')
                df[col] = df[col].fillna('Unknown')
            message = f"AI Analysis: Cleaned {cleaned} invalid text entries and standardized formatting."
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        # Re-process to ensure clean formatting
        df = process_dataframe(df)
        
        # Update and recalculate
        CURRENT_DF = df
        
        total_cells = df.size
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        missing_pct = (missing / total_cells * 100) if total_cells > 0 else 0
        dup_pct = (duplicates / len(df) * 100) if len(df) > 0 else 0
        score = round(max(0, 100 - missing_pct - dup_pct), 2)
        
        preview = build_preview(df)
        
        return {
            "success": True,
            "message": message,
            "new_score": score,
            "preview": preview
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/download")
def download_file():
    if CURRENT_DF is None:
        return JSONResponse(status_code=400, content={"detail": "No data"})
    
    output = io.BytesIO()
    CURRENT_DF.to_csv(output, index=False)
    output.seek(0)
    
    headers = {
        "Content-Disposition": "attachment; filename=cleaned_data.csv",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Expose-Headers": "Content-Disposition"
    }
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers=headers
    )


# For Railway - run with: uvicorn server:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
