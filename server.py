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
ORIGINAL_DF = None  # Keep original to track what was missing
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def is_missing_value(value):
    """Check if a value should be considered as missing"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', 'missing', 'na', 'n/a']:
        return True
    return False


def is_invalid_date(value):
    """Check if date is invalid"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', '00-00-0000', '0000-00-00']:
        return True
    # Check for invalid month/day
    try:
        normalized = str_val.replace('/', '-')
        parts = normalized.split('-')
        if len(parts) == 3:
            if len(parts[0]) == 4:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            if month < 1 or month > 12 or day < 1 or day > 31:
                return True
    except:
        return True
    return False


def format_order_id(value):
    """Format order ID as 'ORD X' or return 0 if missing"""
    if is_missing_value(value):
        return 0
    str_val = str(value).strip()
    if str_val.upper().startswith('ORD'):
        num_part = re.sub(r'[^0-9]', '', str_val)
        if num_part and int(num_part) > 0:
            return f"ORD {int(num_part)}"
        return 0
    try:
        num = int(float(str_val))
        if num > 0:
            return f"ORD {num}"
        return 0
    except:
        return 0


def format_for_display(df):
    """Format dataframe for display - convert missing to placeholders"""
    df_display = df.copy()
    
    for col in df_display.columns:
        col_lower = col.lower()
        
        # Order ID columns
        if 'order' in col_lower:
            df_display[col] = df_display[col].apply(format_order_id)
        
        # Date columns
        elif 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
            df_display[col] = df_display[col].apply(
                lambda x: "00-00-0000" if is_invalid_date(x) else str(x)
            )
        
        # Numeric columns - keep NaN as NaN for now, format later
        elif df_display[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            pass
        
        # Text columns
        elif df_display[col].dtype == 'object':
            df_display[col] = df_display[col].apply(
                lambda x: "Unknown" if is_missing_value(x) else str(x)
            )
    
    return df_display


def build_preview(df):
    """Convert dataframe to JSON-safe preview with proper formatting"""
    preview = []
    for _, row in df.head(20).iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            col_lower = col.lower()
            
            if pd.isna(val):
                # Show placeholder based on column type
                if 'order' in col_lower:
                    row_dict[col] = 0
                elif 'date' in col_lower or 'dob' in col_lower:
                    row_dict[col] = "00-00-0000"
                elif 'price' in col_lower or 'quantity' in col_lower or 'amount' in col_lower:
                    row_dict[col] = 0
                else:
                    row_dict[col] = "Unknown"
            elif isinstance(val, (np.integer)):
                row_dict[col] = int(val)
            elif isinstance(val, (np.floating)):
                if val == int(val):
                    row_dict[col] = int(val)
                else:
                    row_dict[col] = round(float(val), 2)
            else:
                row_dict[col] = str(val)
        preview.append(row_dict)
    return preview


def calculate_quality_score(df):
    """Calculate quality score - 100% means no missing values and no duplicates"""
    if df.empty:
        return 0.0
    
    total_cells = df.size
    total_rows = len(df)
    
    # Count actual missing values (NaN)
    missing_cells = df.isnull().sum().sum()
    
    # Also count placeholder values as missing
    for col in df.columns:
        col_lower = col.lower()
        for val in df[col]:
            if pd.notna(val):
                str_val = str(val).lower().strip()
                if str_val in ['unknown', '0', '0.0', '00-00-0000', '??', '?']:
                    missing_cells += 1
    
    # Calculate penalties
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    duplicates = df.duplicated().sum()
    dup_pct = (duplicates / total_rows * 100) if total_rows > 0 else 0
    
    score = 100 - missing_pct - dup_pct
    return round(max(0, min(100, score)), 2)


@app.get("/")
def root():
    return {"status": "ok", "message": "DataForge API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/health")
def api_health():
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global CURRENT_DF, ORIGINAL_DF
    
    try:
        file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.csv")
        contents = await file.read()
        
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return JSONResponse(status_code=400, content={"detail": "Use CSV or XLSX"})
        
        os.remove(file_path)
        
        # Mark missing values as NaN for proper tracking
        for col in df.columns:
            col_lower = col.lower()
            
            # Convert text missing indicators to NaN
            if df[col].dtype == 'object':
                df[col] = df[col].replace(['', 'nan', 'NaN', 'None', 'null', '??', '?', 'missing', 'NA', 'N/A'], np.nan)
            
            # For numeric columns, convert to numeric (invalid become NaN)
            if 'price' in col_lower or 'quantity' in col_lower or 'amount' in col_lower or 'age' in col_lower:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Negative values to NaN for quantity
                if 'quantity' in col_lower:
                    df.loc[df[col] < 0, col] = np.nan
        
        ORIGINAL_DF = df.copy()
        CURRENT_DF = df.copy()
        
        # Calculate stats
        score = calculate_quality_score(df)
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        
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
        
        preview = build_preview(format_for_display(df))
        
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
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})


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
            filled = 0
            
            # Fill numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                missing_mask = df[col].isnull()
                missing_count = missing_mask.sum()
                
                if missing_count > 0:
                    if strategy == "mean" or strategy == "ai":
                        fill_val = df[col].mean()
                    elif strategy == "median":
                        fill_val = df[col].median()
                    elif strategy == "mode":
                        mode = df[col].mode()
                        fill_val = mode[0] if len(mode) > 0 else df[col].mean()
                    else:  # constant
                        fill_val = 0
                    
                    # Round to reasonable precision
                    if pd.notna(fill_val):
                        if fill_val == int(fill_val):
                            fill_val = int(fill_val)
                        else:
                            fill_val = round(fill_val, 2)
                        
                        df.loc[missing_mask, col] = fill_val
                        filled += missing_count
            
            # Fill text columns with mode
            text_cols = df.select_dtypes(include=['object']).columns
            for col in text_cols:
                missing_mask = df[col].isnull()
                missing_count = missing_mask.sum()
                
                if missing_count > 0:
                    if strategy == "mode" or strategy == "ai":
                        mode = df[col].mode()
                        fill_val = mode[0] if len(mode) > 0 else "Unknown"
                    else:
                        fill_val = "Unknown"
                    
                    df.loc[missing_mask, col] = fill_val
                    filled += missing_count
            
            message = f"AI Analysis: Filled {filled} missing values using {strategy} strategy. Data quality improved."
            
        elif action == "remove_outliers":
            before = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].notna().sum() > 4:  # Need enough data for IQR
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isnull()]
            
            removed = before - len(df)
            message = f"AI Analysis: Detected and removed {removed} statistical outliers using IQR method."
            
        elif action == "clean_text":
            cleaned = 0
            for col in df.select_dtypes(include=['object']).columns:
                # Replace invalid values with NaN first
                invalid_mask = df[col].isin(['??', '?', '', 'nan', 'NaN', 'None', 'null'])
                cleaned += invalid_mask.sum()
                df.loc[invalid_mask, col] = np.nan
                
                # Then fill with mode
                missing_mask = df[col].isnull()
                if missing_mask.sum() > 0:
                    mode = df[col].mode()
                    fill_val = mode[0] if len(mode) > 0 else "Unknown"
                    df.loc[missing_mask, col] = fill_val
            
            message = f"AI Analysis: Cleaned {cleaned} invalid text entries and filled with most common values."
            
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        CURRENT_DF = df
        
        # Recalculate stats
        score = calculate_quality_score(df)
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        
        preview = build_preview(format_for_display(df))
        
        return {
            "success": True,
            "message": message,
            "new_score": score,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": missing,
                "duplicates": duplicates
            },
            "preview": preview
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/download")
def download_file():
    if CURRENT_DF is None:
        return JSONResponse(status_code=400, content={"detail": "No data"})
    
    # Format for download
    df_download = format_for_display(CURRENT_DF)
    
    output = io.BytesIO()
    df_download.to_csv(output, index=False)
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
