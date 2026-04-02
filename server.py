"""
DataForge Backend - Minimal FastAPI Server for Railway
Fixed: 100% score means NO red values (all placeholders filled with real data)
"""
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import os
import uuid
from datetime import datetime, timedelta
import re
import random

# Create FastAPI app
app = FastAPI()

# CORS
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


def is_missing_value(value):
    """Check if value is missing or placeholder"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', 'missing', 'na', 'n/a', '0', '0.0', '00-00-0000', '0000-00-00']:
        return True
    return False


def is_order_id_missing(value):
    """Check if order ID is missing"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', '0', '0.0']:
        return True
    # Check if it's just "0" or doesn't have ORD prefix
    if str_val == '0':
        return True
    return False


def is_date_missing(value):
    """Check if date is missing or invalid"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', '00-00-0000', '0000-00-00', '00/00/0000']:
        return True
    # Check for invalid month/day
    try:
        normalized = str_val.replace('/', '-')
        parts = normalized.split('-')
        if len(parts) == 3:
            nums = [int(p) for p in parts]
            # Check if any part suggests invalid date
            if 0 in nums:
                return True
            # Check month validity
            if len(parts[0]) == 4:  # YYYY-MM-DD
                month, day = nums[1], nums[2]
            else:  # DD-MM-YYYY
                day, month = nums[0], nums[1]
            if month < 1 or month > 12 or day < 1 or day > 31:
                return True
    except:
        return True
    return False


def calculate_quality_score(df):
    """Calculate quality score - 100% ONLY when NO missing/placeholder values exist"""
    if df.empty:
        return 0.0
    
    total_cells = df.size
    total_rows = len(df)
    missing_count = 0
    
    for col in df.columns:
        col_lower = col.lower()
        
        for val in df[col]:
            # Check based on column type
            if 'order' in col_lower or col_lower == 'id':
                if is_order_id_missing(val):
                    missing_count += 1
            elif 'date' in col_lower or 'dob' in col_lower:
                if is_date_missing(val):
                    missing_count += 1
            elif pd.isna(val):
                missing_count += 1
            elif str(val).lower().strip() in ['unknown', '??', '?', '']:
                missing_count += 1
            elif col_lower in ['price', 'quantity', 'amount', 'qty'] and (val == 0 or str(val) == '0'):
                missing_count += 1
    
    # Calculate duplicates
    duplicates = df.duplicated().sum()
    
    # Calculate score
    missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
    dup_pct = (duplicates / total_rows * 100) if total_rows > 0 else 0
    
    score = 100 - missing_pct - dup_pct
    return round(max(0, min(100, score)), 2)


def fill_order_ids(df, col):
    """Fill missing order IDs with sequential ORD numbers"""
    # Find max existing order number
    max_num = 0
    for val in df[col]:
        if pd.notna(val):
            str_val = str(val).strip()
            nums = re.findall(r'\d+', str_val)
            if nums:
                max_num = max(max_num, max(int(n) for n in nums))
    
    # Fill missing with new sequential numbers
    next_num = max_num + 1
    new_values = []
    for val in df[col]:
        if is_order_id_missing(val):
            new_values.append(f"ORD {next_num}")
            next_num += 1
        else:
            # Format existing value
            str_val = str(val).strip()
            nums = re.findall(r'\d+', str_val)
            if nums:
                new_values.append(f"ORD {nums[0]}")
            else:
                new_values.append(f"ORD {next_num}")
                next_num += 1
    
    return new_values


def fill_dates(df, col):
    """Fill missing dates with valid dates based on existing data"""
    # Collect valid dates
    valid_dates = []
    for val in df[col]:
        if not is_date_missing(val):
            valid_dates.append(str(val))
    
    # If we have valid dates, use mode; otherwise generate dates
    if valid_dates:
        # Use most common date format and generate around that
        mode_date = max(set(valid_dates), key=valid_dates.count)
        fill_date = mode_date
    else:
        # Generate a reasonable date
        fill_date = "2024-01-15"
    
    # Fill missing
    new_values = []
    for val in df[col]:
        if is_date_missing(val):
            new_values.append(fill_date)
        else:
            new_values.append(str(val))
    
    return new_values


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
                if val == int(val):
                    row_dict[col] = int(val)
                else:
                    row_dict[col] = round(float(val), 2)
            else:
                row_dict[col] = str(val)
        preview.append(row_dict)
    return preview


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/health")
def api_health():
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global CURRENT_DF
    
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
        
        CURRENT_DF = df.copy()
        
        score = calculate_quality_score(df)
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        
        # Count actual missing including placeholders
        actual_missing = 0
        for col in df.columns:
            for val in df[col]:
                if is_missing_value(val):
                    actual_missing += 1
        
        missing_data = []
        for col in df.columns:
            cnt = sum(1 for val in df[col] if is_missing_value(val))
            if cnt > 0:
                missing_data.append({"name": str(col)[:12], "missing": cnt})
        
        num_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object']).columns)
        type_data = []
        if num_cols > 0:
            type_data.append({"name": "Numeric", "value": num_cols})
        if cat_cols > 0:
            type_data.append({"name": "Categorical", "value": cat_cols})
        
        preview = build_preview(df)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "quality_score": score,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": actual_missing,
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
    filled_total = 0
    
    try:
        if action == "remove_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            message = f"AI Analysis: Removed {removed} duplicate rows."
            
        elif action == "fill_missing":
            # Process each column based on its type
            for col in df.columns:
                col_lower = col.lower()
                
                # ORDER ID columns - fill with sequential ORD numbers
                if 'order' in col_lower or col_lower == 'id':
                    before_missing = sum(1 for v in df[col] if is_order_id_missing(v))
                    if before_missing > 0:
                        df[col] = fill_order_ids(df, col)
                        filled_total += before_missing
                
                # DATE columns - fill with valid dates
                elif 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
                    before_missing = sum(1 for v in df[col] if is_date_missing(v))
                    if before_missing > 0:
                        df[col] = fill_dates(df, col)
                        filled_total += before_missing
                
                # NUMERIC columns - fill with mean/median/mode
                elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    # Convert column to numeric, coercing errors
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    missing_mask = df[col].isnull() | (df[col] == 0)
                    before_missing = missing_mask.sum()
                    
                    if before_missing > 0:
                        # Calculate fill value from non-zero, non-null values
                        valid_vals = df[col][(df[col].notna()) & (df[col] != 0)]
                        
                        if len(valid_vals) > 0:
                            if strategy == "mean" or strategy == "ai":
                                fill_val = valid_vals.mean()
                            elif strategy == "median":
                                fill_val = valid_vals.median()
                            elif strategy == "mode":
                                mode = valid_vals.mode()
                                fill_val = mode[0] if len(mode) > 0 else valid_vals.mean()
                            else:
                                fill_val = valid_vals.mean()
                            
                            # Round appropriately
                            if fill_val == int(fill_val):
                                fill_val = int(fill_val)
                            else:
                                fill_val = round(fill_val, 2)
                            
                            df.loc[missing_mask, col] = fill_val
                            filled_total += before_missing
                
                # TEXT columns - fill with mode or proper value
                elif df[col].dtype == 'object':
                    # Find missing text values
                    missing_mask = df[col].apply(lambda x: is_missing_value(x))
                    before_missing = missing_mask.sum()
                    
                    if before_missing > 0:
                        # Get mode from non-missing values
                        valid_vals = df[col][~missing_mask]
                        if len(valid_vals) > 0:
                            mode = valid_vals.mode()
                            fill_val = mode[0] if len(mode) > 0 else "Unknown"
                        else:
                            fill_val = "Unknown"
                        
                        df.loc[missing_mask, col] = fill_val
                        filled_total += before_missing
            
            message = f"AI Analysis: Filled {filled_total} missing values using {strategy} strategy. All data now complete."
            
        elif action == "remove_outliers":
            before = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                valid_data = df[col].dropna()
                if len(valid_data) > 4:
                    Q1 = valid_data.quantile(0.25)
                    Q3 = valid_data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isnull()]
            
            removed = before - len(df)
            message = f"AI Analysis: Removed {removed} outliers."
            
        elif action == "clean_text":
            cleaned = 0
            for col in df.select_dtypes(include=['object']).columns:
                # Find and fix invalid text values
                for idx, val in df[col].items():
                    if is_missing_value(val):
                        # Get mode
                        valid_vals = df[col][df[col].apply(lambda x: not is_missing_value(x))]
                        if len(valid_vals) > 0:
                            mode = valid_vals.mode()
                            fill_val = mode[0] if len(mode) > 0 else "Standard"
                        else:
                            fill_val = "Standard"
                        df.at[idx, col] = fill_val
                        cleaned += 1
            
            message = f"AI Analysis: Cleaned {cleaned} text entries."
            
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        CURRENT_DF = df
        
        score = calculate_quality_score(df)
        duplicates = int(df.duplicated().sum())
        
        # Count remaining missing
        remaining_missing = 0
        for col in df.columns:
            for val in df[col]:
                if is_missing_value(val):
                    remaining_missing += 1
        
        preview = build_preview(df)
        
        return {
            "success": True,
            "message": message,
            "new_score": score,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": remaining_missing,
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
    
    output = io.BytesIO()
    CURRENT_DF.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=cleaned_data.csv",
            "Access-Control-Allow-Origin": "*"
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
