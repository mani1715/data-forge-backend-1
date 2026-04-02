"""
DataForge Backend - Minimal FastAPI Server for Railway
Fixed: Quantity shows numbers, all buttons work correctly
"""
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import os
import uuid
import re

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


def is_missing_text(value):
    """Check if text value is missing"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    return str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', 'missing', 'na', 'n/a']


def is_missing_number(value):
    """Check if numeric value is missing (NaN or 0 placeholder)"""
    if pd.isna(value) or value is None:
        return True
    try:
        num = float(value)
        return num == 0 or num < 0
    except:
        return True


def is_missing_date(value):
    """Check if date is missing or invalid"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', '00-00-0000', '0000-00-00']:
        return True
    try:
        parts = str_val.replace('/', '-').split('-')
        if len(parts) == 3:
            nums = [int(p) for p in parts]
            if 0 in nums:
                return True
            if len(parts[0]) == 4:
                month, day = nums[1], nums[2]
            else:
                day, month = nums[0], nums[1]
            if month < 1 or month > 12 or day < 1 or day > 31:
                return True
    except:
        return True
    return False


def is_missing_order(value):
    """Check if order ID is missing"""
    if pd.isna(value) or value is None:
        return True
    str_val = str(value).strip().lower()
    if str_val in ['', 'nan', 'none', 'null', 'unknown', '??', '?', '0', '0.0']:
        return True
    return False


def calculate_quality_score(df):
    """Calculate quality score - 100% means NO missing values"""
    if df.empty:
        return 0.0
    
    total_cells = df.size
    total_rows = len(df)
    missing_count = 0
    
    for col in df.columns:
        col_lower = col.lower()
        
        for val in df[col]:
            if 'order' in col_lower:
                if is_missing_order(val):
                    missing_count += 1
            elif 'date' in col_lower or 'dob' in col_lower:
                if is_missing_date(val):
                    missing_count += 1
            elif col_lower in ['price', 'quantity', 'qty', 'amount', 'cost', 'total', 'age', 'salary']:
                if is_missing_number(val):
                    missing_count += 1
            elif df[col].dtype == 'object':
                if is_missing_text(val):
                    missing_count += 1
            elif pd.isna(val):
                missing_count += 1
    
    duplicates = df.duplicated().sum()
    missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
    dup_pct = (duplicates / total_rows * 100) if total_rows > 0 else 0
    
    score = 100 - missing_pct - dup_pct
    return round(max(0, min(100, score)), 2)


def fill_order_ids(series):
    """Fill missing order IDs with sequential ORD numbers"""
    max_num = 0
    for val in series:
        if not is_missing_order(val):
            nums = re.findall(r'\d+', str(val))
            if nums:
                max_num = max(max_num, max(int(n) for n in nums))
    
    result = []
    next_num = max_num + 1
    for val in series:
        if is_missing_order(val):
            result.append(f"ORD {next_num}")
            next_num += 1
        else:
            nums = re.findall(r'\d+', str(val))
            if nums:
                result.append(f"ORD {nums[0]}")
            else:
                result.append(f"ORD {next_num}")
                next_num += 1
    return result


def fill_dates(series):
    """Fill missing dates with valid date"""
    valid_dates = [str(v) for v in series if not is_missing_date(v)]
    
    if valid_dates:
        fill_date = max(set(valid_dates), key=valid_dates.count)
    else:
        fill_date = "2024-01-15"
    
    return [fill_date if is_missing_date(v) else str(v) for v in series]


def fill_numeric(series, strategy="mean"):
    """Fill missing numeric values with calculated value"""
    # Convert to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Get valid values (not NaN and > 0)
    valid_vals = numeric_series[(numeric_series.notna()) & (numeric_series > 0)]
    
    if len(valid_vals) > 0:
        if strategy == "mean" or strategy == "ai":
            fill_val = valid_vals.mean()
        elif strategy == "median":
            fill_val = valid_vals.median()
        elif strategy == "mode":
            mode = valid_vals.mode()
            fill_val = mode.iloc[0] if len(mode) > 0 else valid_vals.mean()
        else:
            fill_val = valid_vals.mean()
        
        # Round to integer if close to integer
        if abs(fill_val - round(fill_val)) < 0.01:
            fill_val = int(round(fill_val))
        else:
            fill_val = round(fill_val, 2)
    else:
        fill_val = 1  # Default if no valid values
    
    # Fill missing values
    result = []
    for val in series:
        if is_missing_number(val):
            result.append(fill_val)
        else:
            try:
                num = float(val)
                if num == int(num):
                    result.append(int(num))
                else:
                    result.append(round(num, 2))
            except:
                result.append(fill_val)
    
    return result


def fill_text(series):
    """Fill missing text with mode"""
    valid_vals = [str(v) for v in series if not is_missing_text(v)]
    
    if valid_vals:
        fill_val = max(set(valid_vals), key=valid_vals.count)
    else:
        fill_val = "Standard"
    
    return [fill_val if is_missing_text(v) else str(v) for v in series]


def build_preview(df):
    """Build JSON-safe preview"""
    preview = []
    for _, row in df.head(20).iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_dict[col] = None
            elif isinstance(val, (int, np.integer)):
                row_dict[col] = int(val)
            elif isinstance(val, (float, np.floating)):
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
        
        # Count missing per column
        missing_data = []
        total_missing = 0
        for col in df.columns:
            col_lower = col.lower()
            cnt = 0
            for val in df[col]:
                if 'order' in col_lower:
                    if is_missing_order(val): cnt += 1
                elif 'date' in col_lower:
                    if is_missing_date(val): cnt += 1
                elif col_lower in ['price', 'quantity', 'qty', 'amount']:
                    if is_missing_number(val): cnt += 1
                elif df[col].dtype == 'object':
                    if is_missing_text(val): cnt += 1
                elif pd.isna(val):
                    cnt += 1
            if cnt > 0:
                missing_data.append({"name": str(col)[:12], "missing": cnt})
            total_missing += cnt
        
        duplicates = int(df.duplicated().sum())
        
        num_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object']).columns)
        type_data = []
        if num_cols > 0:
            type_data.append({"name": "Numeric", "value": num_cols})
        if cat_cols > 0:
            type_data.append({"name": "Categorical", "value": cat_cols})
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "quality_score": score,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": total_missing,
                "duplicates": duplicates
            },
            "chart_data": {"missing_data": missing_data[:10], "type_data": type_data},
            "preview": build_preview(df)
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
        # ============ REMOVE DUPLICATES ============
        if action == "remove_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            message = f"AI Analysis: Removed {removed} duplicate rows successfully."
        
        # ============ FILL MISSING VALUES ============
        elif action == "fill_missing":
            filled = 0
            
            for col in df.columns:
                col_lower = col.lower()
                
                # Order ID column
                if 'order' in col_lower:
                    before = sum(1 for v in df[col] if is_missing_order(v))
                    df[col] = fill_order_ids(df[col])
                    filled += before
                
                # Date column
                elif 'date' in col_lower or 'dob' in col_lower:
                    before = sum(1 for v in df[col] if is_missing_date(v))
                    df[col] = fill_dates(df[col])
                    filled += before
                
                # Price, Quantity, Amount columns (NUMERIC)
                elif col_lower in ['price', 'quantity', 'qty', 'amount', 'cost', 'total', 'age', 'salary', 'revenue']:
                    before = sum(1 for v in df[col] if is_missing_number(v))
                    df[col] = fill_numeric(df[col], strategy)
                    filled += before
                
                # Other numeric columns
                elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    numeric_vals = pd.to_numeric(df[col], errors='coerce')
                    before = numeric_vals.isna().sum()
                    if before > 0:
                        df[col] = fill_numeric(df[col], strategy)
                        filled += before
                
                # Text columns
                elif df[col].dtype == 'object':
                    before = sum(1 for v in df[col] if is_missing_text(v))
                    if before > 0:
                        df[col] = fill_text(df[col])
                        filled += before
            
            message = f"AI Analysis: Filled {filled} missing values using {strategy}. All data complete."
        
        # ============ REMOVE OUTLIERS ============
        elif action == "remove_outliers":
            before = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                valid = df[col].dropna()
                if len(valid) > 4:
                    Q1 = valid.quantile(0.25)
                    Q3 = valid.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isna()]
            
            removed = before - len(df)
            message = f"AI Analysis: Removed {removed} outliers using IQR method."
        
        # ============ CLEAN TEXT ============
        elif action == "clean_text":
            cleaned = 0
            for col in df.select_dtypes(include=['object']).columns:
                before = sum(1 for v in df[col] if is_missing_text(v))
                df[col] = fill_text(df[col])
                cleaned += before
            
            message = f"AI Analysis: Cleaned {cleaned} text entries."
        
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        CURRENT_DF = df
        score = calculate_quality_score(df)
        
        # Count remaining missing
        total_missing = 0
        for col in df.columns:
            col_lower = col.lower()
            for val in df[col]:
                if 'order' in col_lower and is_missing_order(val):
                    total_missing += 1
                elif ('date' in col_lower or 'dob' in col_lower) and is_missing_date(val):
                    total_missing += 1
                elif col_lower in ['price', 'quantity', 'qty', 'amount'] and is_missing_number(val):
                    total_missing += 1
                elif df[col].dtype == 'object' and is_missing_text(val):
                    total_missing += 1
        
        return {
            "success": True,
            "message": message,
            "new_score": score,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": total_missing,
                "duplicates": int(df.duplicated().sum())
            },
            "preview": build_preview(df)
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
