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
        
        # Clean up file
        os.remove(file_path)
        
        # Fill NaN for text columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna("Unknown")
        
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
        
        # Preview - convert to JSON-safe
        preview = []
        for _, row in df.head(20).iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = float(val) if np.isfinite(val) else None
                else:
                    row_dict[col] = str(val)
            preview.append(row_dict)
        
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
            message = f"Removed {before - len(df)} duplicates"
            
        elif action == "fill_missing":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            filled = 0
            
            for col in numeric_cols:
                missing = df[col].isnull().sum()
                if missing > 0:
                    if strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "mode":
                        mode = df[col].mode()
                        if len(mode) > 0:
                            df[col] = df[col].fillna(mode[0])
                    else:  # constant or ai
                        df[col] = df[col].fillna(0)
                    filled += missing
            
            message = f"Filled {filled} missing values with {strategy}"
            
        elif action == "remove_outliers":
            before = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
            
            message = f"Removed {before - len(df)} outliers"
            
        elif action == "clean_text":
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna("Unknown")
            message = "Cleaned text columns"
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        # Update and recalculate
        CURRENT_DF = df
        
        total_cells = df.size
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        missing_pct = (missing / total_cells * 100) if total_cells > 0 else 0
        dup_pct = (duplicates / len(df) * 100) if len(df) > 0 else 0
        score = round(max(0, 100 - missing_pct - dup_pct), 2)
        
        # Preview
        preview = []
        for _, row in df.head(20).iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = float(val) if np.isfinite(val) else None
                else:
                    row_dict[col] = str(val)
            preview.append(row_dict)
        
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
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"}
    )


# For Railway - run with: uvicorn server:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
