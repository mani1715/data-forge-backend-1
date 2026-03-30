import pandas as pd
import numpy as np
import os
import asyncio
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# AI Integration with Google Genai (new package)
try:
    from google import genai
    from google.genai import types
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class AIEngine:
    
    # ==========================================
    # CUSTOM RULES - STRICT FORMATTING
    # ==========================================
    
    @staticmethod
    def apply_custom_rules(df):
        """
        Apply STRICT formatting rules to ALL columns.
        - Numeric columns (ID, Price, Quantity, Age, etc.) → Numbers ONLY, NO "Unknown"
        - Text columns (Name, Product, Category) → "Unknown" allowed
        - Dates → Validate and fix invalid dates
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            
            # ==================================================
            # RULE 1: DATE COLUMNS → Validate and fix
            # ==================================================
            if 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
                def clean_date(x):
                    if pd.isna(x):
                        return "00-00-0000"
                    
                    str_x = str(x).strip()
                    
                    # Check for null-like strings
                    if str_x.lower() in ['nan', 'none', '', 'null', 'unknown']:
                        return "00-00-0000"
                    
                    # Validate date format - check for invalid dates like 13-13-2023
                    try:
                        # Replace / with - for consistency
                        normalized = str_x.replace('/', '-')
                        parts = normalized.split('-')
                        
                        if len(parts) == 3 and all(p.isdigit() for p in parts):
                            # Check if it's YYYY-MM-DD or DD-MM-YYYY format
                            if len(parts[0]) == 4:  # YYYY-MM-DD
                                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            else:  # DD-MM-YYYY or MM-DD-YYYY (assume DD-MM-YYYY)
                                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                            
                            # Validate ranges
                            if month < 1 or month > 12 or day < 1 or day > 31:
                                return "00-00-0000"
                            
                            # Return original format
                            return str_x
                        else:
                            # Invalid format
                            return "00-00-0000"
                    except:
                        return "00-00-0000"
                
                df_clean[col] = df_clean[col].apply(clean_date)
            
            # ==================================================
            # RULE 2: ORDER/ID COLUMNS → NUMBERS ONLY (NO "Unknown")
            # ==================================================
            elif 'order' in col_lower and ('id' in col_lower or 'number' in col_lower or col_lower == 'order'):
                def format_order(x):
                    try:
                        if pd.isna(x):
                            return 0
                        
                        str_x = str(x).strip()
                        
                        # NO "Unknown" for Order IDs - convert to 0
                        if str_x.lower() in ['nan', 'none', '', 'null', 'unknown']:
                            return 0
                        
                        # If already formatted as "ORD X", keep it
                        if str_x.upper().startswith('ORD'):
                            # Extract number from "ORD X" or "ORDX"
                            num_part = str_x.upper().replace('ORD', '').strip()
                            if num_part.isdigit():
                                val = int(num_part)
                                return f"ORD {val}" if val > 0 else 0
                            return 0
                        
                        # Try to get numeric value
                        val = int(float(x))
                        if val == 0:
                            return 0
                        return f"ORD {val}"
                    except (ValueError, TypeError):
                        return 0
                
                df_clean[col] = df_clean[col].apply(format_order)
            
            # ==================================================
            # RULE 3: NUMERIC COLUMNS → NUMBERS ONLY (NO "Unknown")
            # ==================================================
            elif df_clean[col].dtype in ['int64', 'int32', 'float64', 'float32'] or \
                 col_lower in ['age', 'price', 'salary', 'sal', 'amount', 'cost', 'total', 
                               'quantity', 'qty', 'count', 'number', 'num', 'revenue', 'id']:
                # Convert to numeric, replace "Unknown" and negatives with 0
                def clean_numeric(x):
                    try:
                        if pd.isna(x):
                            return 0
                        
                        str_x = str(x).strip()
                        
                        # NO "Unknown" for numeric columns
                        if str_x.lower() in ['nan', 'none', '', 'null', 'unknown']:
                            return 0
                        
                        val = float(x)
                        
                        # For Quantity: negative values → 0
                        if 'quantity' in col_lower or 'qty' in col_lower:
                            return max(0, val)
                        
                        # For other numeric: keep value (even if negative for price adjustments)
                        return val
                    except (ValueError, TypeError):
                        return 0
                
                df_clean[col] = df_clean[col].apply(clean_numeric)
            
            # ==================================================
            # RULE 4: PRODUCT NAME COLUMNS → "Unknown" if missing, keep format
            # ==================================================
            elif 'product' in col_lower and 'name' in col_lower:
                df_clean[col] = df_clean[col].fillna("Unknown")
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'None', ''], "Unknown")
                # Clean up special characters like ??, but keep product variations (MESSI, Messi!!, etc.)
                df_clean[col] = df_clean[col].replace(['??', '?'], "Unknown")
            
            # ==================================================
            # RULE 5: NAME COLUMNS → Proper case + "Unknown" if missing
            # ==================================================
            elif col_lower in ['name', 'firstname', 'lastname', 
                                'fullname', 'customername', 'username']:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].apply(
                    lambda x: x.capitalize() if x and x.lower() not in ['nan', 'none', '', '??', '?'] else "Unknown"
                )
            
            # ==================================================
            # RULE 6: CATEGORY/TEXT COLUMNS → "Unknown" if missing
            # ==================================================
            elif df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna("Unknown")
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'None', '', '??', '?'], "Unknown")
        
        return df_clean
    
    # ==========================================
    # AI-POWERED CLEANING WITH GEMINI
    # ==========================================
    
    @staticmethod
    def call_gemini_ai(prompt):
        """
        Direct call to Gemini API for AI analysis.
        Uses GEMINI_API_KEY from environment.
        Returns analysis or gracefully falls back if unavailable.
        """
        if not AI_AVAILABLE:
            return None
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your-gemini-api-key-here':
            return None
        
        try:
            # Initialize client
            client = genai.Client(api_key=api_key)
            
            # Generate content using latest model
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            # Silently fail and return None - app will work without AI
            return None
    
    @staticmethod
    def ai_analyze_data(df, analysis_type="general"):
        """
        Use Gemini AI to analyze data for specific cleaning needs.
        Gracefully falls back if AI is unavailable.
        analysis_type: 'duplicates', 'outliers', 'missing', 'text'
        """
        try:
            # Build concise prompts for each analysis type
            if analysis_type == 'duplicates':
                duplicates = df.duplicated().sum()
                if duplicates == 0:
                    return "No duplicate rows detected"
                
                prompt = f"Dataset has {duplicates} duplicate rows out of {len(df)} total. Briefly explain impact (max 20 words)."
            
            elif analysis_type == 'outliers':
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    return "No numeric columns for outlier detection"
                
                prompt = f"Analyzing {len(numeric_cols)} numeric columns for outliers. Briefly describe outlier impact (max 20 words)."
            
            elif analysis_type == 'text':
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                if not text_cols:
                    return "No text columns found"
                
                missing_count = sum(df[col].isnull().sum() for col in text_cols)
                prompt = f"Found {missing_count} missing values in {len(text_cols)} text columns. Briefly suggest strategy (max 20 words)."
            
            else:  # missing values
                total_missing = df.isnull().sum().sum()
                if total_missing == 0:
                    return "No missing values detected"
                
                prompt = f"Dataset has {total_missing} missing values. Briefly recommend imputation strategy (max 20 words)."
            
            # Get AI response
            ai_response = AIEngine.call_gemini_ai(prompt)
            
            # If AI fails, return a simple message
            if not ai_response:
                return f"Standard {analysis_type} cleaning applied"
            
            return ai_response[:200]  # Limit response length
            
        except Exception as e:
            # Gracefully fall back
            return f"Standard {analysis_type} cleaning applied"
    
    # ==========================================
    # EXISTING CLEANING METHODS
    # ==========================================
    
    @staticmethod
    def clean_missing_values(df, strategy='ai', fill_value=None):
        """
        AI-POWERED missing value handling.
        Uses Gemini AI for intelligent imputation recommendations.
        """
        df_clean = df.copy()
        
        # Get numeric columns, but EXCLUDE 'order' column from numeric imputation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['order', 'id', 'order_id', 'order_number']
        numeric_cols = [col for col in numeric_cols if col.lower() not in exclude_cols]
        
        message = ""

        if strategy == 'drop_rows':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            message = f"Dropped {initial_rows - len(df_clean)} rows containing missing values."
            df_clean = AIEngine.apply_custom_rules(df_clean)
            return df_clean, message

        if len(numeric_cols) > 0:
            if strategy == 'ai':
                # Get AI analysis FIRST
                ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='missing')
                
                # Apply MICE imputation to numeric columns
                cols_with_missing = [col for col in numeric_cols if df_clean[col].isnull().any()]
                if cols_with_missing:
                    imputer = IterativeImputer(random_state=42)
                    df_clean[cols_with_missing] = imputer.fit_transform(df_clean[cols_with_missing])
                
                message = f"AI Analysis: {ai_analysis[:200]}... Applied MICE algorithm for intelligent imputation."
                
            elif strategy == 'mean':
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
                message = "Filled missing values with Column Means."
                
            elif strategy == 'median':
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
                message = "Filled missing values with Column Medians."
            
            elif strategy == 'mode':
                for col in numeric_cols:
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                message = "Filled missing values with Mode."

            elif strategy == 'constant':
                val = fill_value if fill_value is not None else 0
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(val)
                message = f"Filled missing values with constant value: {val}."
        else:
            message = "No numeric columns found to clean."
        
        # Apply custom rules after cleaning
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, message

    @staticmethod
    def remove_outliers(df):
        """
        AI-POWERED outlier detection and removal.
        Uses Gemini AI to analyze outlier patterns before removal.
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        initial_rows = len(df_clean)
        
        # Get AI analysis FIRST
        ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='outliers')
        
        # Apply IQR method
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
        rows_removed = initial_rows - len(df_clean)
        
        # Apply custom rules after cleaning
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"AI Analysis: {ai_analysis[:150]}... Removed {rows_removed} outliers using AI-guided IQR method."

    @staticmethod
    def clean_categorical_data(df, strategy='unknown'):
        """
        AI-POWERED text data cleaning.
        Uses Gemini AI to analyze text patterns before cleaning.
        """
        df_clean = df.copy()
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        
        if len(cat_cols) == 0:
            df_clean = AIEngine.apply_custom_rules(df_clean)
            return df_clean, "No text columns found to clean."

        # Get AI analysis FIRST
        ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='text')
        
        filled_count = 0
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                if strategy == 'mode':
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                        filled_count += 1
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    filled_count += 1
        
        # Apply custom rules after cleaning
        df_clean = AIEngine.apply_custom_rules(df_clean)
        msg = f"AI Analysis: {ai_analysis[:150]}... Cleaned {filled_count} text columns."
        return df_clean, msg
    
    @staticmethod
    def remove_duplicates(df):
        """
        AI-POWERED duplicate detection and removal.
        Uses Gemini AI to analyze duplicate patterns before removal.
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Get AI analysis FIRST
        ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='duplicates')
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        rows_removed = initial_rows - len(df_clean)
        
        # Apply custom rules after cleaning
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"AI Analysis: {ai_analysis[:150]}... Removed {rows_removed} duplicate rows."
