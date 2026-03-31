import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# AI Integration with Google Genai (new package)
try:
    from google import genai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class AIEngine:
    """
    AI-powered data cleaning engine with proper handling of:
    - Multiple cleaning operations (idempotent where possible)
    - Mean/Median/Mode strategies working correctly
    - Proper missing value detection and handling
    """
    
    # ==========================================
    # MISSING VALUE DETECTION
    # ==========================================
    
    @staticmethod
    def is_missing_value(value):
        """Check if a value should be considered as missing"""
        if pd.isna(value):
            return True
        if value is None:
            return True
        str_val = str(value).strip().lower()
        if str_val in ['', 'nan', 'none', 'null', 'na', 'n/a', '-', '?', '??']:
            return True
        return False
    
    # ==========================================
    # CUSTOM RULES - STRICT FORMATTING
    # ==========================================
    
    @staticmethod
    def apply_custom_rules(df):
        """
        Apply STRICT formatting rules to ALL columns.
        - Numeric columns (ID, Price, Quantity, Age, etc.) -> Numbers ONLY, NO "Unknown"
        - Text columns (Name, Product, Category) -> "Unknown" allowed
        - Dates -> Validate and fix invalid dates
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            
            # ==================================================
            # RULE 1: DATE COLUMNS -> Validate and fix
            # ==================================================
            if 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
                def clean_date(x):
                    if pd.isna(x):
                        return "00-00-0000"
                    
                    str_x = str(x).strip()
                    
                    if str_x.lower() in ['nan', 'none', '', 'null', 'unknown']:
                        return "00-00-0000"
                    
                    try:
                        normalized = str_x.replace('/', '-')
                        parts = normalized.split('-')
                        
                        if len(parts) == 3 and all(p.isdigit() for p in parts):
                            if len(parts[0]) == 4:  # YYYY-MM-DD
                                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            else:  # DD-MM-YYYY
                                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                            
                            if month < 1 or month > 12 or day < 1 or day > 31:
                                return "00-00-0000"
                            return str_x
                        else:
                            return "00-00-0000"
                    except:
                        return "00-00-0000"
                
                df_clean[col] = df_clean[col].apply(clean_date)
            
            # ==================================================
            # RULE 2: ORDER/ID COLUMNS -> NUMBERS ONLY
            # ==================================================
            elif 'order' in col_lower and ('id' in col_lower or 'number' in col_lower or col_lower == 'order'):
                def format_order(x):
                    try:
                        if pd.isna(x):
                            return 0
                        
                        str_x = str(x).strip()
                        
                        if str_x.lower() in ['nan', 'none', '', 'null', 'unknown']:
                            return 0
                        
                        if str_x.upper().startswith('ORD'):
                            num_part = str_x.upper().replace('ORD', '').strip()
                            if num_part.isdigit():
                                val = int(num_part)
                                return f"ORD {val}" if val > 0 else 0
                            return 0
                        
                        val = int(float(x))
                        if val == 0:
                            return 0
                        return f"ORD {val}"
                    except (ValueError, TypeError):
                        return 0
                
                df_clean[col] = df_clean[col].apply(format_order)
            
            # ==================================================
            # RULE 3: NUMERIC COLUMNS -> NUMBERS ONLY (preserve calculated values)
            # ==================================================
            elif df_clean[col].dtype in ['int64', 'int32', 'float64', 'float32'] or \
                 col_lower in ['age', 'price', 'salary', 'sal', 'amount', 'cost', 'total', 
                               'quantity', 'qty', 'count', 'number', 'num', 'revenue', 'id']:
                def clean_numeric(x):
                    try:
                        if pd.isna(x):
                            return 0
                        
                        str_x = str(x).strip()
                        
                        if str_x.lower() in ['nan', 'none', '', 'null', 'unknown']:
                            return 0
                        
                        val = float(x)
                        
                        if 'quantity' in col_lower or 'qty' in col_lower:
                            return max(0, val)
                        
                        return val
                    except (ValueError, TypeError):
                        return 0
                
                df_clean[col] = df_clean[col].apply(clean_numeric)
            
            # ==================================================
            # RULE 4: PRODUCT NAME COLUMNS -> "Unknown" if missing
            # ==================================================
            elif 'product' in col_lower and 'name' in col_lower:
                df_clean[col] = df_clean[col].fillna("Unknown")
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'None', ''], "Unknown")
                df_clean[col] = df_clean[col].replace(['??', '?'], "Unknown")
            
            # ==================================================
            # RULE 5: NAME COLUMNS -> Proper case + "Unknown" if missing
            # ==================================================
            elif col_lower in ['name', 'firstname', 'lastname', 
                                'fullname', 'customername', 'username']:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].apply(
                    lambda x: x.capitalize() if x and x.lower() not in ['nan', 'none', '', '??', '?'] else "Unknown"
                )
            
            # ==================================================
            # RULE 6: CATEGORY/TEXT COLUMNS -> "Unknown" if missing
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
        """Direct call to Gemini API for AI analysis."""
        if not AI_AVAILABLE:
            return None
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your-gemini-api-key-here':
            return None
        
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return None
    
    @staticmethod
    def ai_analyze_data(df, analysis_type="general"):
        """Use Gemini AI to analyze data for specific cleaning needs."""
        try:
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
            
            ai_response = AIEngine.call_gemini_ai(prompt)
            
            if not ai_response:
                return f"Standard {analysis_type} cleaning applied"
            
            return ai_response[:200]
            
        except Exception as e:
            return f"Standard {analysis_type} cleaning applied"
    
    # ==========================================
    # CLEANING METHODS
    # ==========================================
    
    @staticmethod
    def clean_missing_values(df, strategy='ai', fill_value=None):
        """
        Clean missing values with proper strategy handling.
        Each strategy is now properly implemented.
        """
        df_clean = df.copy()
        
        # Get numeric columns, excluding ID/order columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['order', 'id', 'order_id', 'order_number']
        numeric_cols = [col for col in numeric_cols if col.lower() not in exclude_cols]
        
        # Count initial missing values
        initial_missing = df_clean.isnull().sum().sum()
        message = ""
        filled_count = 0

        if strategy == 'drop_rows':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            rows_dropped = initial_rows - len(df_clean)
            message = f"Dropped {rows_dropped} rows containing missing values. Remaining: {len(df_clean)} rows."
            df_clean = AIEngine.apply_custom_rules(df_clean)
            return df_clean, message

        # Process numeric columns
        if len(numeric_cols) > 0:
            cols_with_missing = [col for col in numeric_cols if df_clean[col].isnull().any()]
            
            if cols_with_missing:
                if strategy == 'ai':
                    # Get AI analysis
                    ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='missing')
                    
                    # Apply MICE imputation
                    try:
                        imputer = IterativeImputer(random_state=42, max_iter=10)
                        df_clean[cols_with_missing] = imputer.fit_transform(df_clean[cols_with_missing])
                        filled_count = initial_missing - df_clean.isnull().sum().sum()
                        message = f"AI Analysis: {ai_analysis[:150]}... Filled {filled_count} missing values using MICE algorithm."
                    except Exception as e:
                        # Fallback to mean if MICE fails
                        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
                        filled_count = initial_missing - df_clean.isnull().sum().sum()
                        message = f"Filled {filled_count} missing values using mean (AI fallback)."
                    
                elif strategy == 'mean':
                    # Calculate mean for each column and fill
                    for col in cols_with_missing:
                        col_mean = df_clean[col].mean()
                        missing_count = df_clean[col].isnull().sum()
                        df_clean[col] = df_clean[col].fillna(col_mean)
                        filled_count += missing_count
                    message = f"Filled {filled_count} missing values with column means."
                    
                elif strategy == 'median':
                    # Calculate median for each column and fill
                    for col in cols_with_missing:
                        col_median = df_clean[col].median()
                        missing_count = df_clean[col].isnull().sum()
                        df_clean[col] = df_clean[col].fillna(col_median)
                        filled_count += missing_count
                    message = f"Filled {filled_count} missing values with column medians."
                
                elif strategy == 'mode':
                    # Calculate mode for each column and fill
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
                    message = f"Filled {filled_count} missing values with constant value: {val}."
            else:
                message = "No missing numeric values found to clean."
        else:
            message = "No numeric columns found to clean."
        
        # Apply custom rules after cleaning
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, message

    @staticmethod
    def remove_outliers(df):
        """AI-POWERED outlier detection and removal using IQR method."""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        initial_rows = len(df_clean)
        
        # Get AI analysis
        ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='outliers')
        
        # Apply IQR method only if we have enough data
        if len(df_clean) > 10:
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Only remove outliers if IQR is meaningful
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
        rows_removed = initial_rows - len(df_clean)
        
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"AI Analysis: {ai_analysis[:100]}... Removed {rows_removed} outliers using IQR method. Remaining: {len(df_clean)} rows."

    @staticmethod
    def clean_categorical_data(df, strategy='unknown'):
        """AI-POWERED text data cleaning."""
        df_clean = df.copy()
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        
        if len(cat_cols) == 0:
            df_clean = AIEngine.apply_custom_rules(df_clean)
            return df_clean, "No text columns found to clean."

        ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='text')
        
        filled_count = 0
        for col in cat_cols:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                if strategy == 'mode':
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                        filled_count += missing_before
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    filled_count += missing_before
        
        df_clean = AIEngine.apply_custom_rules(df_clean)
        msg = f"AI Analysis: {ai_analysis[:100]}... Cleaned {filled_count} missing values in {len(cat_cols)} text columns."
        return df_clean, msg
    
    @staticmethod
    def remove_duplicates(df):
        """AI-POWERED duplicate detection and removal."""
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Get AI analysis
        ai_analysis = AIEngine.ai_analyze_data(df_clean, analysis_type='duplicates')
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        rows_removed = initial_rows - len(df_clean)
        
        df_clean = AIEngine.apply_custom_rules(df_clean)
        return df_clean, f"AI Analysis: {ai_analysis[:100]}... Removed {rows_removed} duplicate rows. Remaining: {len(df_clean)} unique rows."
