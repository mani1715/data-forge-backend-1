import pandas as pd
import numpy as np

class DataProfiler:
    """
    Data profiling service for calculating quality metrics and chart data.
    """
    
    @staticmethod
    def calculate_quality_score(df):
        """
        Calculates a health score based on missing values and duplicates.
        Score starts at 100 and decreases based on issues found.
        """
        if df.empty:
            return 0.0
            
        total_cells = df.size
        total_rows = len(df)
        
        # Calculate Missing Penalty
        missing_cells = df.isnull().sum().sum()
        
        # Also count "Unknown", 0 in ID columns, and other missing indicators
        for col in df.columns:
            col_lower = col.lower()
            for val in df[col]:
                if pd.notna(val):
                    str_val = str(val).lower().strip()
                    # Count "Unknown" in text columns as partial missing
                    if str_val == 'unknown' and df[col].dtype == 'object':
                        missing_cells += 0.5  # Half penalty for "Unknown"
                    # Count 0 in ID/numeric columns that shouldn't be 0
                    elif str_val == '0' and any(x in col_lower for x in ['id', 'age', 'price', 'quantity']):
                        missing_cells += 0.5
                    # Count invalid dates
                    elif str_val in ['00-00-0000', '0000-00-00']:
                        missing_cells += 0.5
        
        missing_penalty = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate Duplicate Penalty
        duplicate_rows = df.duplicated().sum()
        duplicate_penalty = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Final Score - cap penalties
        missing_penalty = min(missing_penalty, 50)  # Max 50% penalty for missing
        duplicate_penalty = min(duplicate_penalty, 30)  # Max 30% penalty for duplicates
        
        score = 100 - (missing_penalty + duplicate_penalty)
        return round(max(0, min(100, score)), 2)

    @staticmethod
    def count_missing_values(df):
        """
        Count all types of missing values including nulls, empty strings, 
        "Unknown", zeros in ID columns, and invalid dates.
        """
        missing_count = 0
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Count actual nulls
            missing_count += df[col].isnull().sum()
            
            # Count other missing indicators
            for val in df[col]:
                if pd.notna(val):
                    str_val = str(val).lower().strip()
                    if str_val in ['', 'nan', 'none', 'null', 'unknown']:
                        missing_count += 1
                    elif str_val == '0' and any(x in col_lower for x in ['id', 'age', 'price', 'quantity', 'order']):
                        missing_count += 1
                    elif str_val in ['00-00-0000', '0000-00-00']:
                        missing_count += 1
        
        return int(missing_count)

    @staticmethod
    def get_summary(df):
        """
        Returns a dictionary of basic stats.
        Uses same logic as quality score for consistency.
        """
        # Count true missing values (nulls only for summary)
        true_missing = int(df.isnull().sum().sum())
        
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': true_missing,
            'duplicates': int(df.duplicated().sum()),
            'column_types': df.dtypes.astype(str).to_dict()
        }
        return summary

    @staticmethod
    def get_chart_data(df):
        """
        Prepares data specifically for frontend charts.
        """
        # 1. Missing Data per Column (Top 10)
        missing_counts = df.isnull().sum().sort_values(ascending=False).head(10)
        missing_data = []
        for col, count in missing_counts.items():
            if count > 0:
                missing_data.append({
                    'name': col if len(col) <= 15 else col[:12] + '...', 
                    'missing': int(count)
                })

        # 2. Data Types Distribution
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        datetime_count = len(df.select_dtypes(include=['datetime']).columns)
        
        type_data = []
        if numeric_count > 0:
            type_data.append({ 'name': 'Numeric', 'value': numeric_count })
        if categorical_count > 0:
            type_data.append({ 'name': 'Categorical', 'value': categorical_count })
        if datetime_count > 0:
            type_data.append({ 'name': 'DateTime', 'value': datetime_count })

        # 3. Column Statistics (for additional charts)
        column_stats = []
        for col in df.columns[:10]:  # First 10 columns
            missing = int(df[col].isnull().sum())
            filled = len(df) - missing
            column_stats.append({
                'name': col if len(col) <= 10 else col[:8] + '...',
                'fullName': col,
                'missing': missing,
                'filled': filled
            })

        return { 
            'missing_data': missing_data, 
            'type_data': type_data,
            'column_stats': column_stats
        }
