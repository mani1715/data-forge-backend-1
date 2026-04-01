import pandas as pd
import numpy as np

class DataProfiler:
    """
    Data profiling service for calculating quality metrics and chart data.
    """
    
    @staticmethod
    def count_all_issues(df):
        """
        Count ALL data quality issues:
        - Null/NaN values
        - "Unknown" text values
        - 0 in ID/Order columns
        - 0.0 in Price columns
        - 0 in Quantity columns
        - Invalid dates like 00-00-0000
        """
        issue_count = 0
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Count actual nulls
            null_count = df[col].isnull().sum()
            issue_count += null_count
            
            # Check non-null values for issues
            for val in df[col].dropna():
                str_val = str(val).strip()
                str_lower = str_val.lower()
                
                # UNKNOWN values in any column
                if str_lower == 'unknown':
                    issue_count += 1
                
                # 0 in ORDER ID column (should be ORD X format)
                elif 'order' in col_lower and 'id' in col_lower:
                    if str_val == '0' or str_val == '0.0':
                        issue_count += 1
                
                # 0 in ID columns
                elif 'id' in col_lower and str_val in ['0', '0.0']:
                    issue_count += 1
                
                # 0.0 in PRICE column
                elif 'price' in col_lower and str_val in ['0', '0.0']:
                    issue_count += 1
                
                # 0 in QUANTITY column
                elif 'quantity' in col_lower and str_val in ['0', '0.0']:
                    issue_count += 1
                
                # Invalid dates
                elif 'date' in col_lower and str_val in ['00-00-0000', '0000-00-00', '00/00/0000']:
                    issue_count += 1
        
        return int(issue_count)
    
    @staticmethod
    def calculate_quality_score(df):
        """
        Calculates a health score based on ALL issues:
        - Missing/null values
        - Invalid values (0, UNKNOWN, 00-00-0000)
        - Duplicate rows
        
        Score = 100 - penalties
        If issues exist, score CANNOT be 100%
        """
        if df.empty:
            return 0.0
            
        total_cells = df.size
        total_rows = len(df)
        
        # Count ALL issues (nulls + invalid values)
        issue_count = DataProfiler.count_all_issues(df)
        
        # Count duplicates
        duplicate_rows = df.duplicated().sum()
        
        # Calculate penalties
        # Issue penalty: each issue reduces score
        issue_penalty = (issue_count / total_cells) * 100 if total_cells > 0 else 0
        
        # Duplicate penalty: percentage of duplicate rows
        duplicate_penalty = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Total penalty (no caps - show real score)
        total_penalty = issue_penalty + duplicate_penalty
        
        # Calculate score
        score = 100 - total_penalty
        
        # IMPORTANT: If there are ANY issues, score cannot be 100%
        if issue_count > 0 or duplicate_rows > 0:
            score = min(score, 99.9)  # Cap at 99.9 if issues exist
        
        return round(max(0, min(100, score)), 2)

    @staticmethod
    def get_summary(df):
        """
        Returns a dictionary of basic stats.
        Counts ALL issues, not just nulls.
        """
        # Count all issues including UNKNOWN, 0 values, invalid dates
        all_issues = DataProfiler.count_all_issues(df)
        
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': all_issues,  # Show ALL issues, not just nulls
            'duplicates': int(df.duplicated().sum()),
            'column_types': df.dtypes.astype(str).to_dict()
        }
        return summary

    @staticmethod
    def get_chart_data(df):
        """
        Prepares data specifically for frontend charts.
        Counts ALL issues per column.
        """
        # 1. Count issues per column (including UNKNOWN, 0, invalid dates)
        issue_counts = {}
        for col in df.columns:
            col_lower = col.lower()
            issues = 0
            
            # Count nulls
            issues += df[col].isnull().sum()
            
            # Count other issues
            for val in df[col].dropna():
                str_val = str(val).strip()
                str_lower = str_val.lower()
                
                if str_lower == 'unknown':
                    issues += 1
                elif 'order' in col_lower and str_val in ['0', '0.0']:
                    issues += 1
                elif 'id' in col_lower and str_val in ['0', '0.0']:
                    issues += 1
                elif 'price' in col_lower and str_val in ['0', '0.0']:
                    issues += 1
                elif 'quantity' in col_lower and str_val in ['0', '0.0']:
                    issues += 1
                elif 'date' in col_lower and str_val in ['00-00-0000', '0000-00-00', '00/00/0000']:
                    issues += 1
            
            issue_counts[col] = issues
        
        # Sort by issue count and get top 10
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        missing_data = []
        for col, count in sorted_issues:
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

        # 3. Column Statistics
        column_stats = []
        for col in df.columns[:10]:
            issues = issue_counts.get(col, 0)
            filled = len(df) - issues
            column_stats.append({
                'name': col if len(col) <= 10 else col[:8] + '...',
                'fullName': col,
                'missing': int(issues),
                'filled': int(max(0, filled))
            })

        return { 
            'missing_data': missing_data, 
            'type_data': type_data,
            'column_stats': column_stats
        }
