import pandas as pd

class DataProfiler:
    
    @staticmethod
    def calculate_quality_score(df):
        """
        Calculates a health score based on missing values and duplicates.
        """
        if df.empty:
            return 0.0
            
        total_cells = df.size
        total_rows = len(df)
        
        # Calculate Missing Penalty
        missing_cells = df.isnull().sum().sum()
        missing_penalty = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate Duplicate Penalty
        duplicate_rows = df.duplicated().sum()
        duplicate_penalty = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Final Score
        score = 100 - (missing_penalty + duplicate_penalty)
        return round(max(0, score), 2)

    @staticmethod
    def get_summary(df):
        """
        Returns a dictionary of basic stats.
        """
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'column_types': df.dtypes.astype(str).to_dict()
        }
        return summary

    @staticmethod
    def get_chart_data(df):
        """
        Prepares data specifically for frontend charts.
        """
        # 1. Missing Data per Column (Top 5)
        missing_counts = df.isnull().sum().sort_values(ascending=False).head(5)
        missing_data = []
        for col, count in missing_counts.items():
            if count > 0:
                missing_data.append({
                    'name': col, 
                    'missing': int(count)
                })

        # 2. Data Types Distribution
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        
        type_data = [
            { 'name': 'Numeric', 'value': numeric_count },
            { 'name': 'Categorical', 'value': categorical_count }
        ]

        return { 'missing_data': missing_data, 'type_data': type_data }