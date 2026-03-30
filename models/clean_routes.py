from flask import Blueprint, request, jsonify
from services.ai_engine import AIEngine
from services.profiler import DataProfiler
from routes.data_routes import get_current_df, set_current_df

clean_bp = Blueprint('clean', __name__)

@clean_bp.route('/action', methods=['POST'])
def perform_action():
    # Get the current data
    df = get_current_df()
    
    if df is None:
        return jsonify({'error': 'No data uploaded'}), 400

    data = request.json
    action = data.get('action')
    
    cleaned_df = df.copy()
    message = "No action performed"

    try:
        if action == 'remove_duplicates':
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed = initial_rows - len(cleaned_df)
            message = f"Removed {removed} duplicate rows."

        elif action == 'fill_missing_ai':
            cleaned_df, message = AIEngine.clean_missing_values(cleaned_df, strategy='ai')

        elif action == 'fill_missing_mean':
            cleaned_df, message = AIEngine.clean_missing_values(cleaned_df, strategy='mean')

        elif action == 'remove_outliers':
            cleaned_df, message = AIEngine.remove_outliers(cleaned_df)

        else:
            return jsonify({'error': 'Invalid action'}), 400

        # Update the global data
        set_current_df(cleaned_df)

        # Calculate new score
        new_score = DataProfiler.calculate_quality_score(cleaned_df)

        return jsonify({
            'message': message,
            'new_score': new_score,
            'preview': cleaned_df.head(20).fillna("NaN").to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500