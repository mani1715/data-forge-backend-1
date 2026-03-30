from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from services.ai_engine import AIEngine
from services.profiler import DataProfiler
from routes.data_routes import get_current_df, set_current_df

# Load environment variables
load_dotenv()

clean_bp = Blueprint('clean', __name__)

@clean_bp.route('/action', methods=['POST'])
def perform_action():
    df = get_current_df()
    
    if df is None:
        return jsonify({'error': 'No data uploaded'}), 400

    data = request.json
    action = data.get('action')
    strategy = data.get('strategy', 'ai')
    fill_value = data.get('fill_value')
    
    cleaned_df = df.copy()
    message = "No action performed"

    try:
        if action == 'remove_duplicates':
            cleaned_df, message = AIEngine.remove_duplicates(cleaned_df)

        elif action == 'fill_missing':
            cleaned_df, message = AIEngine.clean_missing_values(cleaned_df, strategy=strategy, fill_value=fill_value)

        elif action == 'remove_outliers':
            cleaned_df, message = AIEngine.remove_outliers(cleaned_df)

        elif action == 'clean_text':
            cleaned_df, message = AIEngine.clean_categorical_data(cleaned_df, strategy='unknown')

        else:
            return jsonify({'error': 'Invalid action'}), 400

        # Custom rules are already applied inside each AIEngine method
        # DO NOT call apply_custom_rules again here to avoid double formatting
        
        set_current_df(cleaned_df)
        new_score = DataProfiler.calculate_quality_score(cleaned_df)

        return jsonify({
            'message': message,
            'new_score': new_score,
            'preview': cleaned_df.head(20).to_dict(orient='records')  # No fillna - data already clean
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
