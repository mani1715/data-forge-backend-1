from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from services.ai_engine import AIEngine
from services.profiler import DataProfiler
from routes.data_routes import get_current_df, set_current_df

# Load environment variables
load_dotenv()

clean_bp = Blueprint('clean', __name__)

@clean_bp.route('/action', methods=['POST', 'OPTIONS'])
def perform_action():
    print("ACTION API HIT")
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    df = get_current_df()
    
    if df is None:
        return jsonify({
            'error': 'No data uploaded',
            'message': 'Please upload a file first before performing actions',
            'success': False
        }), 200  # Return 200 with error message instead of 400
    
    # Safely parse JSON data
    data = request.json or {}
    
    # Extract action - required field
    action = data.get('action')
    if not action:
        return jsonify({
            'error': 'Missing action',
            'message': 'Please specify an action to perform',
            'valid_actions': ['remove_duplicates', 'fill_missing', 'remove_outliers', 'clean_text'],
            'success': False
        }), 200
    
    # Strategy defaults based on action type
    # Only 'fill_missing' requires strategy, others ignore it
    default_strategies = {
        'fill_missing': 'ai',
        'remove_duplicates': None,
        'remove_outliers': None,
        'clean_text': 'unknown'
    }
    
    strategy = data.get('strategy') or default_strategies.get(action, 'ai')
    fill_value = data.get('fill_value')  # Optional, only used for fill_missing with 'constant' strategy
    
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
            cleaned_df, message = AIEngine.clean_categorical_data(cleaned_df, strategy=strategy)

        else:
            return jsonify({
                'error': 'Invalid action',
                'message': f"Unknown action: '{action}'",
                'valid_actions': ['remove_duplicates', 'fill_missing', 'remove_outliers', 'clean_text'],
                'success': False
            }), 200

        # Custom rules are already applied inside each AIEngine method
        # DO NOT call apply_custom_rules again here to avoid double formatting
        
        set_current_df(cleaned_df)
        new_score = DataProfiler.calculate_quality_score(cleaned_df)

        return jsonify({
            'success': True,
            'message': message,
            'new_score': new_score,
            'preview': cleaned_df.head(20).to_dict(orient='records')
        }), 200

    except Exception as e:
        print(f"Action error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': f"Failed to perform action '{action}'",
            'success': False
        }), 200  # Return 200 with error details instead of 500
