from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.database import execute_query
import logging
import uuid

bp = Blueprint('interventions', __name__)
logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/trigger', methods=['POST'])
@jwt_required()
def trigger_intervention():
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        intervention_id = generate_uuid_str()
        execute_query(
            """INSERT INTO interventions (id, session_id, student_id, intervention_type, triggered_emotion)
               VALUES (%s, %s, %s, %s, %s)""",
            (intervention_id, data['sessionId'], current_user['id'], data['interventionType'], data['triggeredEmotion'])
        )
        
        return jsonify({
            'intervention': {'id': intervention_id},
            'message': 'Intervention triggered'
        }), 201
        
    except Exception as e:
        logger.error(f"Trigger intervention error: {str(e)}")
        return jsonify({'error': 'Failed to trigger intervention'}), 500

@bp.route('/<intervention_id>/complete', methods=['POST'])
@jwt_required()
def complete_intervention(intervention_id):
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        execute_query(
            """UPDATE interventions
               SET completed_at = CURRENT_TIMESTAMP, duration = %s
               WHERE id = %s AND student_id = %s""",
            (data['duration'], intervention_id, current_user['id'])
        )
        
        return jsonify({'message': 'Intervention completed'}), 200
        
    except Exception as e:
        logger.error(f"Complete intervention error: {str(e)}")
        return jsonify({'error': 'Failed to complete intervention'}), 500

