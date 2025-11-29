from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.database import execute_query
import logging
from datetime import datetime
import uuid

bp = Blueprint('sessions', __name__)
logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/live', methods=['POST'])
@jwt_required()
def create_live_session():
    try:
        current_user = get_jwt_identity()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        session_id = generate_uuid_str()
        
        execute_query(
            """INSERT INTO live_sessions (id, teacher_id, title, description, scheduled_at, meet_url)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                session_id,
                current_user['id'],
                data['title'],
                data.get('description', ''),
                data['scheduledAt'],
                data.get('meetUrl')
            )
        )
        
        return jsonify({
            'session': {'id': session_id},
            'message': 'Live session created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Create session error: {str(e)}")
        return jsonify({'error': 'Failed to create session'}), 500

@bp.route('/live/<session_id>/join', methods=['POST'])
@jwt_required()
def join_live_session(session_id):
    try:
        current_user = get_jwt_identity()
        
        # Check if already joined
        existing = execute_query(
            """SELECT id FROM session_participants 
               WHERE session_type = 'live' AND session_id = %s AND student_id = %s""",
            (session_id, current_user['id']),
            fetch_one=True
        )
        
        if not existing:
            participant_id = generate_uuid_str()
            execute_query(
                """INSERT INTO session_participants (id, session_type, session_id, student_id)
                   VALUES (%s, 'live', %s, %s)""",
                (participant_id, session_id, current_user['id'])
            )
        
        return jsonify({'message': 'Joined session successfully'}), 201
        
    except Exception as e:
        logger.error(f"Join session error: {str(e)}")
        return jsonify({'error': 'Failed to join session'}), 500

@bp.route('/recorded/<video_id>/join', methods=['POST'])
@jwt_required()
def join_recorded_session(video_id):
    try:
        current_user = get_jwt_identity()
        
        existing = execute_query(
            """SELECT id FROM session_participants 
               WHERE session_type = 'recorded' AND session_id = %s AND student_id = %s""",
            (video_id, current_user['id']),
            fetch_one=True
        )
        
        if not existing:
            participant_id = generate_uuid_str()
            execute_query(
                """INSERT INTO session_participants (id, session_type, session_id, student_id)
                   VALUES (%s, 'recorded', %s, %s)""",
                (participant_id, video_id, current_user['id'])
            )
        
        return jsonify({'message': 'Joined session successfully'}), 201
        
    except Exception as e:
        logger.error(f"Join session error: {str(e)}")
        return jsonify({'error': 'Failed to join session'}), 500

@bp.route('/<session_type>/<session_id>/leave', methods=['POST'])
@jwt_required()
def leave_session(session_type, session_id):
    try:
        current_user = get_jwt_identity()
        
        execute_query(
            """UPDATE session_participants
               SET left_at = CURRENT_TIMESTAMP,
                   duration = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - joined_at))::INTEGER
               WHERE session_type = %s AND session_id = %s AND student_id = %s""",
            (session_type, session_id, current_user['id'])
        )
        
        return jsonify({'message': 'Left session successfully'}), 200
        
    except Exception as e:
        logger.error(f"Leave session error: {str(e)}")
        return jsonify({'error': 'Failed to leave session'}), 500

