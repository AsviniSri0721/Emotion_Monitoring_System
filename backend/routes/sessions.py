from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from utils.jwt_helpers import get_current_user
from services.database import execute_query
import logging
from datetime import datetime
import uuid

bp = Blueprint('sessions', __name__)
logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/live', methods=['GET', 'POST'])
@jwt_required()
def live_sessions():
    # The @jwt_required() decorator validates the token before this function runs
    # So get_jwt_identity() should work here
    if request.method == 'GET':
        return get_live_sessions()
    else:
        return create_live_session()

def get_live_sessions():
    """GET endpoint to fetch all live sessions"""
    try:
        current_user = get_current_user()
        logger.info(f"Token validated for user: {current_user}")
        
        if not current_user['id'] or not current_user['role']:
            logger.error(f"Invalid user data from token: {current_user}")
            return jsonify({'error': 'Invalid token payload'}), 422
        
        # Teachers see their own sessions, students see all available sessions
        if current_user['role'] == 'teacher':
            sessions = execute_query(
                """SELECT ls.id, ls.title, ls.description, ls.scheduled_at, 
                          ls.meet_url, ls.status, ls.created_at,
                          CONCAT(u.first_name, ' ', u.last_name) as teacher_name
                   FROM live_sessions ls
                   JOIN users u ON ls.teacher_id = u.id
                   WHERE ls.teacher_id = %s
                   ORDER BY ls.scheduled_at DESC""",
                (current_user['id'],),
                fetch_all=True
            )
        else:
            # Students see all available sessions
            sessions = execute_query(
                """SELECT ls.id, ls.title, ls.description, ls.scheduled_at, 
                          ls.meet_url, ls.status, ls.created_at,
                          CONCAT(u.first_name, ' ', u.last_name) as teacher_name
                   FROM live_sessions ls
                   JOIN users u ON ls.teacher_id = u.id
                   ORDER BY ls.scheduled_at DESC""",
                None,
                fetch_all=True
            )
        
        # Convert to list of dicts
        session_list = []
        for row in sessions:
            session_list.append({
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'scheduled_at': row[3].isoformat() if row[3] else None,
                'meet_url': row[4],
                'status': row[5] or 'scheduled',
                'created_at': row[6].isoformat() if row[6] else None,
                'teacher_name': row[7]
            })
        
        return jsonify({'sessions': session_list}), 200
        
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        return jsonify({'error': 'Failed to fetch sessions'}), 500

def create_live_session():
    try:
        current_user = get_current_user()
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
        current_user = get_current_user()
        
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
        current_user = get_current_user()
        
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
        current_user = get_current_user()
        
        # MySQL compatible query
        execute_query(
            """UPDATE session_participants
               SET left_at = NOW(),
                   duration = TIMESTAMPDIFF(SECOND, joined_at, NOW())
               WHERE session_type = %s AND session_id = %s AND student_id = %s""",
            (session_type, session_id, current_user['id'])
        )
        
        return jsonify({'message': 'Left session successfully'}), 200
        
    except Exception as e:
        logger.error(f"Leave session error: {str(e)}")
        return jsonify({'error': 'Failed to leave session'}), 500

@bp.route('/live/<session_id>/start', methods=['POST'])
@jwt_required()
def start_live_session(session_id):
    """Start a live session (change status to 'live')"""
    try:
        current_user = get_current_user()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Update session status to 'live'
        execute_query(
            """UPDATE live_sessions 
               SET status = 'live', started_at = NOW()
               WHERE id = %s AND teacher_id = %s""",
            (session_id, current_user['id'])
        )
        
        return jsonify({'message': 'Session started successfully'}), 200
        
    except Exception as e:
        logger.error(f"Start session error: {str(e)}")
        return jsonify({'error': 'Failed to start session'}), 500

@bp.route('/live/<session_id>/end', methods=['POST'])
@jwt_required()
def end_live_session(session_id):
    """End a live session (change status to 'ended')"""
    try:
        current_user = get_current_user()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Update session status to 'ended'
        execute_query(
            """UPDATE live_sessions 
               SET status = 'ended', ended_at = NOW()
               WHERE id = %s AND teacher_id = %s""",
            (session_id, current_user['id'])
        )
        
        return jsonify({'message': 'Session ended successfully'}), 200
        
    except Exception as e:
        logger.error(f"End session error: {str(e)}")
        return jsonify({'error': 'Failed to end session'}), 500

