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
    """End a live session and generate engagement reports"""
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
        
        # Generate engagement reports for all students in this session
        from live_sessions.service import LiveSessionService
        report_result = LiveSessionService.generate_engagement_reports(session_id)
        
        logger.info(f"Session {session_id} ended. {report_result.get('reports_generated', 0)} report(s) generated.")
        
        return jsonify({
            'message': 'Session ended successfully',
            'reports_generated': report_result.get('reports_generated', 0),
            'report_message': report_result.get('message', '')
        }), 200
        
    except Exception as e:
        logger.error(f"End session error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to end session'}), 500

@bp.route('/live/<session_id>', methods=['DELETE'])
@jwt_required()
def delete_live_session(session_id):
    """
    Delete a live session (teacher only, owner only)
    
    NOTE: This will delete the session record but will PRESERVE:
    - Engagement reports (engagement_reports table) - these remain for historical analysis
    - Live session logs (live_session_logs table) - these remain for historical data
    
    Only the session record itself is deleted. Reports and logs are kept for record-keeping purposes.
    """
    try:
        current_user = get_current_user()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if session exists and belongs to current teacher
        result = execute_query(
            """SELECT ls.id, ls.teacher_id
               FROM live_sessions ls
               WHERE ls.id = %s""",
            (session_id,),
            fetch_one=True
        )
        
        if not result:
            return jsonify({'error': 'Session not found'}), 404
        
        if result[1] != current_user['id']:
            return jsonify({'error': 'Unauthorized - You can only delete your own sessions'}), 403
        
        # Delete only the session record from live_sessions table
        # Engagement reports and logs are preserved (no foreign key CASCADE constraint)
        # This allows teachers to delete sessions while keeping historical data
        execute_query(
            """DELETE FROM live_sessions WHERE id = %s""",
            (session_id,)
        )
        
        logger.info(f"Session {session_id} deleted by teacher {current_user['id']}. Reports and logs preserved.")
        
        return jsonify({'message': 'Session deleted successfully. Engagement reports and logs have been preserved.'}), 200
        
    except Exception as e:
        logger.error(f"Delete session error: {str(e)}")
        return jsonify({'error': 'Failed to delete session'}), 500

@bp.route('/live/<session_id>', methods=['PUT', 'PATCH'])
@jwt_required()
def update_live_session(session_id):
    """Update live session details (teacher only, owner only)"""
    try:
        current_user = get_current_user()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if session exists and belongs to current teacher
        result = execute_query(
            """SELECT ls.id, ls.teacher_id
               FROM live_sessions ls
               WHERE ls.id = %s""",
            (session_id,),
            fetch_one=True
        )
        
        if not result:
            return jsonify({'error': 'Session not found'}), 404
        
        if result[1] != current_user['id']:
            return jsonify({'error': 'Unauthorized - You can only edit your own sessions'}), 403
        
        data = request.get_json()
        title = data.get('title')
        description = data.get('description')
        meet_url = data.get('meetUrl')
        scheduled_at = data.get('scheduledAt')
        
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        
        # Build update query dynamically
        updates = []
        params = []
        
        if title:
            updates.append("title = %s")
            params.append(title)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if meet_url is not None:
            updates.append("meet_url = %s")
            params.append(meet_url)
        if scheduled_at is not None:
            updates.append("scheduled_at = %s")
            params.append(scheduled_at)
        
        if not updates:
            return jsonify({'error': 'No fields to update'}), 400
        
        updates.append("updated_at = NOW()")
        params.append(session_id)
        
        query = f"UPDATE live_sessions SET {', '.join(updates)} WHERE id = %s"
        execute_query(query, tuple(params))
        
        return jsonify({'message': 'Session updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Update session error: {str(e)}")
        return jsonify({'error': 'Failed to update session'}), 500

