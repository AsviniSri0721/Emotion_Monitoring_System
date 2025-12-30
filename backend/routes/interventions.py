from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from utils.jwt_helpers import get_current_user
from services.database import execute_query
import logging
import uuid
import os

bp = Blueprint('interventions', __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'mp4', 'webm', 'ogg', 'avi', 'mov'}

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/videos', methods=['GET'])
@jwt_required()
def get_intervention_videos():
    try:
        results = execute_query(
            """SELECT id, title, description, file_path, duration, created_at
               FROM intervention_clips
               ORDER BY created_at DESC""",
            fetch_all=True
        )
        
        videos = []
        for row in results:
            videos.append({
                'id': row[0] if isinstance(row[0], str) else str(row[0]),
                'title': row[1],
                'description': row[2],
                'file_path': row[3],
                'duration': row[4],
                'created_at': row[5].isoformat() if row[5] else None
            })
        
        return jsonify({'videos': videos})
    except Exception as e:
        logger.error(f"Get intervention videos error: {str(e)}")
        return jsonify({'error': 'Failed to fetch videos'}), 500

@bp.route('/videos/upload', methods=['POST'])
@jwt_required()
def upload_intervention_video():
    try:
        current_user = get_current_user()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized - Teacher role required'}), 403

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Form data
        title = request.form.get('title', 'Untitled')
        description = request.form.get('description', '')
        try:
            duration = int(request.form.get('duration', 60)) # Default 60s
        except ValueError:
            return jsonify({'error': 'Invalid duration format'}), 400

        # Save file
        upload_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"intervention_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # Save to DB
        video_id = generate_uuid_str()
        try:
            execute_query(
                """INSERT INTO intervention_clips (id, title, description, file_path, duration)
                   VALUES (%s, %s, %s, %s, %s)""",
                (video_id, title, description, filename, duration)
            )
        except Exception as db_error:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise db_error

        return jsonify({
            'video': {
                'id': video_id,
                'title': title,
                'description': description,
                'file_path': filename,
                'duration': duration
            },
            'message': 'Intervention video uploaded successfully'
        }), 201

    except Exception as e:
        logger.error(f"Intervention video upload error: {str(e)}")
        return jsonify({'error': 'Failed to upload video'}), 500

@bp.route('/videos/<video_id>', methods=['GET'])
@jwt_required()
def get_intervention_video(video_id):
    try:
        result = execute_query(
            """SELECT id, title, description, file_path, duration, created_at
               FROM intervention_clips
               WHERE id = %s""",
            (video_id,),
            fetch_one=True
        )
        
        if not result:
            return jsonify({'error': 'Video not found'}), 404
            
        return jsonify({
            'video': {
                'id': result[0] if isinstance(result[0], str) else str(result[0]),
                'title': result[1],
                'description': result[2],
                'file_path': result[3],
                'duration': result[4],
                'created_at': result[5].isoformat() if result[5] else None
            }
        })
    except Exception as e:
        logger.error(f"Get intervention video error: {str(e)}")
        return jsonify({'error': 'Failed to fetch video'}), 500

@bp.route('/trigger', methods=['POST'])
@jwt_required()
def trigger_intervention():
    """
    Trigger an intervention manually or automatically
    Accepts: sessionId, interventionType, triggeredEmotion, concentrationScore (optional)
    """
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        intervention_id = generate_uuid_str()
        concentration_score = data.get('concentrationScore', None)
        triggered_emotion = data.get('triggeredEmotion', 'low_concentration')
        
        # Store intervention record
        execute_query(
            """INSERT INTO interventions (id, session_id, student_id, intervention_type, triggered_emotion)
               VALUES (%s, %s, %s, %s, %s)""",
            (intervention_id, data['sessionId'], current_user['id'], data['interventionType'], triggered_emotion)
        )
        
        return jsonify({
            'intervention': {'id': intervention_id},
            'message': 'Intervention triggered',
            'concentration_score': concentration_score
        }), 201
        
    except Exception as e:
        logger.error(f"Trigger intervention error: {str(e)}")
        return jsonify({'error': 'Failed to trigger intervention'}), 500

@bp.route('/check/<session_type>/<session_id>', methods=['GET'])
@jwt_required()
def check_intervention_needed(session_type, session_id):
    """
    Check if intervention should be triggered based on recent concentration scores
    Rule: If concentration_score < 40 for 10 consecutive frames → trigger intervention
    Returns: { should_trigger: bool, consecutive_low: int, latest_score: float }
    """
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        
        # Get last 15 emotion records for this session (to check for 10 consecutive low scores)
        results = execute_query(
            """SELECT engagement_score, timestamp 
               FROM emotion_data
               WHERE session_type = %s AND session_id = %s AND student_id = %s
               ORDER BY timestamp DESC
               LIMIT 15""",
            (session_type, session_id, user_id),
            fetch_all=True
        )
        
        if not results:
            return jsonify({
                'should_trigger': False,
                'consecutive_low': 0,
                'latest_score': None,
                'message': 'No emotion data found'
            })
        
        # Check for consecutive low concentration scores
        # engagement_score is stored as 0-1, so 0.4 = 40%
        threshold = 0.4
        consecutive_low = 0
        latest_score = float(results[0][0]) if results[0][0] else 0.0
        
        for row in results:
            score = float(row[0]) if row[0] else 0.0
            if score < threshold:
                consecutive_low += 1
            else:
                break  # Reset count if we hit a high score
        
        should_trigger = consecutive_low >= 10
        
        return jsonify({
            'should_trigger': should_trigger,
            'consecutive_low': consecutive_low,
            'latest_score': latest_score * 100.0,  # Convert to 0-100 scale
            'threshold': 40.0
        })
        
    except Exception as e:
        logger.error(f"Check intervention error: {str(e)}")
        return jsonify({'error': 'Failed to check intervention status'}), 500

@bp.route('/<intervention_id>/complete', methods=['POST'])
@jwt_required()
def complete_intervention(intervention_id):
    try:
        current_user = get_current_user()
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

