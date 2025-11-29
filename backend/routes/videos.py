from flask import Blueprint, request, jsonify, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
import uuid
from services.database import execute_query
import logging

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

bp = Blueprint('videos', __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'mp4', 'webm', 'ogg', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_video():
    try:
        current_user = get_jwt_identity()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized'}), 403
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(request.app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get form data
        title = request.form.get('title', 'Untitled')
        description = request.form.get('description', '')
        
        # Generate UUID for video
        video_id = generate_uuid_str()
        
        # Save to database
        execute_query(
            """INSERT INTO videos (id, teacher_id, title, description, file_path)
               VALUES (%s, %s, %s, %s, %s)""",
            (video_id, current_user['id'], title, description, filename)
        )
        
        return jsonify({
            'video': {
                'id': video_id,
                'title': title,
                'description': description,
                'file_path': filename
            },
            'message': 'Video uploaded successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        return jsonify({'error': 'Failed to upload video'}), 500

@bp.route('/', methods=['GET'])
@jwt_required()
def get_videos():
    try:
        current_user = get_jwt_identity()
        
        if current_user['role'] == 'teacher':
            results = execute_query(
                """SELECT v.id, v.title, v.description, v.file_path, v.duration, v.created_at,
                          CONCAT(u.first_name, ' ', u.last_name) as teacher_name
                   FROM videos v
                   JOIN users u ON v.teacher_id = u.id
                   WHERE v.teacher_id = %s
                   ORDER BY v.created_at DESC""",
                (current_user['id'],),
                fetch_all=True
            )
        else:
            results = execute_query(
                """SELECT v.id, v.title, v.description, v.file_path, v.duration, v.created_at,
                          CONCAT(u.first_name, ' ', u.last_name) as teacher_name
                   FROM videos v
                   JOIN users u ON v.teacher_id = u.id
                   ORDER BY v.created_at DESC""",
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
                'created_at': row[5].isoformat() if row[5] else None,
                'teacher_name': row[6]
            })
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        logger.error(f"Get videos error: {str(e)}")
        return jsonify({'error': 'Failed to fetch videos'}), 500

@bp.route('/<video_id>', methods=['GET'])
@jwt_required()
def get_video(video_id):
    try:
        result = execute_query(
            """SELECT v.id, v.title, v.description, v.file_path, v.duration, v.created_at,
                      CONCAT(u.first_name, ' ', u.last_name) as teacher_name, u.id as teacher_id
               FROM videos v
               JOIN users u ON v.teacher_id = u.id
               WHERE v.id = %s""",
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
                'created_at': result[5].isoformat() if result[5] else None,
                'teacher_name': result[6],
                'teacher_id': result[7] if isinstance(result[7], str) else str(result[7])
            }
        })
        
    except Exception as e:
        logger.error(f"Get video error: {str(e)}")
        return jsonify({'error': 'Failed to fetch video'}), 500

# Video serving is handled in app.py

