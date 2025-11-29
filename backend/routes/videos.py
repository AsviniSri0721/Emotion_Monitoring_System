from flask import Blueprint, request, jsonify, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from utils.jwt_helpers import get_current_user
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
        # Debug logging
        logger.info("=" * 50)
        logger.info("VIDEO UPLOAD REQUEST RECEIVED")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request files keys: {list(request.files.keys())}")
        logger.info(f"Request form keys: {list(request.form.keys())}")
        logger.info(f"Request form data: {dict(request.form)}")
        
        # Get user identity
        try:
            current_user = get_current_user()
            logger.info(f"Current user: {current_user}")
            logger.info(f"User ID: {current_user.get('id')}")
            logger.info(f"User role: {current_user.get('role')}")
        except Exception as jwt_error:
            logger.error(f"JWT error: {str(jwt_error)}")
            return jsonify({'error': 'Authentication failed', 'details': str(jwt_error)}), 401
        
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized - Teacher role required'}), 403
        
        if 'video' not in request.files:
            logger.error("No 'video' key in request.files")
            logger.error(f"Available keys: {list(request.files.keys())}")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        logger.info(f"File received: {file.filename}")
        logger.info(f"File content type: {file.content_type}")
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
        
        logger.info(f"File validation passed: {file.filename}")
        
        # Ensure upload directory exists
        upload_dir = request.app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(upload_dir, filename)
        
        # Save file
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        logger.info(f"File saved successfully")
        
        # Verify file was saved
        if not os.path.exists(filepath):
            raise Exception("File was not saved successfully")
        
        file_size = os.path.getsize(filepath)
        logger.info(f"Saved file size: {file_size} bytes")
        
        # Get form data
        title = request.form.get('title', 'Untitled')
        description = request.form.get('description', '')
        logger.info(f"Title: {title}, Description: {description}")
        
        # Generate UUID for video
        video_id = generate_uuid_str()
        logger.info(f"Generated video ID: {video_id}")
        
        # Save to database
        try:
            logger.info(f"Inserting into database: teacher_id={current_user['id']}, title={title}")
            execute_query(
                """INSERT INTO videos (id, teacher_id, title, description, file_path)
                   VALUES (%s, %s, %s, %s, %s)""",
                (video_id, current_user['id'], title, description, filename)
            )
            logger.info(f"Video saved to database with ID: {video_id}")
            logger.info("=" * 50)
        except Exception as db_error:
            # If database insert fails, delete the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info("Deleted uploaded file due to database error")
            logger.error(f"Database error: {str(db_error)}")
            logger.error(f"Database error type: {type(db_error).__name__}")
            import traceback
            logger.error(f"Database traceback: {traceback.format_exc()}")
            raise Exception(f"Database error: {str(db_error)}")
        
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
        error_msg = str(e)
        logger.error(f"Video upload error: {error_msg}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return appropriate status code
        status_code = 500
        if 'database' in error_msg.lower() or 'connection' in error_msg.lower():
            status_code = 503  # Service Unavailable
        elif 'unauthorized' in error_msg.lower() or 'permission' in error_msg.lower():
            status_code = 403
        elif 'not found' in error_msg.lower():
            status_code = 404
        
        return jsonify({
            'error': 'Failed to upload video',
            'details': error_msg,
            'message': 'Please check backend logs for more details'
        }), status_code

@bp.route('/', methods=['GET'])
@bp.route('', methods=['GET'])  # Handle both with and without trailing slash
@jwt_required()
def get_videos():
    try:
        current_user = get_jwt_identity()
        logger.info(f"Token validated for videos: {current_user}")
        
        if not current_user or 'id' not in current_user or 'role' not in current_user:
            logger.error(f"Invalid user data from token: {current_user}")
            return jsonify({'error': 'Invalid token payload'}), 422
        
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

