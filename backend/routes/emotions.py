from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from utils.jwt_helpers import get_current_user
import cv2
import numpy as np
import base64
import logging
import uuid
from services.database import execute_query

bp = Blueprint('emotions', __name__)
logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/detect', methods=['POST'])
@jwt_required()
def detect_emotions():
    """
    Receive image frame from frontend and return emotion predictions
    Expected: base64 encoded image or image file
    """
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        
        # Get model service from app context
        model_service = current_app.model_service
        if not model_service:
            return jsonify({'error': 'ML models not loaded'}), 503
        
        # Get image data
        data = request.get_json()
        
        if 'image' in data:
            # Base64 encoded image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif 'image_file' in request.files:
            # Uploaded image file
            file = request.files['image_file']
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Get session info
        session_type = data.get('sessionType', 'recorded')
        session_id = data.get('sessionId')
        timestamp = data.get('timestamp', 0)
        
        # Detect emotions
        detections = model_service.detect_emotions_in_image(image)
        
        if not detections:
            return jsonify({
                'emotions': [],
                'message': 'No faces detected'
            })
        
        # Use first detected face (assuming single student)
        detection = detections[0]
        dominant_emotion = detection['dominant_emotion']
        confidence = detection['confidence']
        engagement_score = detection['engagement_score']
        concentration_score = detection.get('concentration_score', 0.0)
        emotion_state = detection.get('emotion_state', model_service.map_emotion_to_state(dominant_emotion))
        
        # Store emotion data in database
        if session_id:
            try:
                emotion_id = generate_uuid_str()
                # Note: concentration_score column may need to be added to database
                # For now, store engagement_score and calculate concentration from it if needed
                execute_query(
                    """INSERT INTO emotion_data 
                       (id, session_type, session_id, student_id, emotion, confidence, timestamp, engagement_score)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (emotion_id, session_type, session_id, user_id, emotion_state, confidence, timestamp, engagement_score)
                )
            except Exception as e:
                logger.error(f"Error storing emotion data: {str(e)}")
        
        return jsonify({
            'emotion': emotion_state,
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'engagement_score': float(engagement_score),
            'concentration_score': float(concentration_score),
            'all_emotions': detection['emotions'],
            'bbox': detection['bbox']
        })
        
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/stream', methods=['POST'])
@jwt_required()
def stream_emotions():
    """
    Stream endpoint for real-time emotion detection
    Accepts base64 or multipart image frames
    Automatically stores results in database
    Returns emotion, confidence, concentration_score, and timestamp
    """
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        
        # Get model service from app context
        model_service = current_app.model_service
        if not model_service:
            return jsonify({'error': 'ML models not loaded'}), 503
        
        # Get image data
        data = request.get_json() if request.is_json else {}
        
        image = None
        if 'image' in data:
            # Base64 encoded image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            logger.info(f"[EmotionStream] Received base64 image. Length: {len(image_data)} chars")
            image_bytes = base64.b64decode(image_data)
            logger.debug(f"[EmotionStream] Decoded image bytes: {len(image_bytes)} bytes")
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                logger.info(f"[EmotionStream] Decoded image successfully. Shape: {image.shape}, dtype: {image.dtype}")
            else:
                logger.error("[EmotionStream] Failed to decode image from base64")
        elif 'image_file' in request.files:
            # Uploaded image file
            file = request.files['image_file']
            image_bytes = file.read()
            logger.info(f"[EmotionStream] Received image file. Size: {len(image_bytes)} bytes")
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                logger.info(f"[EmotionStream] Decoded image file successfully. Shape: {image.shape}, dtype: {image.dtype}")
            else:
                logger.error("[EmotionStream] Failed to decode image file")
        else:
            logger.error("[EmotionStream] No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        if image is None:
            logger.error("[EmotionStream] Invalid image data - cv2.imdecode returned None")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Get session info (required for stream endpoint)
        session_type = data.get('sessionType', 'recorded')
        session_id = data.get('sessionId')
        timestamp = data.get('timestamp', 0)
        
        if not session_id:
            return jsonify({'error': 'sessionId is required'}), 400
        
        # Detect emotions
        logger.info(f"[EmotionStream] Starting emotion detection for session {session_id}")
        detections = model_service.detect_emotions_in_image(image)
        logger.info(f"[EmotionStream] Emotion detection completed. Found {len(detections)} detection(s)")
        
        if not detections:
            logger.warning(f"[EmotionStream] No faces detected in image. Returning default values.")
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'timestamp': timestamp,
                'message': 'No faces detected'
            })
        
        # Use first detected face (assuming single student)
        detection = detections[0]
        dominant_emotion = detection['dominant_emotion']
        confidence = detection['confidence']
        concentration_score = detection.get('concentration_score', 50.0)
        emotion_state = detection.get('emotion_state', model_service.map_emotion_to_state(dominant_emotion))
        
        # Always store emotion data in database for stream endpoint
        try:
            emotion_id = generate_uuid_str()
            execute_query(
                """INSERT INTO emotion_data 
                   (id, session_type, session_id, student_id, emotion, confidence, timestamp, engagement_score)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (emotion_id, session_type, session_id, user_id, emotion_state, confidence, timestamp, concentration_score / 100.0)
            )
        except Exception as e:
            logger.error(f"Error storing emotion data: {str(e)}")
            # Continue even if storage fails
        
        return jsonify({
            'emotion': emotion_state,
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'concentration_score': float(concentration_score),
            'timestamp': timestamp,
            'all_emotions': detection['emotions']
        })
        
    except Exception as e:
        logger.error(f"Error in emotion streaming: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@bp.route('/record', methods=['POST'])
@jwt_required()
def record_emotion():
    """Record emotion data (for batch processing or manual entry)"""
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        
        data = request.get_json()
        emotion_id = generate_uuid_str()
        
        execute_query(
            """INSERT INTO emotion_data 
               (id, session_type, session_id, student_id, emotion, confidence, timestamp, engagement_score)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                emotion_id,
                data['sessionType'],
                data['sessionId'],
                user_id,
                data['emotion'],
                data['confidence'],
                data['timestamp'],
                data.get('engagementScore')
            )
        )
        
        return jsonify({'message': 'Emotion data recorded'}), 201
        
    except Exception as e:
        logger.error(f"Error recording emotion: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/<session_type>/<session_id>', methods=['GET'])
@jwt_required()
def get_emotion_timeline(session_type, session_id):
    """Get emotion timeline for a session"""
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        user_role = current_user['role']
        
        if user_role == 'teacher':
            # Teachers can see all students
            results = execute_query(
                """SELECT ed.*, CONCAT(u.first_name, ' ', u.last_name) as student_name
                   FROM emotion_data ed
                   JOIN users u ON ed.student_id = u.id
                   WHERE ed.session_type = %s AND ed.session_id = %s
                   ORDER BY ed.timestamp ASC""",
                (session_type, session_id),
                fetch_all=True
            )
        else:
            # Students see only their own
            results = execute_query(
                """SELECT * FROM emotion_data
                   WHERE session_type = %s AND session_id = %s AND student_id = %s
                   ORDER BY timestamp ASC""",
                (session_type, session_id, user_id),
                fetch_all=True
            )
        
        # Format results
        emotions = []
        for row in results:
            emotions.append({
                'id': row[0],
                'emotion': row[4],
                'confidence': float(row[5]),
                'timestamp': row[6],
                'engagement_score': float(row[7]) if row[7] else None
            })
        
        return jsonify({'emotionData': emotions})
        
    except Exception as e:
        logger.error(f"Error fetching emotion timeline: {str(e)}")
        return jsonify({'error': str(e)}), 500

