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
        
        # Detect faces
        faces = model_service.detect_faces(image)
        
        if not faces:
            return jsonify({
                'emotions': [],
                'message': 'No faces detected'
            })
        
        # Use first detected face
        face = faces[0]
        box = face['bbox']
        
        # Preprocess face
        face_tensor, preprocess_error = model_service.preprocess_face(image, box)
        if face_tensor is None or preprocess_error is not None:
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'message': 'Face preprocessing failed',
                'bbox': box
            })
        
        # Predict emotions
        emotion_probs = model_service.predict_emotion_vector(face_tensor)
        emotion_label = max(emotion_probs, key=emotion_probs.get)
        concentration = model_service.compute_concentration(emotion_probs)
        
        confidence = emotion_probs[emotion_label]
        emotion_state = model_service.map_emotion_to_state(emotion_label)
        engagement_score = concentration / 100.0  # Convert to 0-1 range for database
        
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
            'confidence': float(confidence),
            'concentration': float(concentration),
            'probs': emotion_probs,
            'bbox': box
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
        
        # Calculate image hash to verify uniqueness
        import hashlib
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        
        # Detect faces
        faces = model_service.detect_faces(image)
        logger.info(f"[EmotionStream] Face detection completed. Found {len(faces)} face(s)")
        
        if not faces:
            logger.warning(f"[EmotionStream] No faces detected in image. Returning default values.")
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'timestamp': timestamp,
                'message': 'No faces detected',
                'bbox': None
            })
        
        # Use first detected face
        face = faces[0]
        box = face['bbox']
        
        # Preprocess face
        face_tensor, preprocess_error = model_service.preprocess_face(image, box)
        if face_tensor is None or preprocess_error is not None:
            logger.warning(f"[EmotionStream] Face preprocessing failed: {preprocess_error}")
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'timestamp': timestamp,
                'message': 'Face preprocessing failed',
                'bbox': box
            })
        
        # Predict emotions
        emotion_probs = model_service.predict_emotion_vector(face_tensor)
        emotion_label = max(emotion_probs, key=emotion_probs.get)
        concentration = model_service.compute_concentration(emotion_probs)
        
        confidence = emotion_probs[emotion_label]
        emotion_state = model_service.map_emotion_to_state(emotion_label)
        
        # Log detailed detection info
        logger.info(f"[EmotionStream] Detection result - emotion: {emotion_label}, state: {emotion_state}, confidence: {confidence:.4f}, concentration: {concentration:.2f}")
        logger.info(f"[EmotionStream] Returning bbox: {box}, all emotions: {emotion_probs}")
        
        # Always store emotion data in database for stream endpoint
        try:
            emotion_id = generate_uuid_str()
            execute_query(
                """INSERT INTO emotion_data 
                   (id, session_type, session_id, student_id, emotion, confidence, timestamp, engagement_score)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (emotion_id, session_type, session_id, user_id, emotion_state, confidence, timestamp, concentration / 100.0)
            )
        except Exception as e:
            logger.error(f"Error storing emotion data: {str(e)}")
            # Continue even if storage fails
        
        return jsonify({
            'emotion': emotion_state,
            'confidence': float(confidence),
            'concentration': float(concentration),
            'probs': emotion_probs,
            'bbox': box,
            'timestamp': timestamp
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

@bp.route("/test_model", methods=["GET"])
@jwt_required()
def test_model():
    """Test endpoint to validate model inference"""
    try:
        model_service = current_app.model_service
        if not model_service:
            return jsonify({'error': 'ML models not loaded'}), 503
        
        # Create random test image
        sample = np.random.rand(224, 224, 3).astype(np.uint8) * 255
        box = [0, 0, 224, 224]
        
        # Preprocess face
        face_tensor, _ = model_service.preprocess_face(sample, box)
        if face_tensor is None:
            return jsonify({"error": "preprocess failed"}), 500
        
        # Predict emotions
        probs = model_service.predict_emotion_vector(face_tensor)
        
        return jsonify({
            "probs": probs,
            "message": "Model test successful"
        })
    except Exception as e:
        logger.error(f"Error in model test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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

