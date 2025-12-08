"""
Business logic for live sessions
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from flask import current_app
import cv2
import numpy as np
import base64
from live_sessions.models import LiveSession, LiveSessionLog

logger = logging.getLogger(__name__)

class LiveSessionService:
    """Service for live session operations"""
    
    @staticmethod
    def create_session(teacher_id: str, title: str, meet_url: str, start_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Create a new live session"""
        session_id = LiveSession.create(teacher_id, title, meet_url, start_time)
        return {
            'id': session_id,
            'message': 'Live session created successfully'
        }
    
    @staticmethod
    def get_available_sessions() -> List[Dict[str, Any]]:
        """Get all available live sessions"""
        return LiveSession.get_available()
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get a live session by ID"""
        return LiveSession.get_by_id(session_id)
    
    @staticmethod
    def stream_emotion(session_id: str, student_id: str, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process emotion detection for live session stream
        Reuses ModelService but WITHOUT intervention logic
        """
        # Get model service from app context
        model_service = current_app.model_service
        if not model_service:
            raise Exception('ML models not loaded')
        
        # Parse image
        image = None
        if 'image' in image_data:
            # Base64 encoded image
            image_str = image_data['image']
            image_data_str = image_str.split(',')[1] if ',' in image_str else image_str
            image_bytes = base64.b64decode(image_data_str)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise ValueError('No image provided')
        
        if image is None:
            raise ValueError('Invalid image data')
        
        # Get timestamp (seconds from session start)
        timestamp = image_data.get('timestamp', 0)
        
        # Detect faces
        faces = model_service.detect_faces(image)
        
        if not faces:
            # Return default values but don't log
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'timestamp': timestamp,
                'message': 'No faces detected',
                'bbox': None
            }
        
        # Use first detected face
        face = faces[0]
        box = face['bbox']
        
        # Preprocess face
        face_tensor, preprocess_error = model_service.preprocess_face(image, box)
        if face_tensor is None or preprocess_error is not None:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'timestamp': timestamp,
                'message': 'Face preprocessing failed',
                'bbox': box
            }
        
        # Predict emotions
        emotion_probs = model_service.predict_emotion_vector(face_tensor)
        emotion_label = max(emotion_probs, key=emotion_probs.get)
        concentration = model_service.compute_concentration(emotion_probs)
        
        confidence = emotion_probs[emotion_label]
        emotion_state = model_service.map_emotion_to_state(emotion_label)
        engagement_score = concentration / 100.0  # Convert to 0-1 range
        
        # Save to live_session_logs table (NOT emotion_data)
        try:
            LiveSessionLog.create(
                session_id=session_id,
                student_id=student_id,
                emotion=emotion_state,
                confidence=float(confidence),
                engagement_score=engagement_score,
                concentration_score=float(concentration),
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error saving live session log: {str(e)}")
            # Continue even if save fails
        
        return {
            'emotion': emotion_state,
            'confidence': float(confidence),
            'concentration_score': float(concentration),
            'engagement_score': engagement_score,
            'probs': emotion_probs,
            'bbox': box,
            'timestamp': timestamp
        }
    
    @staticmethod
    def get_session_report(session_id: str) -> Dict[str, Any]:
        """Generate analytics report for a live session"""
        session = LiveSession.get_by_id(session_id)
        if not session:
            raise ValueError('Session not found')
        
        # Get all logs for this session
        all_logs = LiveSessionLog.get_by_session(session_id)
        
        if not all_logs:
            return {
                'session': session,
                'total_students': 0,
                'total_logs': 0,
                'students': []
            }
        
        # Group by student
        student_logs = {}
        for log in all_logs:
            student_id = log['student_id']
            if student_id not in student_logs:
                student_logs[student_id] = []
            student_logs[student_id].append(log)
        
        # Get student names
        from services.database import execute_query
        
        students_data = []
        for student_id, logs in student_logs.items():
            # Get student name
            student_result = execute_query(
                """SELECT CONCAT(first_name, ' ', last_name) as name, email
                   FROM users WHERE id = %s""",
                (student_id,),
                fetch_one=True
            )
            
            student_name = student_result[0] if student_result else 'Unknown'
            student_email = student_result[1] if student_result else ''
            
            # Calculate stats
            emotions = [log['emotion'] for log in logs]
            engagement_scores = [log['engagement_score'] for log in logs if log['engagement_score']]
            concentration_scores = [log['concentration_score'] for log in logs if log['concentration_score']]
            
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
            
            # Create timeline
            timeline = []
            for log in logs:
                timeline.append({
                    'timestamp': log['timestamp'],
                    'emotion': log['emotion'],
                    'confidence': log['confidence'],
                    'engagement_score': log['engagement_score'],
                    'concentration_score': log['concentration_score']
                })
            
            students_data.append({
                'student_id': student_id,
                'student_name': student_name,
                'student_email': student_email,
                'total_logs': len(logs),
                'avg_engagement': sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0,
                'avg_concentration': sum(concentration_scores) / len(concentration_scores) if concentration_scores else 0.0,
                'dominant_emotion': dominant_emotion,
                'emotion_counts': emotion_counts,
                'timeline': timeline
            })
        
        # Calculate overall stats
        all_engagement = [log['engagement_score'] for log in all_logs if log['engagement_score']]
        all_concentration = [log['concentration_score'] for log in all_logs if log['concentration_score']]
        
        return {
            'session': session,
            'total_students': len(students_data),
            'total_logs': len(all_logs),
            'overall_avg_engagement': sum(all_engagement) / len(all_engagement) if all_engagement else 0.0,
            'overall_avg_concentration': sum(all_concentration) / len(all_concentration) if all_concentration else 0.0,
            'students': students_data
        }

