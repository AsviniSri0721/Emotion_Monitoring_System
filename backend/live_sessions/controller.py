"""
HTTP request handlers for live sessions
"""
import logging
from flask import request, jsonify, current_app
from flask_jwt_extended import jwt_required
from utils.jwt_helpers import get_current_user
from live_sessions.service import LiveSessionService
from datetime import datetime

logger = logging.getLogger(__name__)

def create_live_session():
    """POST /live-sessions/create - Create a new live session"""
    try:
        current_user = get_current_user()
        if current_user['role'] != 'teacher':
            return jsonify({'error': 'Unauthorized - Teachers only'}), 403
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        teacher_id = current_user['id']
        title = data.get('title')
        meet_url = data.get('meetUrl')
        start_time_str = data.get('startTime')
        
        if not title or not meet_url:
            return jsonify({'error': 'title and meetUrl are required'}), 400
        
        # Parse start_time if provided
        start_time = None
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            except:
                start_time = None
        
        result = LiveSessionService.create_session(teacher_id, title, meet_url, start_time)
        
        return jsonify({
            'session': {'id': result['id']},
            'message': result['message']
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating live session: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_available_sessions():
    """GET /live-sessions/available - Get all available live sessions"""
    try:
        sessions = LiveSessionService.get_available_sessions()
        return jsonify({'sessions': sessions}), 200
        
    except Exception as e:
        logger.error(f"Error fetching available sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500

def stream_emotion(session_id: str):
    """POST /live-sessions/:id/stream - Stream emotion detection for live session"""
    try:
        current_user = get_current_user()
        student_id = current_user['id']
        
        # Get image data
        data = request.get_json() if request.is_json else {}
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Process emotion detection (service will check for model service)
        result = LiveSessionService.stream_emotion(session_id, student_id, data)
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error in stream: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in live session stream: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def get_session_report(session_id: str):
    """GET /live-sessions/:id/report - Get analytics report for a live session"""
    try:
        current_user = get_current_user()
        
        # Verify session exists and user has access
        session = LiveSessionService.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Teachers can see all students, students can only see their own
        if current_user['role'] == 'student':
            # For students, filter to their own data
            from live_sessions.models import LiveSessionLog
            student_logs = LiveSessionLog.get_by_session(session_id, current_user['id'])
            
            if not student_logs:
                return jsonify({
                    'session': session,
                    'student_data': {
                        'total_logs': 0,
                        'avg_engagement': 0.0,
                        'avg_concentration': 0.0,
                        'dominant_emotion': 'neutral',
                        'timeline': []
                    }
                }), 200
            
            # Calculate student stats
            emotions = [log['emotion'] for log in student_logs]
            engagement_scores = [log['engagement_score'] for log in student_logs if log['engagement_score']]
            concentration_scores = [log['concentration_score'] for log in student_logs if log['concentration_score']]
            
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
            
            timeline = []
            for log in student_logs:
                timeline.append({
                    'timestamp': log['timestamp'],
                    'emotion': log['emotion'],
                    'confidence': log['confidence'],
                    'engagement_score': log['engagement_score'],
                    'concentration_score': log['concentration_score']
                })
            
            return jsonify({
                'session': session,
                'student_data': {
                    'total_logs': len(student_logs),
                    'avg_engagement': sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0,
                    'avg_concentration': sum(concentration_scores) / len(concentration_scores) if concentration_scores else 0.0,
                    'dominant_emotion': dominant_emotion,
                    'emotion_counts': emotion_counts,
                    'timeline': timeline
                }
            }), 200
        else:
            # Teacher sees full report
            report = LiveSessionService.get_session_report(session_id)
            return jsonify({'report': report}), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

