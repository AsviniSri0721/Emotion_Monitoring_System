from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.database import execute_query
import logging
import json
import uuid

bp = Blueprint('reports', __name__)
logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/generate/<session_type>/<session_id>', methods=['POST'])
@jwt_required()
def generate_report(session_type, session_id):
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        student_id = data.get('studentId', current_user['id'])
        
        # Get emotion data
        results = execute_query(
            """SELECT emotion, confidence, timestamp, engagement_score
               FROM emotion_data
               WHERE session_type = %s AND session_id = %s AND student_id = %s
               ORDER BY timestamp ASC""",
            (session_type, session_id, student_id),
            fetch_all=True
        )
        
        if not results:
            return jsonify({'error': 'No emotion data found'}), 404
        
        # Calculate statistics
        emotions = [row[0] for row in results]
        engagement_scores = [row[3] for row in results if row[3]]
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(emotions)
        overall_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.5
        
        # Calculate percentages
        focus_pct = (emotion_counts.get('focused', 0) / total) * 100
        boredom_pct = (emotion_counts.get('bored', 0) / total) * 100
        confusion_pct = (emotion_counts.get('confused', 0) / total) * 100
        sleepiness_pct = (emotion_counts.get('sleepy', 0) / total) * 100
        
        # Count engagement drops
        drops = 0
        prev_eng = 1.0
        for eng in engagement_scores:
            if eng < 0.5 and prev_eng >= 0.5:
                drops += 1
            prev_eng = eng
        
        average_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        # Check if report exists
        existing = execute_query(
            """SELECT id FROM engagement_reports 
               WHERE session_type = %s AND session_id = %s AND student_id = %s""",
            (session_type, session_id, student_id),
            fetch_one=True
        )
        
        report_id = existing[0] if existing else generate_uuid_str()
        
        timeline_json = json.dumps([{'emotion': r[0], 'timestamp': r[2]} for r in results])
        behavior_summary = f"Engagement: {overall_engagement:.2%}, Drops: {drops}"
        
        if existing:
            # Update existing report
            execute_query(
                """UPDATE engagement_reports SET
                   overall_engagement = %s,
                   average_emotion = %s,
                   engagement_drops = %s,
                   focus_percentage = %s,
                   boredom_percentage = %s,
                   confusion_percentage = %s,
                   sleepiness_percentage = %s,
                   emotional_timeline = %s,
                   behavior_summary = %s
                   WHERE id = %s""",
                (overall_engagement, average_emotion, drops, focus_pct, boredom_pct,
                 confusion_pct, sleepiness_pct, timeline_json, behavior_summary, report_id)
            )
        else:
            # Insert new report
            execute_query(
                """INSERT INTO engagement_reports 
                   (id, session_type, session_id, student_id, overall_engagement, average_emotion,
                    engagement_drops, focus_percentage, boredom_percentage, confusion_percentage,
                    sleepiness_percentage, emotional_timeline, behavior_summary)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (report_id, session_type, session_id, student_id, overall_engagement, average_emotion,
                 drops, focus_pct, boredom_pct, confusion_pct, sleepiness_pct, timeline_json, behavior_summary)
            )
        
        return jsonify({
            'report': {
                'id': str(report_id),
                'overall_engagement': float(overall_engagement),
                'average_emotion': average_emotion,
                'engagement_drops': drops
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Generate report error: {str(e)}")
        return jsonify({'error': 'Failed to generate report'}), 500

