from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from utils.jwt_helpers import get_current_user
from services.database import execute_query
import logging
import json
import uuid

bp = Blueprint('reports', __name__)
logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/dashboard/all', methods=['GET'])
@jwt_required()
def get_all_reports():
    """GET endpoint to fetch all engagement reports for the dashboard"""
    try:
        # get_jwt_identity() should work here because @jwt_required() already validated the token
        current_user = get_current_user()
        logger.info(f"Token validated for reports: {current_user}")
        
        if not current_user or 'id' not in current_user or 'role' not in current_user:
            logger.error(f"Invalid user data from token: {current_user}")
            return jsonify({'error': 'Invalid token payload'}), 422
        
        # Teachers see reports for all their sessions, students see their own reports
        if current_user['role'] == 'teacher':
            reports = execute_query(
                """SELECT er.id, er.session_type, er.session_id, er.student_id,
                          er.overall_engagement, er.average_emotion, er.engagement_drops,
                          er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                          er.sleepiness_percentage, er.generated_at,
                          CONCAT(u.first_name, ' ', u.last_name) as student_name,
                          CASE 
                              WHEN er.session_type = 'live' THEN ls.title
                              WHEN er.session_type = 'recorded' THEN v.title
                              ELSE 'Unknown'
                          END as session_title
                   FROM engagement_reports er
                   JOIN users u ON er.student_id = u.id
                   LEFT JOIN live_sessions ls ON er.session_type = 'live' AND er.session_id = ls.id
                   LEFT JOIN videos v ON er.session_type = 'recorded' AND er.session_id = v.id
                   WHERE (er.session_type = 'live' AND ls.teacher_id = %s)
                      OR (er.session_type = 'recorded' AND v.teacher_id = %s)
                   ORDER BY er.generated_at DESC
                   LIMIT 50""",
                (current_user['id'], current_user['id']),
                fetch_all=True
            )
        else:
            # Students see their own reports
            reports = execute_query(
                """SELECT er.id, er.session_type, er.session_id, er.student_id,
                          er.overall_engagement, er.average_emotion, er.engagement_drops,
                          er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                          er.sleepiness_percentage, er.generated_at,
                          CONCAT(u.first_name, ' ', u.last_name) as student_name,
                          CASE 
                              WHEN er.session_type = 'live' THEN ls.title
                              WHEN er.session_type = 'recorded' THEN v.title
                              ELSE 'Unknown'
                          END as session_title
                   FROM engagement_reports er
                   JOIN users u ON er.student_id = u.id
                   LEFT JOIN live_sessions ls ON er.session_type = 'live' AND er.session_id = ls.id
                   LEFT JOIN videos v ON er.session_type = 'recorded' AND er.session_id = v.id
                   WHERE er.student_id = %s
                   ORDER BY er.generated_at DESC
                   LIMIT 50""",
                (current_user['id'],),
                fetch_all=True
            )
        
        # Convert to list of dicts
        report_list = []
        for row in reports:
            report_list.append({
                'id': row[0],
                'session_type': row[1],
                'session_id': row[2],
                'student_id': row[3],
                'overall_engagement': float(row[4]) if row[4] else 0.0,
                'average_emotion': row[5],
                'engagement_drops': row[6],
                'focus_percentage': float(row[7]) if row[7] else 0.0,
                'boredom_percentage': float(row[8]) if row[8] else 0.0,
                'confusion_percentage': float(row[9]) if row[9] else 0.0,
                'sleepiness_percentage': float(row[10]) if row[10] else 0.0,
                'generated_at': row[11].isoformat() if row[11] else None,
                'student_name': row[12],
                'session_title': row[13]
            })
        
        return jsonify({'reports': report_list}), 200
        
    except Exception as e:
        logger.error(f"Get reports error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch reports', 'details': str(e)}), 500

@bp.route('/generate/<session_type>/<session_id>', methods=['POST'])
@jwt_required()
def generate_report(session_type, session_id):
    try:
        current_user = get_current_user()
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

