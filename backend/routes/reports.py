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
        data = request.get_json() or {}
        student_id = data.get('studentId', current_user['id'])
        
        logger.info(f"Generating report for session_type={session_type}, session_id={session_id}, student_id={student_id}")
        
        # For live sessions, use live_session_logs table instead of emotion_data
        if session_type == 'live':
            results = execute_query(
                """SELECT emotion, confidence, timestamp, engagement_score, concentration_score
                   FROM live_session_logs
                   WHERE live_session_id = %s AND student_id = %s
                   ORDER BY timestamp ASC""",
                (session_id, student_id),
                fetch_all=True
            )
        else:
            # For recorded sessions, use emotion_data table
            results = execute_query(
                """SELECT emotion, confidence, timestamp, engagement_score
                   FROM emotion_data
                   WHERE session_type = %s AND session_id = %s AND student_id = %s
                   ORDER BY timestamp ASC""",
                (session_type, session_id, student_id),
                fetch_all=True
            )
        
        logger.info(f"Found {len(results) if results else 0} emotion data records")
        
        if not results:
            logger.warning(f"No emotion data found for session_type={session_type}, session_id={session_id}, student_id={student_id}")
            return jsonify({
                'error': 'No emotion data found',
                'message': 'No emotion data has been recorded for this session. Please watch the video with emotion monitoring enabled first.',
                'session_type': session_type,
                'session_id': session_id,
                'student_id': student_id
            }), 404
        
        # Calculate statistics
        emotions = [row[0] for row in results]
        engagement_scores = [row[3] for row in results if row[3]]
        timestamps = [row[2] for row in results]
        
        # For live sessions, concentration_score is already in the data (0-100)
        # For recorded sessions, convert engagement_score (0-1) to concentration_score (0-100)
        if session_type == 'live':
            # Live sessions have concentration_score in column 4 (index 4)
            concentration_scores = [float(row[4]) if len(row) > 4 and row[4] is not None else 50.0 for row in results]
        else:
            # Recorded sessions: convert engagement_score to concentration_score
            concentration_scores = [score * 100.0 if score else 50.0 for score in engagement_scores]
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(emotions)
        overall_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.5
        avg_concentration = sum(concentration_scores) / len(concentration_scores) if concentration_scores else 50.0
        
        # Calculate percentages
        focus_pct = (emotion_counts.get('focused', 0) / total) * 100
        boredom_pct = (emotion_counts.get('bored', 0) / total) * 100
        confusion_pct = (emotion_counts.get('confused', 0) / total) * 100
        sleepiness_pct = (emotion_counts.get('sleepy', 0) / total) * 100
        
        # Count engagement drops and analyze concentration drops/recoveries
        drops = 0
        concentration_drops = 0
        prev_eng = 1.0
        prev_conc = 100.0
        consecutive_low = 0
        
        # Concentration drop/recovery analysis
        concentration_events = []  # List of {type: 'drop'|'recovery', timestamp: int, duration: int, start_time: int}
        low_concentration_threshold = 40  # Below this is considered "low concentration"
        high_concentration_threshold = 60  # Above this is considered "focused"
        
        in_low_concentration = False
        low_concentration_start = None
        low_concentration_start_timestamp = None
        
        for i, eng in enumerate(engagement_scores):
            conc = concentration_scores[i] if i < len(concentration_scores) else 50.0
            current_timestamp = timestamps[i] if i < len(timestamps) else 0
            
            # Traditional engagement drop (0.5 threshold)
            if eng < 0.5 and prev_eng >= 0.5:
                drops += 1
            
            # Concentration drop/recovery tracking
            if conc < low_concentration_threshold:
                # Entering low concentration period
                if not in_low_concentration:
                    in_low_concentration = True
                    low_concentration_start = i
                    low_concentration_start_timestamp = current_timestamp
                    consecutive_low = 1
                else:
                    consecutive_low += 1
                    # Count as a drop if we've been low for 10+ consecutive readings
                    if consecutive_low >= 10 and concentration_drops == 0:
                        concentration_drops += 1
            elif conc >= high_concentration_threshold:
                # Recovered to focused state
                if in_low_concentration:
                    # Calculate duration of low concentration period
                    duration_seconds = current_timestamp - low_concentration_start_timestamp if low_concentration_start_timestamp else 0
                    concentration_events.append({
                        'type': 'drop',
                        'start_timestamp': low_concentration_start_timestamp,
                        'end_timestamp': current_timestamp,
                        'duration_seconds': duration_seconds,
                        'start_concentration': concentration_scores[low_concentration_start] if low_concentration_start < len(concentration_scores) else 0,
                        'recovery_concentration': conc
                    })
                    in_low_concentration = False
                    low_concentration_start = None
                    low_concentration_start_timestamp = None
                    consecutive_low = 0
            else:
                # In between thresholds - reset consecutive counter but keep tracking if we were low
                if in_low_concentration:
                    consecutive_low += 1
                else:
                    consecutive_low = 0
            
            prev_eng = eng
            prev_conc = conc
        
        # Handle case where session ends while in low concentration
        if in_low_concentration and low_concentration_start_timestamp is not None:
            last_timestamp = timestamps[-1] if timestamps else 0
            duration_seconds = last_timestamp - low_concentration_start_timestamp
            concentration_events.append({
                'type': 'drop',
                'start_timestamp': low_concentration_start_timestamp,
                'end_timestamp': last_timestamp,
                'duration_seconds': duration_seconds,
                'start_concentration': concentration_scores[low_concentration_start] if low_concentration_start < len(concentration_scores) else 0,
                'recovery_concentration': None  # Never recovered
            })
        
        average_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        # Check if report exists
        existing = execute_query(
            """SELECT id FROM engagement_reports 
               WHERE session_type = %s AND session_id = %s AND student_id = %s""",
            (session_type, session_id, student_id),
            fetch_one=True
        )
        
        report_id = existing[0] if existing else generate_uuid_str()
        
        # Create timeline with concentration data
        timeline_data = []
        for i, r in enumerate(results):
            if session_type == 'live':
                # Live sessions: concentration_score is in column 4
                concentration = float(r[4]) if len(r) > 4 and r[4] is not None else 50.0
            else:
                # Recorded sessions: use calculated concentration
                concentration = concentration_scores[i] if i < len(concentration_scores) else 50.0
            
            timeline_data.append({
                'emotion': r[0],
                'timestamp': r[2],
                'concentration': concentration,
                'engagement_score': float(r[3]) if r[3] else 0.5
            })
        timeline_json = json.dumps(timeline_data)
        
        # Create concentration analysis summary
        total_drop_duration = sum(event['duration_seconds'] for event in concentration_events)
        avg_drop_duration = total_drop_duration / len(concentration_events) if concentration_events else 0
        longest_drop = max(concentration_events, key=lambda x: x['duration_seconds']) if concentration_events else None
        
        concentration_analysis = {
            'total_drops': len(concentration_events),
            'total_drop_duration_seconds': total_drop_duration,
            'average_drop_duration_seconds': avg_drop_duration,
            'longest_drop': longest_drop,
            'events': concentration_events
        }
        
        behavior_summary = f"Engagement: {overall_engagement:.2%}, Concentration: {avg_concentration:.1f}%, Drops: {drops}, Concentration Drops: {concentration_drops}, Total Low Concentration Time: {total_drop_duration}s"
        
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
                'average_concentration': float(avg_concentration),
                'average_emotion': average_emotion,
                'engagement_drops': drops,
                'concentration_drops': concentration_drops,
                'focus_percentage': float(focus_pct),
                'boredom_percentage': float(boredom_pct),
                'confusion_percentage': float(confusion_pct),
                'sleepiness_percentage': float(sleepiness_pct),
                'timeline': timeline_data,
                'concentration_analysis': concentration_analysis
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Generate report error: {str(e)}")
        return jsonify({'error': 'Failed to generate report'}), 500

