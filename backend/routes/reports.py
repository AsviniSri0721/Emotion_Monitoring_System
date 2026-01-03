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
        
        # Teachers see reports for all their LIVE sessions only, students see their own LIVE reports only
        # Engagement reports are generated exclusively from live sessions where real-time monitoring is active
        if current_user['role'] == 'teacher':
            reports = execute_query(
                """SELECT er.id, er.session_type, er.session_id, er.student_id,
                          er.overall_engagement, er.average_emotion, er.engagement_drops,
                          er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                          er.sleepiness_percentage, er.generated_at,
                          CONCAT(u.first_name, ' ', u.last_name) as student_name,
                          ls.title as session_title
                   FROM engagement_reports er
                   JOIN users u ON er.student_id = u.id
                   JOIN live_sessions ls ON er.session_type = 'live' AND er.session_id = ls.id
                   WHERE er.session_type = 'live' AND ls.teacher_id = %s
                   ORDER BY er.generated_at DESC
                   LIMIT 50""",
                (current_user['id'],),
                fetch_all=True
            )
        else:
            # Students see their own LIVE reports only
            reports = execute_query(
                """SELECT er.id, er.session_type, er.session_id, er.student_id,
                          er.overall_engagement, er.average_emotion, er.engagement_drops,
                          er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                          er.sleepiness_percentage, er.generated_at,
                          CONCAT(u.first_name, ' ', u.last_name) as student_name,
                          ls.title as session_title,
                          er.emotion_segments
                   FROM engagement_reports er
                   JOIN users u ON er.student_id = u.id
                   JOIN live_sessions ls ON er.session_type = 'live' AND er.session_id = ls.id
                   WHERE er.session_type = 'live' AND er.student_id = %s
                   ORDER BY er.generated_at DESC
                   LIMIT 50""",
                (current_user['id'],),
                fetch_all=True
            )
        
        # Convert to list of dicts
        report_list = []
        for row in reports:
            # Parse emotion_segments JSON if present
            emotion_segments = None
            if len(row) > 14 and row[14]:
                try:
                    import json
                    emotion_segments = json.loads(row[14]) if isinstance(row[14], str) else row[14]
                except:
                    emotion_segments = None
            
            report_dict = {
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
            }
            
            if emotion_segments is not None:
                report_dict['emotion_segments'] = emotion_segments
            
            report_list.append(report_dict)
        
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
        
        # Engagement reports are ONLY available for live sessions with real-time monitoring
        if session_type != 'live':
            logger.warning(f"Report generation attempted for non-live session: {session_type}")
            return jsonify({
                'error': 'Report generation not available',
                'message': 'Engagement reports are available only for live monitored sessions. Recorded videos are excluded to ensure accuracy and ethical data usage.',
                'session_type': session_type,
                'session_id': session_id
            }), 400
        
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
        # Note: Recorded sessions are no longer supported for report generation
        # This code path should not be reached due to the check above, but kept for safety
        else:
            logger.error(f"Unexpected session_type in generate_report: {session_type}")
            return jsonify({
                'error': 'Report generation not available',
                'message': 'Engagement reports are available only for live monitored sessions.',
                'session_type': session_type
            }), 400
        
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
        # Live sessions have concentration_score in column 4 (index 4)
        concentration_scores = [float(row[4]) if len(row) > 4 and row[4] is not None else 50.0 for row in results]
        
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
        
        # Check if report exists and get emotion_segments if available
        existing = execute_query(
            """SELECT id, emotion_segments FROM engagement_reports 
               WHERE session_type = %s AND session_id = %s AND student_id = %s""",
            (session_type, session_id, student_id),
            fetch_one=True
        )
        
        report_id = existing[0] if existing else generate_uuid_str()
        existing_emotion_segments = existing[1] if existing and len(existing) > 1 else None
        
        # Create timeline with concentration data
        timeline_data = []
        for i, r in enumerate(results):
            # Live sessions: concentration_score is in column 4
            concentration = float(r[4]) if len(r) > 4 and r[4] is not None else 50.0
            
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
        
        # Parse emotion_segments if they exist in the database
        emotion_segments = None
        if existing_emotion_segments:
            try:
                emotion_segments = json.loads(existing_emotion_segments) if isinstance(existing_emotion_segments, str) else existing_emotion_segments
            except:
                emotion_segments = None
        
        report_data = {
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
        
        if emotion_segments:
            report_data['emotion_segments'] = emotion_segments
        
        return jsonify({
            'report': report_data
        }), 201
        
    except Exception as e:
        logger.error(f"Generate report error: {str(e)}")
        return jsonify({'error': 'Failed to generate report'}), 500

@bp.route('/session/<session_type>/<session_id>', methods=['GET'])
@jwt_required()
def get_report_by_session(session_type, session_id):
    """Get engagement report from engagement_reports table by session type and session ID"""
    try:
        current_user = get_current_user()
        
        # Get session start time for calculating actual timestamps
        session_start = None
        if session_type == 'live':
            try:
                session_info = execute_query(
                    """SELECT ls.started_at, ls.created_at
                       FROM live_sessions ls
                       WHERE ls.id = %s""",
                    (session_id,),
                    fetch_one=True
                )
                if session_info:
                    # Use started_at if available, otherwise use created_at
                    session_start = session_info[0] if session_info[0] else (session_info[1] if len(session_info) > 1 else None)
            except Exception as e:
                logger.warning(f"Could not fetch session start time: {str(e)}")
                session_start = None
        
        # For students, get their own report. For teachers, get the first report for that session
        # Note: emotion_segments column may not exist in all database schemas, so we don't select it
        if current_user['role'] == 'student':
            result = execute_query(
                """SELECT er.id, er.session_type, er.session_id, er.student_id,
                          er.overall_engagement, er.average_emotion, er.engagement_drops,
                          er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                          er.sleepiness_percentage, er.emotional_timeline, er.behavior_summary,
                          er.generated_at
                   FROM engagement_reports er
                   WHERE er.session_type = %s AND er.session_id = %s AND er.student_id = %s
                   LIMIT 1""",
                (session_type, session_id, current_user['id']),
                fetch_one=True
            )
        else:
            # Teacher - get first report for this session (or specific student if provided)
            student_id = request.args.get('studentId')
            if student_id:
                result = execute_query(
                    """SELECT er.id, er.session_type, er.session_id, er.student_id,
                              er.overall_engagement, er.average_emotion, er.engagement_drops,
                              er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                              er.sleepiness_percentage, er.emotional_timeline, er.behavior_summary,
                              er.generated_at
                       FROM engagement_reports er
                       WHERE er.session_type = %s AND er.session_id = %s AND er.student_id = %s
                       LIMIT 1""",
                    (session_type, session_id, student_id),
                    fetch_one=True
                )
            else:
                result = execute_query(
                    """SELECT er.id, er.session_type, er.session_id, er.student_id,
                              er.overall_engagement, er.average_emotion, er.engagement_drops,
                              er.focus_percentage, er.boredom_percentage, er.confusion_percentage,
                              er.sleepiness_percentage, er.emotional_timeline, er.behavior_summary,
                              er.generated_at
                       FROM engagement_reports er
                       WHERE er.session_type = %s AND er.session_id = %s
                       LIMIT 1""",
                    (session_type, session_id),
                    fetch_one=True
                )
        
        if not result:
            logger.warning(f"Report not found for session_type={session_type}, session_id={session_id}, user={current_user['id']}")
            return jsonify({'error': 'Report not found'}), 404
        
        # Parse JSON fields - handle case where result might have fewer fields
        import json
        from datetime import datetime, timedelta
        timeline_data = []
        if len(result) > 11 and result[11]:  # emotional_timeline at index 11
            try:
                timeline_data = json.loads(result[11]) if isinstance(result[11], str) else result[11]
                
                # Convert relative timestamps to actual timestamps if session_start is available
                if session_start and timeline_data:
                    try:
                        # Convert session_start to datetime if needed
                        if isinstance(session_start, str):
                            try:
                                session_start_dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
                            except:
                                # Try parsing MySQL datetime format
                                session_start_dt = datetime.strptime(session_start, '%Y-%m-%d %H:%M:%S')
                        elif hasattr(session_start, 'isoformat'):
                            session_start_dt = session_start  # Already a datetime object
                        else:
                            session_start_dt = None
                        
                        if session_start_dt:
                            for point in timeline_data:
                                if isinstance(point, dict) and 'timestamp' in point:
                                    # timestamp is seconds from session start
                                    seconds_offset = int(point.get('timestamp', 0))
                                    actual_time = session_start_dt + timedelta(seconds=seconds_offset)
                                    point['actual_time'] = actual_time.isoformat()
                                    point['time_display'] = actual_time.strftime('%H:%M:%S')
                    except Exception as e:
                        logger.warning(f"Error converting timestamps to actual time: {str(e)}")
            except Exception as e:
                logger.warning(f"Error parsing timeline: {str(e)}")
                timeline_data = []
        
        # emotion_segments column may not exist in database, so we set it to None
        emotion_segments = None
        
        # Calculate average concentration from timeline if available
        avg_concentration = 50.0
        if timeline_data:
            try:
                concentrations = [point.get('concentration', 50.0) for point in timeline_data if isinstance(point, dict) and 'concentration' in point]
                if concentrations:
                    avg_concentration = sum(concentrations) / len(concentrations)
            except Exception as e:
                logger.warning(f"Error calculating avg concentration: {str(e)}")
                avg_concentration = 50.0
        
        # Safely access result fields
        # Result structure: [id(0), session_type(1), session_id(2), student_id(3), 
        #                   overall_engagement(4), average_emotion(5), engagement_drops(6),
        #                   focus_percentage(7), boredom_percentage(8), confusion_percentage(9),
        #                   sleepiness_percentage(10), emotional_timeline(11), behavior_summary(12),
        #                   generated_at(13)]
        report_data = {
            'id': result[0] if len(result) > 0 else '',
            'session_type': result[1] if len(result) > 1 else session_type,
            'session_id': result[2] if len(result) > 2 else session_id,
            'student_id': result[3] if len(result) > 3 else '',
            'overall_engagement': float(result[4]) if len(result) > 4 and result[4] is not None else 0.0,
            'average_emotion': result[5] if len(result) > 5 and result[5] else 'neutral',
            'engagement_drops': int(result[6]) if len(result) > 6 and result[6] is not None else 0,
            'focus_percentage': float(result[7]) if len(result) > 7 and result[7] is not None else 0.0,
            'boredom_percentage': float(result[8]) if len(result) > 8 and result[8] is not None else 0.0,
            'confusion_percentage': float(result[9]) if len(result) > 9 and result[9] is not None else 0.0,
            'sleepiness_percentage': float(result[10]) if len(result) > 10 and result[10] is not None else 0.0,
            'average_concentration': avg_concentration,
            'timeline': timeline_data,
            'emotion_segments': emotion_segments,  # Will be None if column doesn't exist
            'generated_at': result[13].isoformat() if len(result) > 13 and result[13] else None
        }
        
        return jsonify({'report': report_data}), 200
        
    except Exception as e:
        logger.error(f"Get report error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch report', 'details': str(e)}), 500

@bp.route('/<report_id>', methods=['DELETE'])
@jwt_required()
def delete_report(report_id):
    """Delete an engagement report"""
    try:
        current_user = get_current_user()
        
        # Check if report exists and user has permission
        result = execute_query(
            """SELECT er.id, er.student_id, er.session_type, er.session_id
               FROM engagement_reports er
               WHERE er.id = %s""",
            (report_id,),
            fetch_one=True
        )
        
        if not result:
            return jsonify({'error': 'Report not found'}), 404
        
        # Students can only delete their own reports, teachers can delete any report
        if current_user['role'] == 'student' and result[1] != current_user['id']:
            return jsonify({'error': 'Unauthorized - You can only delete your own reports'}), 403
        
        # Delete the report
        execute_query(
            """DELETE FROM engagement_reports WHERE id = %s""",
            (report_id,)
        )
        
        logger.info(f"Report {report_id} deleted by user {current_user['id']}")
        
        return jsonify({'message': 'Report deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete report error: {str(e)}")
        return jsonify({'error': 'Failed to delete report'}), 500

