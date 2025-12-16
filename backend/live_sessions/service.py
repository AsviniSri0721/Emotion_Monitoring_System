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
    
    @staticmethod
    def generate_engagement_reports(session_id: str) -> Dict[str, Any]:
        """
        Generate engagement reports for all students in a live session.
        Called automatically when teacher ends a session.
        Aggregates data from live_session_logs and inserts into engagement_reports.
        """
        from services.database import execute_query
        import json
        import uuid
        
        logger.info(f"Generating engagement reports for session {session_id}")
        
        # Get all logs for this session, grouped by student
        all_logs = LiveSessionLog.get_by_session(session_id)
        
        if not all_logs:
            logger.warning(f"No logs found for session {session_id}, skipping report generation")
            return {
                'message': 'No emotion data found for this session',
                'reports_generated': 0
            }
        
        # Group logs by student
        student_logs = {}
        for log in all_logs:
            student_id = log['student_id']
            if student_id not in student_logs:
                student_logs[student_id] = []
            student_logs[student_id].append(log)
        
        reports_generated = 0
        
        # Generate report for each student
        for student_id, logs in student_logs.items():
            try:
                # Extract data from logs
                emotions = [log['emotion'] for log in logs]
                engagement_scores = [log['engagement_score'] for log in logs if log['engagement_score'] is not None]
                concentration_scores = [log['concentration_score'] for log in logs if log['concentration_score'] is not None]
                
                # Calculate averages
                overall_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.5
                avg_concentration = sum(concentration_scores) / len(concentration_scores) if concentration_scores else 50.0
                
                # Count emotions
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # Get dominant emotion
                dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
                
                # Calculate percentages (based on emotion counts)
                # Note: map_emotion_to_state returns states like 'focused', 'bored', 'confused', 'sleepy', 'neutral', 'frustrated'
                # Note: 'sleepy' emotion maps to 'bored' state, but we need to check the original emotion
                # For now, we'll use the mapped states as stored in logs
                total = len(emotions)
                focus_pct = (emotion_counts.get('focused', 0) / total) * 100 if total > 0 else 0.0
                boredom_pct = (emotion_counts.get('bored', 0) / total) * 100 if total > 0 else 0.0
                confusion_pct = (emotion_counts.get('confused', 0) / total) * 100 if total > 0 else 0.0
                # Note: 'sleepy' is mapped to 'bored' in map_emotion_to_state, so we can't distinguish them
                # We'll set sleepiness to 0 for now, or use a portion of 'bored' if needed
                # For accuracy, we'd need to store the original emotion label, but for now this is acceptable
                sleepiness_pct = 0.0  # Cannot be calculated from mapped states alone
                
                # Count engagement drops (concentration < 40)
                drops = 0
                prev_conc = 100.0
                for conc in concentration_scores:
                    if conc < 40 and prev_conc >= 40:
                        drops += 1
                    prev_conc = conc
                
                # Sort logs by timestamp to ensure chronological order
                sorted_logs = sorted(logs, key=lambda x: x['timestamp'])
                
                # Create timeline
                timeline_data = []
                for log in sorted_logs:
                    timeline_data.append({
                        'emotion': log['emotion'],
                        'timestamp': log['timestamp'],
                        'concentration': log['concentration_score'] if log['concentration_score'] is not None else 50.0,
                        'engagement_score': log['engagement_score'] if log['engagement_score'] is not None else 0.5
                    })
                timeline_json = json.dumps(timeline_data)
                
                # Build emotion segments (temporal aggregation)
                # Group consecutive identical emotions into segments with start, end, and duration
                def format_timestamp(ts):
                    """Format timestamp (seconds) as HH:MM:SS"""
                    hours = ts // 3600
                    minutes = (ts % 3600) // 60
                    seconds = ts % 60
                    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                emotion_segments = []
                if sorted_logs:
                    current_emotion = sorted_logs[0]['emotion']
                    segment_start_timestamp = sorted_logs[0]['timestamp']
                    
                    for i in range(1, len(sorted_logs)):
                        if sorted_logs[i]['emotion'] != current_emotion:
                            # Emotion changed - end current segment and start new one
                            segment_end_timestamp = sorted_logs[i-1]['timestamp']
                            duration_seconds = segment_end_timestamp - segment_start_timestamp
                            
                            emotion_segments.append({
                                'emotion': current_emotion,
                                'start_timestamp': segment_start_timestamp,
                                'end_timestamp': segment_end_timestamp,
                                'start_time': format_timestamp(segment_start_timestamp),
                                'end_time': format_timestamp(segment_end_timestamp),
                                'duration_seconds': int(duration_seconds)
                            })
                            
                            # Start new segment
                            current_emotion = sorted_logs[i]['emotion']
                            segment_start_timestamp = sorted_logs[i]['timestamp']
                    
                    # Add final segment
                    segment_end_timestamp = sorted_logs[-1]['timestamp']
                    duration_seconds = segment_end_timestamp - segment_start_timestamp
                    
                    emotion_segments.append({
                        'emotion': current_emotion,
                        'start_timestamp': segment_start_timestamp,
                        'end_timestamp': segment_end_timestamp,
                        'start_time': format_timestamp(segment_start_timestamp),
                        'end_time': format_timestamp(segment_end_timestamp),
                        'duration_seconds': int(duration_seconds)
                    })
                
                emotion_segments_json = json.dumps(emotion_segments)
                
                # Create behavior summary
                behavior_summary = f"Engagement: {overall_engagement:.2%}, Concentration: {avg_concentration:.1f}%, Drops: {drops}"
                
                # Check if report already exists
                existing = execute_query(
                    """SELECT id FROM engagement_reports 
                       WHERE session_type = 'live' AND session_id = %s AND student_id = %s""",
                    (session_id, student_id),
                    fetch_one=True
                )
                
                report_id = existing[0] if existing else str(uuid.uuid4())
                
                # Check if emotion_segments column exists, if not we'll handle it gracefully
                try:
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
                               behavior_summary = %s,
                               emotion_segments = %s,
                               generated_at = NOW()
                               WHERE id = %s""",
                            (overall_engagement, dominant_emotion, drops, focus_pct, boredom_pct,
                             confusion_pct, sleepiness_pct, timeline_json, behavior_summary, emotion_segments_json, report_id)
                        )
                        logger.info(f"Updated engagement report for student {student_id} in session {session_id}")
                    else:
                        # Insert new report
                        execute_query(
                            """INSERT INTO engagement_reports 
                               (id, session_type, session_id, student_id, overall_engagement, average_emotion,
                                engagement_drops, focus_percentage, boredom_percentage, confusion_percentage,
                                sleepiness_percentage, emotional_timeline, behavior_summary, emotion_segments, generated_at)
                               VALUES (%s, 'live', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                            (report_id, session_id, student_id, overall_engagement, dominant_emotion,
                             drops, focus_pct, boredom_pct, confusion_pct, sleepiness_pct, timeline_json, behavior_summary, emotion_segments_json)
                        )
                        logger.info(f"Created engagement report for student {student_id} in session {session_id}")
                except Exception as col_error:
                    # If emotion_segments column doesn't exist, try without it
                    logger.warning(f"emotion_segments column may not exist, attempting without it: {str(col_error)}")
                    if existing:
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
                               behavior_summary = %s,
                               generated_at = NOW()
                               WHERE id = %s""",
                            (overall_engagement, dominant_emotion, drops, focus_pct, boredom_pct,
                             confusion_pct, sleepiness_pct, timeline_json, behavior_summary, report_id)
                        )
                    else:
                        execute_query(
                            """INSERT INTO engagement_reports 
                               (id, session_type, session_id, student_id, overall_engagement, average_emotion,
                                engagement_drops, focus_percentage, boredom_percentage, confusion_percentage,
                                sleepiness_percentage, emotional_timeline, behavior_summary, generated_at)
                               VALUES (%s, 'live', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                            (report_id, session_id, student_id, overall_engagement, dominant_emotion,
                             drops, focus_pct, boredom_pct, confusion_pct, sleepiness_pct, timeline_json, behavior_summary)
                        )
                    logger.warning(f"Report saved without emotion_segments. Please run migration to add the column.")
                
                reports_generated += 1
                
            except Exception as e:
                logger.error(f"Error generating report for student {student_id} in session {session_id}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue with other students even if one fails
                continue
        
        logger.info(f"Generated {reports_generated} engagement reports for session {session_id}")
        return {
            'message': f'Generated {reports_generated} engagement report(s)',
            'reports_generated': reports_generated
        }

