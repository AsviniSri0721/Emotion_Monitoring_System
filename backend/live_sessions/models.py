"""
Database models for live sessions
"""
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from services.database import execute_query
import logging

logger = logging.getLogger(__name__)

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

class LiveSession:
    """Model for live session records"""
    
    @staticmethod
    def create(teacher_id: str, title: str, meet_url: str, start_time: Optional[datetime] = None) -> str:
        """Create a new live session"""
        session_id = generate_uuid_str()
        start_time_str = start_time.isoformat() if start_time else datetime.now().isoformat()
        
        execute_query(
            """INSERT INTO live_sessions (id, teacher_id, title, meet_url, scheduled_at, status)
               VALUES (%s, %s, %s, %s, %s, 'scheduled')""",
            (session_id, teacher_id, title, meet_url, start_time_str)
        )
        
        return session_id
    
    @staticmethod
    def get_by_id(session_id: str) -> Optional[Dict[str, Any]]:
        """Get live session by ID"""
        result = execute_query(
            """SELECT ls.id, ls.teacher_id, ls.title, ls.meet_url, ls.scheduled_at, 
                      ls.started_at, ls.ended_at, ls.status, ls.created_at,
                      CONCAT(u.first_name, ' ', u.last_name) as teacher_name
               FROM live_sessions ls
               JOIN users u ON ls.teacher_id = u.id
               WHERE ls.id = %s""",
            (session_id,),
            fetch_one=True
        )
        
        if not result:
            return None
        
        return {
            'id': result[0],
            'teacher_id': result[1],
            'title': result[2],
            'meet_url': result[3],
            'scheduled_at': result[4].isoformat() if result[4] else None,
            'started_at': result[5].isoformat() if result[5] else None,
            'ended_at': result[6].isoformat() if result[6] else None,
            'status': result[7] or 'scheduled',
            'created_at': result[8].isoformat() if result[8] else None,
            'teacher_name': result[9]
        }
    
    @staticmethod
    def get_available() -> list:
        """Get all available live sessions"""
        results = execute_query(
            """SELECT ls.id, ls.teacher_id, ls.title, ls.meet_url, ls.scheduled_at, 
                      ls.status, ls.created_at,
                      CONCAT(u.first_name, ' ', u.last_name) as teacher_name
               FROM live_sessions ls
               JOIN users u ON ls.teacher_id = u.id
               WHERE ls.status IN ('scheduled', 'live')
               ORDER BY ls.scheduled_at DESC""",
            None,
            fetch_all=True
        )
        
        sessions = []
        for row in results:
            sessions.append({
                'id': row[0],
                'teacher_id': row[1],
                'title': row[2],
                'meet_url': row[3],
                'scheduled_at': row[4].isoformat() if row[4] else None,
                'status': row[5] or 'scheduled',
                'created_at': row[6].isoformat() if row[6] else None,
                'teacher_name': row[7]
            })
        
        return sessions
    
    @staticmethod
    def get_by_teacher(teacher_id: str) -> list:
        """Get all live sessions for a teacher"""
        results = execute_query(
            """SELECT ls.id, ls.teacher_id, ls.title, ls.meet_url, ls.scheduled_at, 
                      ls.started_at, ls.ended_at, ls.status, ls.created_at
               FROM live_sessions ls
               WHERE ls.teacher_id = %s
               ORDER BY ls.scheduled_at DESC""",
            (teacher_id,),
            fetch_all=True
        )
        
        sessions = []
        for row in results:
            sessions.append({
                'id': row[0],
                'teacher_id': row[1],
                'title': row[2],
                'meet_url': row[3],
                'scheduled_at': row[4].isoformat() if row[4] else None,
                'started_at': row[5].isoformat() if row[5] else None,
                'ended_at': row[6].isoformat() if row[6] else None,
                'status': row[7] or 'scheduled',
                'created_at': row[8].isoformat() if row[8] else None
            })
        
        return sessions
    
    @staticmethod
    def update_status(session_id: str, status: str):
        """Update live session status"""
        if status == 'live':
            execute_query(
                """UPDATE live_sessions 
                   SET status = 'live', started_at = NOW()
                   WHERE id = %s""",
                (session_id,)
            )
        elif status == 'ended':
            execute_query(
                """UPDATE live_sessions 
                   SET status = 'ended', ended_at = NOW()
                   WHERE id = %s""",
                (session_id,)
            )

class LiveSessionLog:
    """Model for live session emotion logs"""
    
    @staticmethod
    def create(session_id: str, student_id: str, emotion: str, confidence: float,
               engagement_score: float, concentration_score: float, timestamp: int) -> str:
        """Create a new live session log entry"""
        log_id = generate_uuid_str()
        
        execute_query(
            """INSERT INTO live_session_logs 
               (id, live_session_id, student_id, emotion, confidence, engagement_score, 
                concentration_score, timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (log_id, session_id, student_id, emotion, confidence, engagement_score, 
             concentration_score, timestamp)
        )
        
        return log_id
    
    @staticmethod
    def get_by_session(session_id: str, student_id: Optional[str] = None) -> list:
        """Get all logs for a live session"""
        if student_id:
            results = execute_query(
                """SELECT id, live_session_id, student_id, emotion, confidence, 
                          engagement_score, concentration_score, timestamp, created_at
                   FROM live_session_logs
                   WHERE live_session_id = %s AND student_id = %s
                   ORDER BY timestamp ASC""",
                (session_id, student_id),
                fetch_all=True
            )
        else:
            results = execute_query(
                """SELECT id, live_session_id, student_id, emotion, confidence, 
                          engagement_score, concentration_score, timestamp, created_at
                   FROM live_session_logs
                   WHERE live_session_id = %s
                   ORDER BY timestamp ASC""",
                (session_id,),
                fetch_all=True
            )
        
        logs = []
        for row in results:
            logs.append({
                'id': row[0],
                'live_session_id': row[1],
                'student_id': row[2],
                'emotion': row[3],
                'confidence': float(row[4]),
                'engagement_score': float(row[5]) if row[5] else None,
                'concentration_score': float(row[6]) if row[6] else None,
                'timestamp': row[7],
                'created_at': row[8].isoformat() if row[8] else None
            })
        
        return logs
    
    @staticmethod
    def get_student_stats(session_id: str, student_id: str) -> Dict[str, Any]:
        """Get statistics for a student in a live session"""
        logs = LiveSessionLog.get_by_session(session_id, student_id)
        
        if not logs:
            return {
                'total_logs': 0,
                'avg_engagement': 0.0,
                'avg_concentration': 0.0,
                'dominant_emotion': 'neutral',
                'emotion_counts': {}
            }
        
        emotions = [log['emotion'] for log in logs]
        engagement_scores = [log['engagement_score'] for log in logs if log['engagement_score']]
        concentration_scores = [log['concentration_score'] for log in logs if log['concentration_score']]
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        return {
            'total_logs': len(logs),
            'avg_engagement': sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0,
            'avg_concentration': sum(concentration_scores) / len(concentration_scores) if concentration_scores else 0.0,
            'dominant_emotion': dominant_emotion,
            'emotion_counts': emotion_counts
        }










