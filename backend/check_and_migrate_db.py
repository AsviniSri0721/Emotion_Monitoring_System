"""
Database Migration and Verification Script
Checks if all required columns exist and adds them if missing.
Run this script to ensure your database is up to date.
"""

import os
import sys
from dotenv import load_dotenv
from services.database import get_connection, execute_query
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    try:
        conn = get_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        # Get database name from environment
        db_name = os.getenv('DB_NAME', 'emotiondb')
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s 
              AND COLUMN_NAME = %s
        """, (db_name, table_name, column_name))
        
        result = cursor.fetchone()
        cursor.close()
        return result[0] > 0 if result else False
    except Exception as e:
        logger.error(f"Error checking column {table_name}.{column_name}: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def add_column_if_not_exists(table_name, column_name, column_definition):
    """Add a column if it doesn't exist"""
    if check_column_exists(table_name, column_name):
        logger.info(f"Column {table_name}.{column_name} already exists")
        return True
    
    try:
        query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        execute_query(query)
        logger.info(f"Added column {table_name}.{column_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to add column {table_name}.{column_name}: {str(e)}")
        return False

def check_table_exists(table_name):
    """Check if a table exists"""
    try:
        conn = get_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        db_name = os.getenv('DB_NAME', 'emotiondb')
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
        """, (db_name, table_name))
        
        result = cursor.fetchone()
        cursor.close()
        return result[0] > 0 if result else False
    except Exception as e:
        logger.error(f"Error checking table {table_name}: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def verify_table_structure(table_name, required_columns):
    """Verify that a table has all required columns"""
    logger.info(f"\nChecking {table_name} table...")
    missing_columns = []
    
    for column_name in required_columns:
        if not check_column_exists(table_name, column_name):
            missing_columns.append(column_name)
            logger.warning(f"  Missing column: {column_name}")
        else:
            logger.info(f"  ✓ Column {column_name} exists")
    
    return missing_columns

def migrate_database():
    """Main migration function"""
    logger.info("=" * 60)
    logger.info("Database Migration and Verification Script")
    logger.info("=" * 60)
    
    # Check live_sessions table
    required_live_sessions = [
        'id', 'teacher_id', 'title', 'description', 'meet_url',
        'scheduled_at', 'started_at', 'ended_at', 'status',
        'created_at', 'updated_at'
    ]
    
    missing = verify_table_structure('live_sessions', required_live_sessions)
    if missing:
        logger.info(f"\nAdding missing columns to live_sessions...")
        if 'started_at' in missing:
            add_column_if_not_exists('live_sessions', 'started_at', 'TIMESTAMP NULL')
        if 'ended_at' in missing:
            add_column_if_not_exists('live_sessions', 'ended_at', 'TIMESTAMP NULL')
    
    # Check engagement_reports table
    required_reports = [
        'id', 'session_type', 'session_id', 'student_id',
        'overall_engagement', 'average_emotion', 'engagement_drops',
        'focus_percentage', 'boredom_percentage', 'confusion_percentage',
        'sleepiness_percentage', 'emotional_timeline', 'behavior_summary',
        'generated_at'
    ]
    
    missing = verify_table_structure('engagement_reports', required_reports)
    if missing:
        logger.info(f"\nAdding missing columns to engagement_reports...")
        if 'focus_percentage' in missing:
            add_column_if_not_exists('engagement_reports', 'focus_percentage', 'DECIMAL(5, 2)')
        if 'boredom_percentage' in missing:
            add_column_if_not_exists('engagement_reports', 'boredom_percentage', 'DECIMAL(5, 2)')
        if 'confusion_percentage' in missing:
            add_column_if_not_exists('engagement_reports', 'confusion_percentage', 'DECIMAL(5, 2)')
        if 'sleepiness_percentage' in missing:
            add_column_if_not_exists('engagement_reports', 'sleepiness_percentage', 'DECIMAL(5, 2)')
        if 'emotional_timeline' in missing:
            add_column_if_not_exists('engagement_reports', 'emotional_timeline', 'JSON')
        if 'behavior_summary' in missing:
            add_column_if_not_exists('engagement_reports', 'behavior_summary', 'TEXT')
    
    # Check session_participants table
    required_participants = [
        'id', 'session_type', 'session_id', 'student_id',
        'joined_at', 'left_at', 'duration'
    ]
    
    missing = verify_table_structure('session_participants', required_participants)
    if missing:
        logger.info(f"\nAdding missing columns to session_participants...")
        if 'duration' in missing:
            add_column_if_not_exists('session_participants', 'duration', 'INTEGER')
    
    # Check and create live_session_logs table
    logger.info("\nChecking live_session_logs table...")
    if not check_table_exists('live_session_logs'):
        logger.info("Creating live_session_logs table...")
        try:
            execute_query("""
                CREATE TABLE live_session_logs (
                    id VARCHAR(36) PRIMARY KEY,
                    live_session_id VARCHAR(36) NOT NULL,
                    student_id VARCHAR(36) NOT NULL,
                    emotion VARCHAR(50) NOT NULL,
                    confidence DECIMAL(5, 4) NOT NULL,
                    engagement_score DECIMAL(5, 4),
                    concentration_score DECIMAL(5, 2),
                    timestamp INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (live_session_id) REFERENCES live_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_live_session_logs_session (live_session_id),
                    INDEX idx_live_session_logs_student (student_id),
                    INDEX idx_live_session_logs_timestamp (timestamp)
                )
            """)
            logger.info("  ✓ live_session_logs table created")
        except Exception as e:
            logger.error(f"  Failed to create live_session_logs table: {str(e)}")
    else:
        logger.info("  ✓ live_session_logs table already exists")
        # Verify columns
        required_logs = [
            'id', 'live_session_id', 'student_id', 'emotion', 'confidence',
            'engagement_score', 'concentration_score', 'timestamp', 'created_at'
        ]
        missing = verify_table_structure('live_session_logs', required_logs)
        if missing:
            logger.info(f"Adding missing columns to live_session_logs...")
            if 'engagement_score' in missing:
                add_column_if_not_exists('live_session_logs', 'engagement_score', 'DECIMAL(5, 4)')
            if 'concentration_score' in missing:
                add_column_if_not_exists('live_session_logs', 'concentration_score', 'DECIMAL(5, 2)')
    
    # Ensure status enum has all values
    try:
        logger.info("\nEnsuring live_sessions.status enum has all required values...")
        execute_query("""
            ALTER TABLE live_sessions 
            MODIFY COLUMN status ENUM('scheduled', 'live', 'ended', 'cancelled') DEFAULT 'scheduled'
        """)
        logger.info("  ✓ Status enum updated")
    except Exception as e:
        logger.warning(f"  Could not update status enum: {str(e)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Migration completed!")
    logger.info("=" * 60)

if __name__ == '__main__':
    try:
        migrate_database()
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

