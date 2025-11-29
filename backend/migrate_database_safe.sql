-- Safe migration script for MySQL (handles existing columns gracefully)
-- Run this in phpMyAdmin or MySQL command line for the 'emotiondb' database
-- This version uses procedures to check before altering

-- Procedure to add column if it doesn't exist
DELIMITER $$

DROP PROCEDURE IF EXISTS AddColumnIfNotExists$$
CREATE PROCEDURE AddColumnIfNotExists(
    IN tableName VARCHAR(64),
    IN columnName VARCHAR(64),
    IN columnDefinition TEXT
)
BEGIN
    DECLARE columnExists INT DEFAULT 0;
    
    SELECT COUNT(*) INTO columnExists
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = tableName
      AND COLUMN_NAME = columnName;
    
    IF columnExists = 0 THEN
        SET @sql = CONCAT('ALTER TABLE ', tableName, ' ADD COLUMN ', columnName, ' ', columnDefinition);
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END$$

DELIMITER ;

-- Add started_at to live_sessions if it doesn't exist
CALL AddColumnIfNotExists('live_sessions', 'started_at', 'TIMESTAMP NULL AFTER scheduled_at');

-- Add ended_at to live_sessions if it doesn't exist
CALL AddColumnIfNotExists('live_sessions', 'ended_at', 'TIMESTAMP NULL AFTER started_at');

-- Ensure status enum includes all values
ALTER TABLE live_sessions 
    MODIFY COLUMN status ENUM('scheduled', 'live', 'ended', 'cancelled') DEFAULT 'scheduled';

-- Ensure engagement_reports columns exist and have correct types
CALL AddColumnIfNotExists('engagement_reports', 'focus_percentage', 'DECIMAL(5, 2)');
CALL AddColumnIfNotExists('engagement_reports', 'boredom_percentage', 'DECIMAL(5, 2)');
CALL AddColumnIfNotExists('engagement_reports', 'confusion_percentage', 'DECIMAL(5, 2)');
CALL AddColumnIfNotExists('engagement_reports', 'sleepiness_percentage', 'DECIMAL(5, 2)');
CALL AddColumnIfNotExists('engagement_reports', 'emotional_timeline', 'JSON');
CALL AddColumnIfNotExists('engagement_reports', 'behavior_summary', 'TEXT');

-- Ensure session_participants has duration column
CALL AddColumnIfNotExists('session_participants', 'duration', 'INTEGER');

-- Add indexes (will fail gracefully if they exist)
CREATE INDEX idx_live_sessions_status ON live_sessions(status);
CREATE INDEX idx_live_sessions_scheduled ON live_sessions(scheduled_at);
CREATE INDEX idx_engagement_reports_generated ON engagement_reports(generated_at);
CREATE INDEX idx_session_participants_joined ON session_participants(joined_at);

-- Clean up procedure
DROP PROCEDURE IF EXISTS AddColumnIfNotExists;

-- Verify tables
SELECT 'Migration completed. Verifying tables...' AS status;
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME IN ('live_sessions', 'engagement_reports', 'session_participants')
ORDER BY TABLE_NAME, ORDINAL_POSITION;

