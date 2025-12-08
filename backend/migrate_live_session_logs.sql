-- Migration script to create live_session_logs table
-- Run this in phpMyAdmin or MySQL command line for the 'emotiondb' database

-- Create live_session_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS live_session_logs (
    id VARCHAR(36) PRIMARY KEY,
    live_session_id VARCHAR(36) NOT NULL,
    student_id VARCHAR(36) NOT NULL,
    emotion VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    engagement_score DECIMAL(5, 4), -- 0.0000 to 1.0000
    concentration_score DECIMAL(5, 2), -- 0.00 to 100.00
    timestamp INTEGER NOT NULL, -- seconds from session start
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (live_session_id) REFERENCES live_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_live_session_logs_session (live_session_id),
    INDEX idx_live_session_logs_student (student_id),
    INDEX idx_live_session_logs_timestamp (timestamp)
);



