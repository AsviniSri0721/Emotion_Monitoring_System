-- Migration script to ensure all required columns and tables exist
-- Run this in phpMyAdmin or MySQL command line for the 'emotiondb' database

-- Check and add missing columns to live_sessions table
ALTER TABLE live_sessions 
    MODIFY COLUMN status ENUM('scheduled', 'live', 'ended', 'cancelled') DEFAULT 'scheduled',
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMP NULL AFTER scheduled_at,
    ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP NULL AFTER started_at;

-- Ensure engagement_reports has all required columns
ALTER TABLE engagement_reports
    MODIFY COLUMN overall_engagement DECIMAL(5, 4) NOT NULL,
    MODIFY COLUMN average_emotion VARCHAR(50),
    MODIFY COLUMN engagement_drops INTEGER DEFAULT 0,
    MODIFY COLUMN focus_percentage DECIMAL(5, 2),
    MODIFY COLUMN boredom_percentage DECIMAL(5, 2),
    MODIFY COLUMN confusion_percentage DECIMAL(5, 2),
    MODIFY COLUMN sleepiness_percentage DECIMAL(5, 2),
    MODIFY COLUMN emotional_timeline JSON,
    MODIFY COLUMN behavior_summary TEXT,
    MODIFY COLUMN generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Ensure session_participants has all required columns
ALTER TABLE session_participants
    MODIFY COLUMN joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    MODIFY COLUMN left_at TIMESTAMP NULL,
    MODIFY COLUMN duration INTEGER;

-- Add indexes if they don't exist (MySQL will ignore if they already exist)
CREATE INDEX IF NOT EXISTS idx_live_sessions_status ON live_sessions(status);
CREATE INDEX IF NOT EXISTS idx_live_sessions_scheduled ON live_sessions(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_engagement_reports_generated ON engagement_reports(generated_at);
CREATE INDEX IF NOT EXISTS idx_session_participants_joined ON session_participants(joined_at);

-- Verify table structures
-- This will show the current structure (run separately if needed)
-- DESCRIBE live_sessions;
-- DESCRIBE engagement_reports;
-- DESCRIBE session_participants;









