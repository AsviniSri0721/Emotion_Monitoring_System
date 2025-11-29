-- Emotion Monitoring System Database Schema for MySQL/MariaDB (XAMPP)
-- Import this file into phpMyAdmin for your 'emotiondb' database
-- Note: Changed UUID to VARCHAR(36) and removed gen_random_uuid() for MySQL compatibility

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role ENUM('teacher', 'student') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Videos table (recorded lectures)
CREATE TABLE IF NOT EXISTS videos (
    id VARCHAR(36) PRIMARY KEY,
    teacher_id VARCHAR(36) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500) NOT NULL,
    duration INTEGER, -- in seconds
    thumbnail_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Live sessions table
CREATE TABLE IF NOT EXISTS live_sessions (
    id VARCHAR(36) PRIMARY KEY,
    teacher_id VARCHAR(36) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    meet_url VARCHAR(500),
    scheduled_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP NULL,
    ended_at TIMESTAMP NULL,
    status ENUM('scheduled', 'live', 'ended', 'cancelled') DEFAULT 'scheduled',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Session participants (for both live and recorded)
CREATE TABLE IF NOT EXISTS session_participants (
    id VARCHAR(36) PRIMARY KEY,
    session_type ENUM('live', 'recorded') NOT NULL,
    session_id VARCHAR(36) NOT NULL, -- references live_sessions.id or videos.id
    student_id VARCHAR(36) NOT NULL,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    left_at TIMESTAMP NULL,
    duration INTEGER, -- total seconds attended
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_participant (session_type, session_id, student_id)
);

-- Emotion data table
CREATE TABLE IF NOT EXISTS emotion_data (
    id VARCHAR(36) PRIMARY KEY,
    session_type ENUM('live', 'recorded') NOT NULL,
    session_id VARCHAR(36) NOT NULL,
    student_id VARCHAR(36) NOT NULL,
    emotion VARCHAR(50) NOT NULL, -- happy, sad, angry, surprised, fearful, disgusted, neutral, bored, confused, sleepy, focused
    confidence DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    timestamp INTEGER NOT NULL, -- seconds from session start
    engagement_score DECIMAL(5, 4), -- 0.0000 to 1.0000
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Engagement reports
CREATE TABLE IF NOT EXISTS engagement_reports (
    id VARCHAR(36) PRIMARY KEY,
    session_type ENUM('live', 'recorded') NOT NULL,
    session_id VARCHAR(36) NOT NULL,
    student_id VARCHAR(36) NOT NULL,
    overall_engagement DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    average_emotion VARCHAR(50),
    engagement_drops INTEGER DEFAULT 0,
    focus_percentage DECIMAL(5, 2), -- percentage of time focused
    boredom_percentage DECIMAL(5, 2),
    confusion_percentage DECIMAL(5, 2),
    sleepiness_percentage DECIMAL(5, 2),
    emotional_timeline JSON, -- array of emotion data points (MySQL 5.7+)
    behavior_summary TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_report (session_type, session_id, student_id)
);

-- Interventions table
CREATE TABLE IF NOT EXISTS interventions (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL, -- references videos.id (only for recorded sessions)
    student_id VARCHAR(36) NOT NULL,
    intervention_type ENUM('mind_game', 'music_therapy', 'micro_learning') NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    triggered_emotion VARCHAR(50) NOT NULL,
    completed_at TIMESTAMP NULL,
    duration INTEGER, -- seconds spent in intervention
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_videos_teacher ON videos(teacher_id);
CREATE INDEX idx_live_sessions_teacher ON live_sessions(teacher_id);
CREATE INDEX idx_emotion_data_session ON emotion_data(session_type, session_id);
CREATE INDEX idx_emotion_data_student ON emotion_data(student_id);
CREATE INDEX idx_emotion_data_timestamp ON emotion_data(timestamp);
CREATE INDEX idx_engagement_reports_session ON engagement_reports(session_type, session_id);
CREATE INDEX idx_engagement_reports_student ON engagement_reports(student_id);
CREATE INDEX idx_interventions_session ON interventions(session_id, student_id);

