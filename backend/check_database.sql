-- Check database schema and verify all required columns exist
-- Run this in phpMyAdmin to verify your database structure

-- Check live_sessions table
SELECT 'Checking live_sessions table...' AS status;
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'live_sessions'
ORDER BY ORDINAL_POSITION;

-- Check engagement_reports table
SELECT 'Checking engagement_reports table...' AS status;
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'engagement_reports'
ORDER BY ORDINAL_POSITION;

-- Check session_participants table
SELECT 'Checking session_participants table...' AS status;
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'session_participants'
ORDER BY ORDINAL_POSITION;

-- Check indexes
SELECT 'Checking indexes...' AS status;
SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME IN ('live_sessions', 'engagement_reports', 'session_participants')
ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;

-- Summary of required columns
SELECT 'Required columns check:' AS summary;
SELECT 
    'live_sessions' AS table_name,
    CASE 
        WHEN COUNT(*) >= 9 THEN 'OK - All columns present'
        ELSE CONCAT('WARNING - Missing columns. Found: ', COUNT(*), ' Expected: 9')
    END AS status
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'live_sessions'
UNION ALL
SELECT 
    'engagement_reports' AS table_name,
    CASE 
        WHEN COUNT(*) >= 13 THEN 'OK - All columns present'
        ELSE CONCAT('WARNING - Missing columns. Found: ', COUNT(*), ' Expected: 13')
    END AS status
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'engagement_reports'
UNION ALL
SELECT 
    'session_participants' AS table_name,
    CASE 
        WHEN COUNT(*) >= 6 THEN 'OK - All columns present'
        ELSE CONCAT('WARNING - Missing columns. Found: ', COUNT(*), ' Expected: 6')
    END AS status
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'session_participants';















