# Database Migration Guide

This guide helps you verify and update your database schema to ensure all required columns exist for the Emotion Monitoring System.

## Quick Check

The easiest way to check and migrate your database is using the Python script:

```bash
cd backend
venv\Scripts\activate  # On Windows
python check_and_migrate_db.py
```

This script will:
- Check if all required columns exist
- Add any missing columns automatically
- Verify table structures
- Update enum values if needed

## Manual Migration (SQL)

If you prefer to use SQL directly in phpMyAdmin:

### Option 1: Safe Migration (Recommended)
Run `migrate_database_safe.sql` in phpMyAdmin. This script:
- Checks if columns exist before adding them
- Won't fail if columns already exist
- Adds missing indexes

### Option 2: Simple Migration
Run `migrate_database.sql` in phpMyAdmin. This is simpler but may show errors if columns already exist (which is safe to ignore).

### Option 3: Check Only
Run `check_database.sql` to verify your current database structure without making changes.

## Required Tables and Columns

### `live_sessions` Table
Required columns:
- `id` (VARCHAR(36), PRIMARY KEY)
- `teacher_id` (VARCHAR(36), FOREIGN KEY)
- `title` (VARCHAR(255))
- `description` (TEXT)
- `meet_url` (VARCHAR(500))
- `scheduled_at` (TIMESTAMP)
- `started_at` (TIMESTAMP NULL) - **May be missing**
- `ended_at` (TIMESTAMP NULL) - **May be missing**
- `status` (ENUM: 'scheduled', 'live', 'ended', 'cancelled')
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### `engagement_reports` Table
Required columns:
- `id` (VARCHAR(36), PRIMARY KEY)
- `session_type` (ENUM: 'live', 'recorded')
- `session_id` (VARCHAR(36))
- `student_id` (VARCHAR(36), FOREIGN KEY)
- `overall_engagement` (DECIMAL(5, 4))
- `average_emotion` (VARCHAR(50))
- `engagement_drops` (INTEGER)
- `focus_percentage` (DECIMAL(5, 2))
- `boredom_percentage` (DECIMAL(5, 2))
- `confusion_percentage` (DECIMAL(5, 2))
- `sleepiness_percentage` (DECIMAL(5, 2))
- `emotional_timeline` (JSON)
- `behavior_summary` (TEXT)
- `generated_at` (TIMESTAMP)

### `session_participants` Table
Required columns:
- `id` (VARCHAR(36), PRIMARY KEY)
- `session_type` (ENUM: 'live', 'recorded')
- `session_id` (VARCHAR(36))
- `student_id` (VARCHAR(36), FOREIGN KEY)
- `joined_at` (TIMESTAMP)
- `left_at` (TIMESTAMP NULL)
- `duration` (INTEGER) - **May be missing**

## Verification

After running the migration, verify your database:

1. **Using Python script:**
   ```bash
   python check_and_migrate_db.py
   ```

2. **Using SQL in phpMyAdmin:**
   ```sql
   -- Check live_sessions
   DESCRIBE live_sessions;
   
   -- Check engagement_reports
   DESCRIBE engagement_reports;
   
   -- Check session_participants
   DESCRIBE session_participants;
   ```

## Common Issues

### Issue: "Column already exists" error
**Solution:** This is safe to ignore. The column already exists, which is what we want.

### Issue: "Unknown column" error in application
**Solution:** Run the migration script to add the missing column.

### Issue: JSON column type not supported
**Solution:** MySQL 5.7+ is required for JSON columns. If using older MySQL, the `emotional_timeline` column will be TEXT instead.

## Notes

- The original `database_schema_mysql.sql` should already have all required columns
- Migration scripts are safe to run multiple times
- Always backup your database before running migrations in production
- The Python migration script is the safest option as it checks before modifying











