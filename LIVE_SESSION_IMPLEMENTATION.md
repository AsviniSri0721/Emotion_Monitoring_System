# Live Session Monitoring Implementation

## Overview
This document describes the Live Session Monitoring feature that has been added to the Emotion Monitoring System. This feature allows teachers to create live sessions with Google Meet links, and students can join these sessions with silent emotion monitoring (no interventions).

## ✅ Completed Features

### Backend Implementation

#### 1. New Module: `backend/live_sessions/`
- **models.py**: Database models for `LiveSession` and `LiveSessionLog`
- **service.py**: Business logic for live session operations
- **controller.py**: HTTP request handlers
- **routes.py**: Route definitions

#### 2. New API Endpoints

- **POST `/api/live-sessions/create`**
  - Creates a new live session
  - Required fields: `title`, `meetUrl`
  - Optional: `startTime`
  - Teacher only

- **GET `/api/live-sessions/available`**
  - Returns all available live sessions (scheduled or live)
  - Accessible by both teachers and students

- **POST `/api/live-sessions/:id/stream`**
  - Streams emotion detection for live sessions
  - **NO intervention logic** - only records data
  - Saves to `live_session_logs` table (not `emotion_data`)
  - Interval: 5 seconds (configurable)

- **GET `/api/live-sessions/:id/report`**
  - Returns analytics report for a live session
  - Teachers see all students' data
  - Students see only their own data

#### 3. Database Migration

**New Table: `live_session_logs`**
```sql
CREATE TABLE IF NOT EXISTS live_session_logs (
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
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
);
```

**Migration File**: `backend/migrate_live_session_logs.sql`

### Frontend Implementation

#### 1. New API Client: `client/src/api/liveSessions.ts`
- TypeScript interfaces for all live session types
- API functions for all endpoints

#### 2. New Hook: `client/src/hooks/useLiveSessionEmotionStream.ts`
- Emotion detection hook specifically for live sessions
- **NO intervention checks** - only records data
- Default interval: 5 seconds
- Uses `/api/live-sessions/:id/stream` endpoint

#### 3. Updated Components

- **LiveSession.tsx**: 
  - Now uses `useLiveSessionEmotionStream` hook
  - Removed intervention-related UI
  - Uses new API endpoints

- **TeacherDashboard.tsx**:
  - Updated to use `liveSessionsApi.create()`
  - Updated to use `liveSessionsApi.getAvailable()`
  - Create live session modal already exists

- **StudentDashboard.tsx**:
  - Updated to use `liveSessionsApi.getAvailable()`
  - Shows available live sessions

- **ReportPage.tsx**:
  - Updated to support live session reports
  - Uses `liveSessionsApi.getReport()` for live sessions

## 🔑 Key Differences from Recorded Sessions

1. **No Interventions**: Live sessions do NOT trigger interventions, popups, or redirects
2. **Separate Storage**: Data stored in `live_session_logs` table (not `emotion_data`)
3. **Different Endpoint**: Uses `/api/live-sessions/:id/stream` instead of `/api/emotions/stream`
4. **Longer Interval**: Default 5 seconds (vs 1 second for recorded)
5. **No Intervention Checks**: Hook does not check `/api/interventions/check`

## 📋 Setup Instructions

### 1. Database Migration
Run the migration script in phpMyAdmin:
```sql
-- Run: backend/migrate_live_session_logs.sql
```

### 2. Backend
The routes are automatically registered in `app.py`. No additional setup needed.

### 3. Frontend
All components are updated. No additional setup needed.

## 🎯 Usage Flow

### Teacher Flow:
1. Login as teacher
2. Go to Teacher Dashboard → Live Sessions tab
3. Click "Create Session"
4. Enter title and Google Meet URL
5. Session is created and visible to students

### Student Flow:
1. Login as student
2. Go to Student Dashboard → Live Sessions tab
3. See available live sessions
4. Click "Join Session"
5. Google Meet opens in iframe
6. Click "Start Monitoring" to begin emotion detection
7. Monitoring runs silently (no popups/interventions)
8. Data is saved to `live_session_logs`

### Viewing Reports:
- Teachers: Dashboard → Reports tab → View Details
- Students: Can view their own reports via ReportPage

## 🔒 Constraints Maintained

✅ **No modifications to existing endpoints**:
- `/api/emotions/stream` - unchanged
- `/api/sessions/recorded/*` - unchanged

✅ **No modifications to recorded-session workflow**:
- Intervention logic untouched
- Recorded session monitoring unchanged

✅ **No modifications to ModelService**:
- Reuses existing emotion detection models
- No changes to model loading or inference

✅ **Fully separated logic**:
- New module: `backend/live_sessions/`
- New frontend API: `client/src/api/liveSessions.ts`
- New hook: `useLiveSessionEmotionStream`

## 📊 Data Structure

### Live Session Log Entry:
```typescript
{
  id: string;
  live_session_id: string;
  student_id: string;
  emotion: string;
  confidence: number; // 0-1
  engagement_score: number; // 0-1
  concentration_score: number; // 0-100
  timestamp: number; // seconds from session start
  created_at: timestamp;
}
```

## 🐛 Troubleshooting

1. **Migration fails**: Ensure `live_sessions` table exists first
2. **Models not loaded**: Check backend logs for model loading errors
3. **No emotion detection**: Ensure webcam permissions are granted
4. **Report not showing**: Check if any logs exist for the session

## 📝 Notes

- Live sessions use a separate table to keep data isolated
- Emotion detection interval is configurable (default: 5s)
- Reports are generated on-demand (not pre-computed)
- All existing recorded-session functionality remains unchanged






