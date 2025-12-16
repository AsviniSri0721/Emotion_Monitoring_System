# Debugging 422 Error on Video Upload

## Current Issue
Getting 422 "UNPROCESSABLE ENTITY" error when uploading video.

## What 422 Means
- Request reached the server
- Request format is valid
- But server cannot process the request (usually validation or data issue)

## Debugging Steps

### 1. Check Backend Terminal
Look at the backend terminal where `python app.py` is running. You should see:
- `Upload request received. Content-Type: ...`
- `Request files: ...`
- `Request form: ...`
- `Current user: ...`
- Any error messages

### 2. Common Causes

#### A. Database Connection Issue
**Symptoms:**
- Backend logs show "Database connection failed"
- Error mentions "connection" or "database"

**Fix:**
- Check XAMPP MySQL is running
- Verify `emotiondb` database exists
- Check `backend/.env` database settings

#### B. JWT Token Issue
**Symptoms:**
- Backend logs show authentication error
- "Unauthorized" or "Invalid token"

**Fix:**
- Logout and login again
- Check token in browser localStorage
- Verify JWT_SECRET in backend/.env

#### C. File Path Issue
**Symptoms:**
- Error about file path or directory
- "Permission denied" or "File not found"

**Fix:**
- Create `backend/uploads/` directory
- Check write permissions

#### D. Database Schema Issue
**Symptoms:**
- Error about table or column not found
- SQL syntax error

**Fix:**
- Verify database schema is imported
- Check `videos` table exists in phpMyAdmin

### 3. Quick Test

Test the upload endpoint directly:

```bash
# Get your JWT token from browser localStorage
# Then test with curl:

curl -X POST http://localhost:5000/api/videos/upload \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "video=@test.mp4" \
  -F "title=Test Video" \
  -F "description=Test"
```

### 4. Check Database

In phpMyAdmin:
1. Open `emotiondb` database
2. Check `videos` table structure
3. Verify columns: id, teacher_id, title, description, file_path
4. Try manual insert to test:
```sql
INSERT INTO videos (id, teacher_id, title, description, file_path)
VALUES ('test-id', 'teacher-id', 'Test', 'Test desc', 'test.mp4');
```

### 5. Backend Logs to Look For

The backend should now log:
- Request received
- Files in request
- Form data
- Current user
- Database operations
- Any errors

**Share the backend terminal output** to help diagnose the issue.

## Most Likely Issue

Based on 422 error, it's probably:
1. **Database insert failing** - Check backend logs for database error
2. **Teacher ID format issue** - UUID format mismatch
3. **Missing database table** - Videos table not created properly

## Next Steps

1. **Check backend terminal** - Look for error messages
2. **Check browser console** - Expand the error response to see details
3. **Verify database** - Check phpMyAdmin that videos table exists
4. **Test database connection** - Try a simple query

Share the backend terminal output and I can help fix the specific issue!














