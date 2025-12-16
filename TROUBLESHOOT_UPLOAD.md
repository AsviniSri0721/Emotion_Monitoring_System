# Troubleshooting Video Upload Error

## Common Issues and Solutions

### 1. Check Backend is Running
- Open: `http://localhost:5000/api/health`
- Should see: `{"status":"ok",...}`
- If not, start backend:
  ```cmd
  cd backend
  venv\Scripts\activate
  python app.py
  ```

### 2. Check Database Connection
- Verify XAMPP MySQL is running
- Check database `emotiondb` exists
- Verify `backend/.env` has correct database settings

### 3. Check Uploads Directory
- Directory should exist: `backend/uploads/`
- If missing, create it:
  ```cmd
  mkdir backend\uploads
  ```

### 4. Check File Requirements
- **File size**: Must be under 500MB
- **File format**: mp4, webm, ogg, avi, mov
- **File name**: Should not have special characters

### 5. Check Browser Console
- Open browser DevTools (F12)
- Go to Console tab
- Look for error messages
- Check Network tab for failed requests

### 6. Check Backend Terminal
- Look at backend terminal output
- Should see error messages if upload fails
- Common errors:
  - Database connection error
  - File permission error
  - File too large

## Quick Fixes

### Fix 1: Create Uploads Directory
```cmd
cd backend
mkdir uploads
```

### Fix 2: Check Backend Logs
Look at the backend terminal for detailed error messages.

### Fix 3: Test API Directly
```cmd
curl -X POST http://localhost:5000/api/videos/upload
```

### Fix 4: Verify Database
- Open phpMyAdmin
- Check `emotiondb` database exists
- Check `videos` table exists
- Verify you can insert data manually

## Debug Steps

1. **Check Backend Status:**
   - `http://localhost:5000/api/health`

2. **Check Database:**
   - phpMyAdmin → emotiondb → videos table

3. **Check File:**
   - File size
   - File format
   - File permissions

4. **Check Network:**
   - Browser DevTools → Network tab
   - Look for POST request to `/api/videos/upload`
   - Check response status and error message

## Most Common Issue

**Database Connection Error** - Make sure:
- XAMPP MySQL is running
- Database `emotiondb` exists
- `.env` file has correct database credentials















