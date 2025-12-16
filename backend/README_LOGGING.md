# Logging Guide

This document explains the logging system for the Emotion Monitoring System.

## Backend Logs

Backend logs are stored in the `backend/logs/` directory.

### Log Files

1. **`app.log`** - General application logs (INFO level and above)
   - Application startup
   - General information
   - Warnings

2. **`errors.log`** - Error logs only (ERROR and CRITICAL levels)
   - Exceptions
   - Critical errors
   - Failed operations

3. **`api.log`** - API request/response logs
   - All incoming API requests
   - Request methods and paths
   - Response status codes
   - Request data (passwords are masked)

4. **`auth.log`** - Authentication logs
   - Login attempts
   - Token validations
   - Authentication errors

### Log Rotation

- Log files are rotated when they reach 10MB
- Up to 5 backup files are kept
- Old logs are automatically deleted

### Viewing Logs

**Windows:**
```bash
# View app log
type backend\logs\app.log

# View errors
type backend\logs\errors.log

# View API logs
type backend\logs\api.log

# View auth logs
type backend\logs\auth.log
```

**Linux/Mac:**
```bash
# View app log
tail -f backend/logs/app.log

# View errors
tail -f backend/logs/errors.log

# View API logs
tail -f backend/logs/api.log

# View auth logs
tail -f backend/logs/auth.log
```

## Frontend Logs

Frontend logs are stored in the browser's localStorage and can be accessed via the browser console.

### Accessing Logs

1. **Browser Console:**
   - Open DevTools (F12)
   - Go to Console tab
   - All logs are prefixed with timestamp and level

2. **Programmatic Access:**
   ```javascript
   import logger from './utils/logger';
   
   // Get all logs
   const logs = logger.getLogs();
   
   // Get only errors
   const errors = logger.getLogs('error');
   
   // Export logs as JSON
   const logsJson = logger.exportLogs();
   
   // Download logs as file
   logger.downloadLogs();
   ```

### Log Levels

- **INFO** - General information (API calls, user actions)
- **WARN** - Warnings (non-critical issues)
- **ERROR** - Errors (failed operations, exceptions)
- **DEBUG** - Debug information (only in development mode)

### Clearing Logs

```javascript
import logger from './utils/logger';
logger.clearLogs();
```

## Debugging UI Issues

### Common Issues and Log Locations

1. **Login not working:**
   - Check: `backend/logs/auth.log`
   - Check: Browser console for frontend errors
   - Look for: Token validation errors, authentication failures

2. **API calls failing:**
   - Check: `backend/logs/api.log`
   - Check: `backend/logs/errors.log`
   - Check: Browser console for network errors
   - Look for: CORS errors, 422/401/500 status codes

3. **Video upload issues:**
   - Check: `backend/logs/app.log`
   - Check: `backend/logs/errors.log`
   - Look for: File size errors, permission errors

4. **Page refresh redirecting to login:**
   - Check: Browser console for token validation errors
   - Check: `backend/logs/auth.log` for token validation failures
   - Look for: 422 errors on `/api/auth/me`

### Log Format

**Backend:**
```
2024-01-15 10:30:45 - app - INFO - User logged in - [auth.py:95]
2024-01-15 10:30:46 - api - INFO - POST /api/videos/upload - Status: 201 - [app.py:85]
```

**Frontend:**
```
[2024-01-15T10:30:45.123Z] [INFO] User logged in {userId: "123", role: "teacher"}
[2024-01-15T10:30:46.456Z] [ERROR] API Error {method: "POST", url: "/api/videos/upload", status: 422}
```

## Environment Variables

Set log level in `.env`:
```
LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Tips

1. **For development:** Set `LOG_LEVEL=DEBUG` to see all logs
2. **For production:** Set `LOG_LEVEL=WARNING` to reduce log volume
3. **Monitor errors:** Regularly check `errors.log` for issues
4. **API debugging:** Use `api.log` to track all API requests
5. **Frontend debugging:** Use browser DevTools console and localStorage logs














