# Debugging Login Issues

If you're unable to login, follow these steps to diagnose the problem:

## Step 1: Check Backend Logs

1. **Check if backend is running:**
   ```bash
   # Should see Flask server running on port 5000
   ```

2. **Check backend logs:**
   ```bash
   # Windows
   type backend\logs\auth.log
   type backend\logs\api.log
   type backend\logs\errors.log
   ```

3. **Look for:**
   - Login request received
   - Database connection errors
   - User not found errors
   - Password validation errors

## Step 2: Test Database Connection

Run the test script:
```bash
cd backend
venv\Scripts\activate
python test_login.py
```

This will:
- Test database connection
- List all users
- Test user lookup
- Test password verification

## Step 3: Check Browser Console

1. Open browser DevTools (F12)
2. Go to Console tab
3. Try to login
4. Look for:
   - Network errors
   - CORS errors
   - API errors
   - Token errors

## Step 4: Check Network Tab

1. Open browser DevTools (F12)
2. Go to Network tab
3. Try to login
4. Check the `/api/auth/login` request:
   - Status code (should be 200)
   - Request payload (email and password)
   - Response (should have token and user)

## Step 5: Verify User Exists

1. Check if user exists in database:
   ```sql
   SELECT * FROM users WHERE email = 'your-email@test.com';
   ```

2. If user doesn't exist, register first:
   - Go to `/register` page
   - Create a new account
   - Then try to login

## Step 6: Common Issues

### Issue: "Invalid credentials"
- **Cause:** User doesn't exist or password is wrong
- **Solution:** 
  - Verify user exists in database
  - Check password is correct
  - Try registering a new user

### Issue: "Network Error" or CORS error
- **Cause:** Backend not running or CORS misconfiguration
- **Solution:**
  - Make sure backend is running on port 5000
  - Check CORS configuration in `backend/app.py`
  - Check backend logs for errors

### Issue: "Token validation failed"
- **Cause:** Token not being sent or invalid
- **Solution:**
  - Check browser console for token errors
  - Clear localStorage and try again
  - Check backend logs for token validation errors

### Issue: "Database connection failed"
- **Cause:** Database not accessible
- **Solution:**
  - Check XAMPP MySQL is running
  - Verify database credentials in `.env`
  - Check `backend/logs/errors.log` for database errors

## Step 7: Manual Test

Test the login endpoint directly:

```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"teacher@test.com\",\"password\":\"password123\"}"
```

Or use Postman/Insomnia to test the API directly.

## Step 8: Check Environment Variables

Verify `.env` file has correct values:
```env
DB_HOST=localhost
DB_NAME=emotiondb
DB_USER=root
DB_PASSWORD=
DB_PORT=3306
JWT_SECRET=your-secret-key
```

## Still Having Issues?

1. Check all log files in `backend/logs/`
2. Check browser console for frontend errors
3. Check network tab for API request/response
4. Run `python backend/test_login.py` to test database
5. Verify backend is running and accessible














