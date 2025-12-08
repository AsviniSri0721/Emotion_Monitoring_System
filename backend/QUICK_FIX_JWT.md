# Quick Fix for JWT Token Validation (422 Errors)

## The Problem
You're getting 422 errors on API calls even though:
- Login works (token is created)
- Token is being sent in requests
- But token validation fails

## Most Likely Cause
**JWT_SECRET_KEY mismatch** - The token was created with one secret key, but validated with a different one.

## Quick Fix Steps

### Step 1: Check Your .env File
Make sure `backend/.env` has `JWT_SECRET` set:

```env
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
```

### Step 2: Verify JWT Secret is Consistent
Run this test:
```bash
cd backend
venv\Scripts\activate
python test_jwt_validation.py
```

This will show if token creation/validation works.

### Step 3: Check Backend Logs
After trying to login and make API calls, check:
```bash
type backend\logs\auth.log
```

Look for "Invalid token error" messages - they will show the exact error.

### Step 4: Restart Backend
After checking/changing JWT_SECRET:
1. Stop the backend server
2. Make sure `.env` file has correct JWT_SECRET
3. Restart the backend server
4. Try logging in again

## Common Issues

1. **JWT_SECRET not set in .env**
   - Solution: Run `backend\setup_env.bat` or create `.env` manually

2. **JWT_SECRET changed after tokens were created**
   - Solution: Clear browser localStorage and login again
   - Or: Use the same JWT_SECRET that was used to create existing tokens

3. **Token format issue**
   - Solution: Check browser console - token should start with `eyJ...`
   - Check backend logs for token preview

## Test Token Validation

After login, test the token in browser console:
```javascript
// Get your token
const token = localStorage.getItem('token');
console.log('Token:', token);

// Test the token endpoint
fetch('http://localhost:5000/api/test-token', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
})
.then(r => r.json())
.then(console.log)
.catch(console.error);
```

If this works, the token is valid. If it fails, check the backend logs for the error.











