# Testing Login Functionality

## Backend Status
✅ Backend is now configured to run **without ML models** for testing login functionality.

## Testing Steps

### 1. Verify Backend is Running
- Backend should be on: `http://localhost:5000`
- Test: Open `http://localhost:5000/api/health` in browser
- Should see: `{"status":"ok","message":"Emotion Monitoring System API"}`

### 2. Test Registration

#### Create a Teacher Account:
1. Go to: `http://localhost:3000/register`
2. Fill in:
   - First Name: `John`
   - Last Name: `Teacher`
   - Email: `teacher@test.com`
   - Password: `teacher123`
   - Role: **Teacher**
3. Click "Register"
4. Should redirect to: `/teacher/dashboard`

#### Create a Student Account:
1. Logout (if logged in)
2. Go to: `http://localhost:3000/register`
3. Fill in:
   - First Name: `Jane`
   - Last Name: `Student`
   - Email: `student@test.com`
   - Password: `student123`
   - Role: **Student**
4. Click "Register"
5. Should redirect to: `/student/dashboard`

### 3. Test Login

#### Teacher Login:
1. Go to: `http://localhost:3000/login`
2. Enter:
   - Email: `teacher@test.com`
   - Password: `teacher123`
3. Click "Login"
4. Should redirect to: `/teacher/dashboard`
5. Should see: "Teacher Dashboard" header

#### Student Login:
1. Logout
2. Go to: `http://localhost:3000/login`
3. Enter:
   - Email: `student@test.com`
   - Password: `student123`
4. Click "Login"
5. Should redirect to: `/student/dashboard`
6. Should see: "Student Dashboard" header

### 4. Test Dashboard Differences

#### Teacher Dashboard Features:
- ✅ "Upload Video" button
- ✅ "Create Session" button
- ✅ Tabs: Videos, Live Sessions, Reports
- ✅ Can view engagement reports

#### Student Dashboard Features:
- ✅ List of available videos
- ✅ "Watch Lecture" buttons
- ✅ Tabs: Recorded Lectures, Live Sessions
- ✅ Can join live sessions

### 5. Test Logout
- Click "Logout" button in header
- Should redirect to: `/login`
- Should clear session

## Expected Behavior

✅ **Registration**:
- Creates user in database
- Returns JWT token
- Redirects to appropriate dashboard based on role

✅ **Login**:
- Validates credentials
- Returns JWT token
- Redirects to appropriate dashboard based on role

✅ **Role-based Access**:
- Teachers see teacher features
- Students see student features
- Cannot access wrong dashboard

✅ **Session Management**:
- Token stored in localStorage
- Token sent with API requests
- Logout clears token

## Troubleshooting

### Backend not starting:
- Check if XAMPP MySQL is running
- Verify database `emotiondb` exists
- Check `backend/.env` configuration

### Registration/Login fails:
- Check browser console for errors
- Check backend terminal for error messages
- Verify database connection

### Wrong dashboard redirect:
- Check user role in database
- Verify JWT token contains correct role

## Notes

- **ML models not required** for login testing
- Backend will show warning about missing models (this is OK)
- Emotion detection won't work without models (expected)
- All other features (auth, videos, sessions) should work

