import React from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import './App.css';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import LiveSession from './pages/LiveSession';
import Login from './pages/Login';
import Register from './pages/Register';
import ReportPage from './pages/ReportPage';
import StudentDashboard from './pages/StudentDashboard';
import TeacherDashboard from './pages/TeacherDashboard';
import VideoPlayer from './pages/VideoPlayer';

const PrivateRoute: React.FC<{ children: React.ReactNode; allowedRoles: ('teacher' | 'student')[] }> = ({ 
  children, 
  allowedRoles 
}) => {
  const { user, loading, token } = useAuth();

  // Show loading while checking authentication
  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  // Check if user has a valid token (even if user data is still loading)
  const hasToken = token || localStorage.getItem('token');
  
  // If no token at all, redirect to login
  if (!hasToken) {
    return <Navigate to="/login" replace />;
  }

  // If we have a token but user data is not loaded yet, try to get from localStorage
  if (hasToken && !user) {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        // Validate that stored user data is valid JSON
        JSON.parse(storedUser);
        // If we have cached user data, use it temporarily while waiting for server validation
        // But still show loading to indicate we're verifying
        return <div className="loading">Loading...</div>;
      } catch {
        // Invalid user data, wait for fetchUser to complete
        return <div className="loading">Loading...</div>;
      }
    }
    // No cached user, wait for fetchUser
    return <div className="loading">Loading...</div>;
  }

  // If user is loaded but role doesn't match, redirect to their dashboard
  if (user && !allowedRoles.includes(user.role)) {
    return <Navigate to={user.role === 'teacher' ? '/teacher/dashboard' : '/student/dashboard'} replace />;
  }

  // User is authenticated and has correct role
  return <>{children}</>;
};

const RootRoute: React.FC = () => {
  const token = localStorage.getItem('token');
  const userStr = localStorage.getItem('user');
  
  if (token && userStr) {
    try {
      const user = JSON.parse(userStr);
      return <Navigate to={user.role === 'teacher' ? '/teacher/dashboard' : '/student/dashboard'} replace />;
    } catch {
      return <Navigate to="/login" replace />;
    }
  }
  
  return <Navigate to="/login" replace />;
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          
          <Route
            path="/teacher/dashboard"
            element={
              <PrivateRoute allowedRoles={['teacher']}>
                <TeacherDashboard />
              </PrivateRoute>
            }
          />
          
          <Route
            path="/student/dashboard"
            element={
              <PrivateRoute allowedRoles={['student']}>
                <StudentDashboard />
              </PrivateRoute>
            }
          />
          
          <Route
            path="/video/:id"
            element={
              <PrivateRoute allowedRoles={['student', 'teacher']}>
                <VideoPlayer />
              </PrivateRoute>
            }
          />
          
          <Route
            path="/live/:id"
            element={
              <PrivateRoute allowedRoles={['student', 'teacher']}>
                <LiveSession />
              </PrivateRoute>
            }
          />
          
          <Route
            path="/report/:sessionType/:sessionId"
            element={
              <PrivateRoute allowedRoles={['student', 'teacher']}>
                <ReportPage />
              </PrivateRoute>
            }
          />
          
          <Route path="/" element={<RootRoute />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;

