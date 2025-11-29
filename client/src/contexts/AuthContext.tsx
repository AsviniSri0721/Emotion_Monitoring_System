import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import api from '../services/api';
import logger from '../utils/logger';

interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'teacher' | 'student';
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, firstName: string, lastName: string, role: 'teacher' | 'student') => Promise<void>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing token on mount
    const storedToken = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user');
    
    if (storedToken) {
      setToken(storedToken);
      api.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`;
      
      // If user data is stored, use it immediately (faster UX)
      if (storedUser) {
        try {
          const userData = JSON.parse(storedUser);
          setUser(userData);
          setLoading(false); // Set loading to false immediately if we have cached user
        } catch (e) {
          console.error('Error parsing stored user:', e);
        }
      }
      
      // Verify token is still valid by fetching user from server (in background)
      // Don't block UI if we have cached user data
      fetchUser();
    } else {
      setLoading(false);
    }
  }, []); // Only run on mount

  const fetchUser = async () => {
    try {
      logger.info('Fetching user data from server');
      const response = await api.get('/auth/me');
      const userData = response.data;
      setUser(userData);
      // Update stored user data
      localStorage.setItem('user', JSON.stringify(userData));
      logger.info('User data fetched successfully', { userId: userData.id, role: userData.role });
      setLoading(false);
    } catch (error: any) {
      // Only clear token if it's an auth error (401/422), not network errors
      const status = error?.response?.status;
      if (status === 401 || status === 422) {
        // Token is invalid, clear everything
        logger.error('Token validation failed', { status, error: error.message });
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setToken(null);
        setUser(null);
        delete api.defaults.headers.common['Authorization'];
      } else {
        // Network error or other issue - keep the token and user
        logger.warn('Failed to fetch user, but keeping token', { status, error: error.message });
      }
      setLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    try {
      logger.info('Attempting login', { email });
      const response = await api.post('/auth/login', { email, password });
      const { token: newToken, user: newUser } = response.data;
      
      // Store token and user
      localStorage.setItem('token', newToken);
      localStorage.setItem('user', JSON.stringify(newUser));
      
      // Update state immediately
      setToken(newToken);
      setUser(newUser);
      api.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
      
      logger.info('Login successful', { userId: newUser.id, role: newUser.role });
      
      // Don't call fetchUser() here - we already have the user data from login
      // The useEffect won't run again because we're not changing dependencies
    } catch (error: any) {
      logger.error('Login failed', { email, error: error.message, status: error?.response?.status });
      // Clear any existing token if login fails
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setToken(null);
      setUser(null);
      delete api.defaults.headers.common['Authorization'];
      throw error; // Re-throw so Login component can handle it
    }
  };

  const register = async (
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    role: 'teacher' | 'student'
  ) => {
    try {
      const response = await api.post('/auth/register', {
        email,
        password,
        firstName,
        lastName,
        role,
      });
      const { token: newToken, user: newUser } = response.data;
      
      // Store token and user
      localStorage.setItem('token', newToken);
      localStorage.setItem('user', JSON.stringify(newUser));
      
      // Update state immediately
      setToken(newToken);
      setUser(newUser);
      api.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
      
      // Don't call fetchUser() here - we already have the user data from register
    } catch (error: any) {
      // Clear any existing token if register fails
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setToken(null);
      setUser(null);
      delete api.defaults.headers.common['Authorization'];
      throw error; // Re-throw so Register component can handle it
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
    delete api.defaults.headers.common['Authorization'];
  };

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

