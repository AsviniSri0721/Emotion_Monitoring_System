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
          
          // Validate token in background (don't block UI)
          // Use a small delay to ensure token is properly set
          // Don't clear token if validation fails - let the user stay logged in with cached data
          setTimeout(() => {
            fetchUser().catch(err => {
              // Only log the error, don't clear token
              // This prevents logout on network errors or temporary validation issues
              const status = err?.response?.status;
              if (status === 401 || status === 422) {
                logger.warn('Background token validation failed - but keeping cached user data', {
                  status,
                  error: err?.response?.data?.error
                });
              } else {
                logger.warn('Background token validation failed (network error?)', err);
              }
            });
          }, 100);
        } catch (e) {
          console.error('Error parsing stored user:', e);
          setLoading(false);
        }
      } else {
        // No cached user, must fetch from server
        fetchUser();
      }
    } else {
      setLoading(false);
    }
  }, []); // Only run on mount

  const fetchUser = async () => {
    try {
      logger.info('Fetching user data from server');
      
      // Ensure token is in headers
      const token = localStorage.getItem('token');
      if (!token) {
        logger.warn('No token found in localStorage');
        setLoading(false);
        return null;
      }
      
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      const response = await api.get('/auth/me');
      const userData = response.data;
      setUser(userData);
      // Update stored user data
      localStorage.setItem('user', JSON.stringify(userData));
      logger.info('User data fetched successfully', { userId: userData.id, role: userData.role });
      setLoading(false);
      return userData;
    } catch (error: any) {
      const status = error?.response?.status;
      const errorMessage = error?.response?.data?.error || error?.message;
      const errorDetails = error?.response?.data?.details;
      
      logger.error('Token validation failed', { 
        status, 
        error: errorMessage,
        details: errorDetails,
        url: error?.config?.url,
        hasToken: !!localStorage.getItem('token')
      });
      
      // Only clear token if it's a definitive auth error
      // 401 = Unauthorized (token missing/invalid)
      // 422 = Unprocessable Entity (could be token validation or other issues)
      // But check the error message to be sure
      const isTokenError = status === 401 || 
                          (status === 422 && (
                            errorMessage?.toLowerCase().includes('token') ||
                            errorMessage?.toLowerCase().includes('invalid') ||
                            errorDetails?.toLowerCase().includes('token')
                          ));
      
      if (isTokenError) {
        // Token is definitely invalid, clear everything
        logger.error('Clearing invalid token', { errorMessage, errorDetails });
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setToken(null);
        setUser(null);
        delete api.defaults.headers.common['Authorization'];
      } else {
        // Network error or other issue - keep the token and user
        // This prevents logout on network issues or temporary server problems
        logger.warn('Failed to fetch user, but keeping token (may be temporary issue)', { 
          status, 
          error: errorMessage,
          details: errorDetails
        });
      }
      setLoading(false);
      throw error; // Re-throw so caller can handle it
    }
  };

  const login = async (email: string, password: string) => {
    try {
      logger.info('Attempting login', { email });
      const response = await api.post('/auth/login', { email, password });
      const { token: newToken, user: newUser } = response.data;
      
      if (!newToken || !newUser) {
        throw new Error('Invalid response from server');
      }
      
      // Store token and user
      localStorage.setItem('token', newToken);
      localStorage.setItem('user', JSON.stringify(newUser));
      
      // Update state immediately
      setToken(newToken);
      setUser(newUser);
      api.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
      
      logger.info('Login successful', { userId: newUser.id, role: newUser.role });
      
      // Set loading to false since we have user data
      setLoading(false);
      
      // Don't call fetchUser() here - we already have the user data from login
      // This prevents the immediate logout issue
    } catch (error: any) {
      logger.error('Login failed', { email, error: error.message, status: error?.response?.status });
      // Clear any existing token if login fails
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setToken(null);
      setUser(null);
      delete api.defaults.headers.common['Authorization'];
      setLoading(false);
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

