import axios, { AxiosError } from 'axios';
import logger from '../utils/logger';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add token dynamically and handle file uploads
api.interceptors.request.use(
  (config) => {
    // Log API request
    logger.debug(`API Request: ${config.method?.toUpperCase()} ${config.url}`, {
      params: config.params,
      hasData: !!config.data,
    });
    
    // Add token from localStorage dynamically for each request
    // Also check api.defaults.headers in case it was set there
    let token = localStorage.getItem('token');
    if (!token) {
      const authHeader = api.defaults.headers.common['Authorization'];
      if (typeof authHeader === 'string') {
        token = authHeader.replace('Bearer ', '');
      }
    }
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
      logger.debug(`Token added to request: ${token.substring(0, 20)}...`);
    } else {
      logger.warn(`No token found for request: ${config.url}`);
    }
    
    // Don't set Content-Type for FormData - let browser set it with boundary
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }
    return config;
  },
  (error) => {
    logger.error('API Request Error', error);
    return Promise.reject(error);
  }
);

// Response interceptor to handle token errors
api.interceptors.response.use(
  (response) => {
    // Log successful API response
    logger.debug(`API Response: ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`);
    return response;
  },
  (error: AxiosError) => {
    // Log API error
    const errorInfo = {
      method: error.config?.method?.toUpperCase(),
      url: error.config?.url,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message,
    };
    
    logger.error('API Error', errorInfo);
    
    // Only clear token for auth-related endpoints on 401/422
    // Don't clear token for other endpoints (like /videos, /sessions) as they might fail for other reasons
    const isAuthEndpoint = error.config?.url?.includes('/auth/');
    
    if (error.response && (error.response.status === 401 || error.response.status === 422)) {
      if (isAuthEndpoint) {
        // Only clear token if it's an auth endpoint (like /auth/me)
        logger.warn('Authentication failed on auth endpoint - clearing token');
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        delete api.defaults.headers.common['Authorization'];
        // Redirect to login if not already there
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }
      } else {
        // For other endpoints, just log the error but don't clear token
        // The component can handle the error
        logger.warn(`API call failed but keeping token: ${error.config?.url}`);
      }
    }
    return Promise.reject(error);
  }
);

export default api;

