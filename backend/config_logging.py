"""
Logging configuration for the Emotion Monitoring System backend
Creates log files for debugging and monitoring
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(app):
    """Setup comprehensive logging for the Flask application"""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file paths
    app_log_file = os.path.join(log_dir, 'app.log')
    error_log_file = os.path.join(log_dir, 'errors.log')
    api_log_file = os.path.join(log_dir, 'api.log')
    auth_log_file = os.path.join(log_dir, 'auth.log')
    
    # Remove default Flask logger handlers
    app.logger.handlers.clear()
    
    # Set log level
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    app.logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always show INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    app.logger.addHandler(console_handler)
    
    # Main app log file (rotating, max 10MB, keep 5 backups)
    app_file_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(formatter)
    app.logger.addHandler(app_file_handler)
    
    # Error log file (only errors and critical)
    error_file_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    app.logger.addHandler(error_file_handler)
    
    # API request/response logger
    api_logger = logging.getLogger('api')
    api_logger.setLevel(logging.INFO)
    api_file_handler = RotatingFileHandler(
        api_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    api_file_handler.setLevel(logging.INFO)
    api_file_handler.setFormatter(formatter)
    api_logger.addHandler(api_file_handler)
    api_logger.addHandler(console_handler)
    
    # Authentication logger
    auth_logger = logging.getLogger('auth')
    auth_logger.setLevel(logging.INFO)
    auth_file_handler = RotatingFileHandler(
        auth_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    auth_file_handler.setLevel(logging.INFO)
    auth_file_handler.setFormatter(formatter)
    auth_logger.addHandler(auth_file_handler)
    auth_logger.addHandler(console_handler)
    
    # Database logger
    db_logger = logging.getLogger('database')
    db_logger.setLevel(logging.WARNING)  # Only log warnings and errors
    
    # Log startup message
    app.logger.info("=" * 60)
    app.logger.info("Emotion Monitoring System - Backend Started")
    app.logger.info(f"Log Level: {log_level}")
    app.logger.info(f"Log Directory: {log_dir}")
    app.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    app.logger.info("=" * 60)
    
    return app.logger









