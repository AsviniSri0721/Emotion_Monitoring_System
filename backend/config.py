import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET', 'jwt-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES = False
    
    # Database
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_NAME = os.getenv('DB_NAME', 'emotion_monitoring')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_PORT = os.getenv('DB_PORT', '5432')
    
    # Uploads
    UPLOAD_FOLDER = os.getenv('UPLOAD_DIR', './uploads')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    
    # Models
    CNN_MODEL_PATH = os.getenv('CNN_MODEL_PATH', 'models/emotion_cnn_model.pkl')
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/yolov8_face.pt')
    ADDITIONAL_MODEL_PATHS = os.getenv('ADDITIONAL_MODEL_PATHS', '').split(',')
    
    # Google
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')

