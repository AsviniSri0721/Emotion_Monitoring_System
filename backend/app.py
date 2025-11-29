from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

from routes import auth, videos, sessions, emotions, reports, interventions
from services.database import init_db
from config_logging import setup_logging

# Import model service only if available (optional for testing without models)
try:
    from services.model_service import ModelService
    MODEL_SERVICE_AVAILABLE = True
except ImportError:
    MODEL_SERVICE_AVAILABLE = False
    ModelService = None

load_dotenv()

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_DIR', './uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize extensions
jwt = JWTManager(app)
CORS(app, 
     resources={r"/api/*": {
         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": True
     }},
     supports_credentials=True,
     automatic_options=True)

# Setup comprehensive logging
logger = setup_logging(app)
api_logger = logging.getLogger('api')
auth_logger = logging.getLogger('auth')

# Initialize database
init_db()

# Initialize ML models (optional - only if ModelService is available)
if MODEL_SERVICE_AVAILABLE and ModelService:
    try:
        model_service = ModelService()
        model_service.load_models()
        app.model_service = model_service
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {str(e)}")
        logger.warning("Running without ML models - emotion detection will not work")
        app.model_service = None
else:
    logger.warning("ModelService not available - running without ML models")
    logger.info("Backend will work for authentication and basic features")
    app.model_service = None

# Request logging middleware
@app.before_request
def log_request():
    """Log all incoming requests"""
    if request.path.startswith('/api/'):
        api_logger.info(f"{request.method} {request.path} - IP: {request.remote_addr}")
        if request.method in ['POST', 'PUT', 'PATCH']:
            # Log request data (but not passwords)
            if request.is_json:
                data = request.get_json() or {}
                # Don't log password fields
                safe_data = {k: '***' if 'password' in k.lower() else v for k, v in data.items()}
                api_logger.debug(f"Request data: {safe_data}")

@app.after_request
def log_response(response):
    """Log all responses"""
    if request.path.startswith('/api/'):
        api_logger.info(f"{request.method} {request.path} - Status: {response.status_code}")
    return response

# Error logging
@app.errorhandler(Exception)
def handle_exception(e):
    """Log all exceptions"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

# Register blueprints
app.register_blueprint(auth.bp, url_prefix='/api/auth')
app.register_blueprint(videos.bp, url_prefix='/api/videos')
app.register_blueprint(sessions.bp, url_prefix='/api/sessions')
app.register_blueprint(emotions.bp, url_prefix='/api/emotions')
app.register_blueprint(reports.bp, url_prefix='/api/reports')
app.register_blueprint(interventions.bp, url_prefix='/api/interventions')

# Serve uploaded videos
@app.route('/uploads/<filename>')
def serve_video(filename):
    """Serve video files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token', 'details': str(error)}), 422

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token is missing'}), 401

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large', 'max_size': '500MB'}), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'details': str(error)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Emotion Monitoring System API',
        'models_loaded': app.model_service.models_loaded if app.model_service else False
    })

@app.route('/api/models/status', methods=['GET'])
def model_status():
    if not app.model_service:
        return jsonify({
            'models_loaded': False,
            'message': 'Models not available - running in test mode',
            'cnn_model_loaded': False,
            'yolo_model_loaded': False
        })
    return jsonify({
        'models_loaded': app.model_service.models_loaded,
        'cnn_model_loaded': app.model_service.cnn_model is not None,
        'yolo_model_loaded': app.model_service.yolo_model is not None
    })

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV') == 'development')

