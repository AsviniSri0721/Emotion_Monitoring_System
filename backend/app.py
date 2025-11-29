from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
from dotenv import load_dotenv
import logging

from routes import auth, videos, sessions, emotions, reports, interventions
from services.database import init_db

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
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

