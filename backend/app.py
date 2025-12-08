from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

from routes import auth, videos, sessions, emotions, reports, interventions
from live_sessions import routes as live_sessions_routes
from services.database import init_db
from config_logging import setup_logging

# Import model service only if available (optional for testing without models)
# Note: setup_logging hasn't been called yet, so we use print for errors
try:
    from services.model_service import ModelService
    MODEL_SERVICE_AVAILABLE = True
except ImportError as e:
    MODEL_SERVICE_AVAILABLE = False
    ModelService = None
    # Log the actual import error - use print since logging not set up yet
    import traceback
    error_msg = f"ModelService import failed: {str(e)}"
    tb = traceback.format_exc()
    print(f"\n{'='*60}")
    print("MODELSERVICE IMPORT ERROR:")
    print(f"{'='*60}")
    print(error_msg)
    print(f"\nTraceback:\n{tb}")
    print(f"{'='*60}\n")
except Exception as e:
    MODEL_SERVICE_AVAILABLE = False
    ModelService = None
    import traceback
    error_msg = f"ModelService import error: {str(e)}"
    tb = traceback.format_exc()
    print(f"\n{'='*60}")
    print("MODELSERVICE IMPORT ERROR:")
    print(f"{'='*60}")
    print(error_msg)
    print(f"\nTraceback:\n{tb}")
    print(f"{'='*60}\n")

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
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
         "expose_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "max_age": 3600
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
        setattr(app, 'model_service', model_service)
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Running without ML models - emotion detection will not work")
        setattr(app, 'model_service', None)
else:
    logger.warning("ModelService not available - running without ML models")
    if not MODEL_SERVICE_AVAILABLE:
        logger.error("ModelService import failed - check if all dependencies are installed")
    logger.info("Backend will work for authentication and basic features")
    setattr(app, 'model_service', None)

# Request logging middleware
@app.before_request
def log_request():
    """Log all incoming requests"""
    if request.path.startswith('/api/'):
        # Log Authorization header presence (but not the token itself)
        auth_header = request.headers.get('Authorization', 'None')
        has_token = 'Bearer' in auth_header if auth_header != 'None' else False
        api_logger.info(f"{request.method} {request.path} - IP: {request.remote_addr} - Token: {'Present' if has_token else 'Missing'}")
        
        if request.method in ['POST', 'PUT', 'PATCH']:
            # Log request data (but not passwords)
            try:
                if request.is_json:
                    data = request.get_json() or {}
                    # Don't log password fields
                    safe_data = {k: '***' if 'password' in k.lower() else v for k, v in data.items()}
                    api_logger.debug(f"Request data: {safe_data}")
            except:
                pass  # Ignore errors in logging

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
app.register_blueprint(live_sessions_routes.bp, url_prefix='/api/live-sessions')

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
    error_str = str(error)
    logger.error(f"Invalid token error: {error_str}")
    auth_logger.error(f"Invalid token error: {error_str}")
    # Log request details for debugging
    auth_logger.error(f"Request path: {request.path}")
    auth_logger.error(f"Request method: {request.method}")
    auth_header = request.headers.get('Authorization', 'None')
    has_bearer = 'Bearer' in auth_header if auth_header != 'None' else False
    auth_logger.error(f"Authorization header present: {has_bearer}")
    if has_bearer:
        token_preview = auth_header[:50] + '...' if len(auth_header) > 50 else auth_header
        auth_logger.error(f"Token preview: {token_preview}")
        # Try to decode manually to see the exact error
        try:
            token = auth_header.replace('Bearer ', '').strip()
            from flask_jwt_extended import decode_token
            decoded = decode_token(token)
            auth_logger.error(f"Token decoded successfully: {decoded}")
        except Exception as decode_err:
            auth_logger.error(f"Manual decode failed: {str(decode_err)}")
            auth_logger.error(f"Decode error type: {type(decode_err).__name__}")
            import traceback
            auth_logger.error(f"Decode traceback: {traceback.format_exc()}")
    # Also log JWT_SECRET status (but not the actual secret)
    jwt_secret = os.getenv('JWT_SECRET', 'NOT SET')
    auth_logger.error(f"JWT_SECRET is set: {bool(jwt_secret and jwt_secret != 'NOT SET')}")
    auth_logger.error(f"JWT_SECRET length: {len(jwt_secret) if jwt_secret != 'NOT SET' else 0}")
    return jsonify({
        'error': 'Invalid token', 
        'details': error_str,
        'path': request.path,
        'method': request.method
    }), 422

@jwt.unauthorized_loader
def missing_token_callback(error):
    logger.error(f"Missing token error: {str(error)}")
    auth_logger.error(f"Missing token: {str(error)}")
    auth_logger.error(f"Request path: {request.path}")
    auth_logger.error(f"Request method: {request.method}")
    auth_header = request.headers.get('Authorization', 'None')
    auth_logger.error(f"Authorization header: {auth_header[:50] if auth_header != 'None' and len(auth_header) > 50 else auth_header}")
    return jsonify({'error': 'Authorization token is missing', 'details': str(error)}), 401

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large', 'max_size': '500MB'}), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'details': str(error)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    model_service = getattr(app, 'model_service', None)
    return jsonify({
        'status': 'ok',
        'message': 'Emotion Monitoring System API',
        'models_loaded': model_service.models_loaded if model_service else False
    })

@app.route('/api/test-token', methods=['GET'])
@jwt_required()
def test_token():
    """Test endpoint to verify token validation"""
    try:
        current_user = get_jwt_identity()
        logger.info(f"Test token endpoint - Token validated successfully: {current_user}")
        return jsonify({
            'status': 'success',
            'message': 'Token is valid',
            'user': current_user
        })
    except Exception as e:
        logger.error(f"Token test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Token validation failed',
            'error': str(e)
        }), 422

@app.route('/api/debug-token', methods=['GET'])
def debug_token():
    """Debug endpoint to check token without validation (for troubleshooting)"""
    auth_header = request.headers.get('Authorization', 'None')
    has_bearer = 'Bearer' in auth_header if auth_header != 'None' else False
    
    token_info = {
        'has_authorization_header': auth_header != 'None',
        'has_bearer': has_bearer,
        'header_length': len(auth_header) if auth_header != 'None' else 0,
        'token_preview': auth_header[:50] + '...' if has_bearer and len(auth_header) > 50 else auth_header,
        'jwt_secret_set': bool(os.getenv('JWT_SECRET')),
        'jwt_secret_length': len(os.getenv('JWT_SECRET', '')) if os.getenv('JWT_SECRET') else 0,
        'jwt_secret_preview': os.getenv('JWT_SECRET', '')[:20] + '...' if os.getenv('JWT_SECRET') else 'NOT SET'
    }
    
    # Try to decode the token manually to see what error we get
    if has_bearer:
        try:
            token = auth_header.replace('Bearer ', '').strip()
            from flask_jwt_extended import decode_token
            decoded = decode_token(token)
            token_info['decoded_successfully'] = True
            token_info['decoded_identity'] = decoded.get('sub', 'N/A')
        except Exception as decode_error:
            token_info['decoded_successfully'] = False
            token_info['decode_error'] = str(decode_error)
            token_info['decode_error_type'] = type(decode_error).__name__
    
    return jsonify(token_info)

@app.route('/api/models/status', methods=['GET'])
def model_status():
    model_service = getattr(app, 'model_service', None)
    if not model_service:
        return jsonify({
            'models_loaded': False,
            'message': 'Models not available - running in test mode',
            'cnn_model_loaded': False,
            'yolo_model_loaded': False
        })
    return jsonify({
        'models_loaded': model_service.models_loaded,
        'cnn_model_loaded': model_service.cnn_model is not None,
        'yolo_model_loaded': model_service.yolo_model is not None
    })

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV') == 'development')

