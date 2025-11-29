from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from werkzeug.security import generate_password_hash, check_password_hash
from services.database import execute_query
import logging
import uuid

bp = Blueprint('auth', __name__)
logger = logging.getLogger('auth')  # Use auth logger

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

@bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('firstName')
        last_name = data.get('lastName')
        role = data.get('role')
        
        # Check if user exists
        existing = execute_query(
            "SELECT id FROM users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if existing:
            return jsonify({'error': 'User already exists'}), 400
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Generate UUID for user
        user_id = generate_uuid_str()
        
        # Create user
        execute_query(
            """INSERT INTO users (id, email, password_hash, first_name, last_name, role)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (user_id, email, password_hash, first_name, last_name, role)
        )
        
        # Create token
        # Flask-JWT-Extended requires identity to be a string
        # Store additional user data in additional_claims
        token = create_access_token(
            identity=str(user_id),  # User ID as string
            additional_claims={
                'email': email,
                'role': role
            }
        )
        
        return jsonify({
            'token': token,
            'user': {
                'id': user_id,
                'email': email,
                'firstName': first_name,
                'lastName': last_name,
                'role': role
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@bp.route('/login', methods=['POST'])
def login():
    try:
        logger.info("=" * 50)
        logger.info("LOGIN REQUEST RECEIVED")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Request method: {request.method}")
        
        data = request.get_json()
        if not data:
            logger.error("No JSON data in request")
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email')
        password = data.get('password')
        
        logger.info(f"Login attempt for email: {email}")
        
        if not email or not password:
            logger.warning("Missing email or password")
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        logger.info(f"Querying database for user: {email}")
        user = execute_query(
            "SELECT id, email, password_hash, first_name, last_name, role FROM users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if not user:
            logger.warning(f"User not found: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        logger.info(f"User found: {user[1]} (ID: {user[0]})")
        
        # Check password
        password_valid = check_password_hash(user[2], password)
        logger.info(f"Password check result: {password_valid}")
        
        if not password_valid:
            logger.warning(f"Invalid password for user: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create token
        # Flask-JWT-Extended requires identity to be a string
        # Store additional user data in additional_claims
        token = create_access_token(
            identity=str(user[0]),  # User ID as string
            additional_claims={
                'email': user[1],
                'role': user[5]
            }
        )
        
        logger.info(f"Token created successfully for user: {user[1]} (Role: {user[5]})")
        logger.info("=" * 50)
        
        return jsonify({
            'token': token,
            'user': {
                'id': str(user[0]),
                'email': user[1],
                'firstName': user[3],
                'lastName': user[4],
                'role': user[5]
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Login failed', 'details': str(e)}), 500

@bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        logger.info("=" * 50)
        logger.info("GET /api/auth/me REQUEST RECEIVED")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Authorization header present: {'Authorization' in request.headers}")
        
        auth_header = request.headers.get('Authorization', 'None')
        if auth_header != 'None':
            token_preview = auth_header[:50] + '...' if len(auth_header) > 50 else auth_header
            logger.info(f"Token preview: {token_preview}")
        
        # Get user ID from identity (which is now a string)
        user_id = get_jwt_identity()
        # Get additional claims (email, role)
        claims = get_jwt()
        
        logger.info(f"Token validated successfully for user ID: {user_id}")
        
        if not user_id:
            logger.error("Invalid token payload: missing user id")
            return jsonify({'error': 'Invalid token payload', 'details': 'Missing user id in token'}), 422
        logger.info(f"Fetching user from database: {user_id}")
        
        user = execute_query(
            "SELECT id, email, first_name, last_name, role, created_at FROM users WHERE id = %s",
            (user_id,),
            fetch_one=True
        )
        
        if not user:
            logger.error(f"User not found in database: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        logger.info(f"User found: {user[1]}")
        return jsonify({
            'id': str(user[0]),
            'email': user[1],
            'firstName': user[2],
            'lastName': user[3],
            'role': user[4],
            'createdAt': user[5].isoformat() if user[5] else None
        })
        
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch user', 'details': str(e)}), 500

