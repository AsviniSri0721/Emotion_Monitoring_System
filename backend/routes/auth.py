from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from services.database import execute_query
import logging
import uuid

bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

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
        token = create_access_token(identity={
            'id': user_id,
            'email': email,
            'role': role
        })
        
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
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        # Find user
        user = execute_query(
            "SELECT id, email, password_hash, first_name, last_name, role FROM users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if not user or not check_password_hash(user[2], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create token
        token = create_access_token(identity={
            'id': str(user[0]),
            'email': user[1],
            'role': user[5]
        })
        
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
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        current_user = get_jwt_identity()
        logger.info(f"Token validated for user: {current_user}")
        
        if not current_user or 'id' not in current_user:
            logger.error("Invalid token payload: missing user id")
            return jsonify({'error': 'Invalid token payload'}), 422
        
        user_id = current_user['id']
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

