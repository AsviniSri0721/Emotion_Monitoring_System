"""
Helper functions for JWT token handling
"""
from flask_jwt_extended import get_jwt_identity, get_jwt

def get_current_user():
    """
    Get current user from JWT token.
    Returns a dict with 'id', 'role', and 'email'.
    """
    user_id = get_jwt_identity()  # Returns string ID
    claims = get_jwt()  # Returns dict with additional claims
    
    return {
        'id': user_id,
        'role': claims.get('role'),
        'email': claims.get('email')
    }















