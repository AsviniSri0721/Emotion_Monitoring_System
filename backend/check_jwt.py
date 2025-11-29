"""
Script to check JWT token configuration and test token creation/validation
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token, decode_token

# Create a test Flask app
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False

jwt = JWTManager(app)

print("=" * 60)
print("JWT Configuration Check")
print("=" * 60)
print(f"JWT_SECRET_KEY from .env: {os.getenv('JWT_SECRET', 'NOT SET')}")
print(f"JWT_SECRET_KEY in app config: {app.config['JWT_SECRET_KEY']}")
print(f"JWT_SECRET_KEY length: {len(app.config['JWT_SECRET_KEY'])}")
print()

# Test token creation
print("Testing token creation...")
test_identity = {
    'id': 'test-user-id',
    'email': 'test@test.com',
    'role': 'teacher'
}

try:
    token = create_access_token(identity=test_identity)
    print(f"✓ Token created successfully")
    print(f"Token length: {len(token)}")
    print(f"Token preview: {token[:50]}...")
    print()
    
    # Test token decoding
    print("Testing token decoding...")
    try:
        decoded = decode_token(token)
        print(f"✓ Token decoded successfully")
        print(f"Decoded identity: {decoded.get('sub', 'N/A')}")
        print(f"Token type: {decoded.get('type', 'N/A')}")
    except Exception as e:
        print(f"✗ Token decoding failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ Token creation failed: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Check Complete")
print("=" * 60)

