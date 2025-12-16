"""
Quick test to verify JWT token validation
This will help diagnose why tokens are failing validation
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Test the JWT_SECRET_KEY
print("=" * 60)
print("JWT Configuration Test")
print("=" * 60)
jwt_secret = os.getenv('JWT_SECRET', 'your-secret-key-change-this')
print(f"JWT_SECRET from .env: {jwt_secret}")
print(f"JWT_SECRET length: {len(jwt_secret)}")
print(f"JWT_SECRET preview: {jwt_secret[:20]}...")
print()

# Test token creation and validation
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token, decode_token

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = jwt_secret
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False

jwt = JWTManager(app)

print("Testing token creation and validation...")
test_identity = {
    'id': 'test-123',
    'email': 'test@test.com',
    'role': 'teacher'
}

try:
    # Create token
    token = create_access_token(identity=test_identity)
    print(f"✓ Token created: {token[:50]}...")
    print()
    
    # Try to decode it
    try:
        decoded = decode_token(token)
        print(f"✓ Token decoded successfully")
        print(f"  Identity: {decoded.get('sub', 'N/A')}")
        print(f"  Token type: {decoded.get('type', 'N/A')}")
        print()
        
        # Verify identity matches
        if decoded.get('sub') == test_identity:
            print("✓ Token identity matches!")
        else:
            print("✗ Token identity mismatch!")
            print(f"  Expected: {test_identity}")
            print(f"  Got: {decoded.get('sub')}")
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
print("Test Complete")
print("=" * 60)
print()
print("If token creation/decoding works here but fails in the app,")
print("the issue might be:")
print("1. JWT_SECRET_KEY changed between token creation and validation")
print("2. Token format issue when sent from frontend")
print("3. Authorization header not being read correctly")














