"""
Test script to verify login functionality
Run this to test if the backend can connect to the database and authenticate users
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from services.database import get_connection, execute_query
from werkzeug.security import check_password_hash

def test_database_connection():
    """Test if we can connect to the database"""
    print("=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    try:
        conn = get_connection()
        if conn:
            print("✓ Database connection successful")
            conn.close()
            return True
        else:
            print("✗ Database connection failed - no connection returned")
            return False
    except Exception as e:
        print(f"✗ Database connection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_user_exists(email):
    """Test if a user exists in the database"""
    print("\n" + "=" * 60)
    print(f"Testing if user exists: {email}")
    print("=" * 60)
    
    try:
        user = execute_query(
            "SELECT id, email, password_hash, first_name, last_name, role FROM users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if user:
            print(f"✓ User found:")
            print(f"  ID: {user[0]}")
            print(f"  Email: {user[1]}")
            print(f"  Name: {user[3]} {user[4]}")
            print(f"  Role: {user[5]}")
            return user
        else:
            print(f"✗ User not found: {email}")
            return None
    except Exception as e:
        print(f"✗ Error querying user: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_password_check(email, password):
    """Test password verification"""
    print("\n" + "=" * 60)
    print(f"Testing password verification for: {email}")
    print("=" * 60)
    
    try:
        user = execute_query(
            "SELECT id, email, password_hash FROM users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if not user:
            print(f"✗ User not found: {email}")
            return False
        
        password_valid = check_password_hash(user[2], password)
        
        if password_valid:
            print(f"✓ Password is valid")
            return True
        else:
            print(f"✗ Password is invalid")
            return False
    except Exception as e:
        print(f"✗ Error checking password: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def list_all_users():
    """List all users in the database"""
    print("\n" + "=" * 60)
    print("Listing all users in database")
    print("=" * 60)
    
    try:
        users = execute_query(
            "SELECT id, email, first_name, last_name, role FROM users",
            None,
            fetch_all=True
        )
        
        if users:
            print(f"Found {len(users)} user(s):")
            for user in users:
                print(f"  - {user[1]} ({user[4]}) - ID: {user[0]}")
        else:
            print("No users found in database")
    except Exception as e:
        print(f"✗ Error listing users: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Login Test Script")
    print("=" * 60)
    
    # Test database connection
    if not test_database_connection():
        print("\n✗ Cannot proceed - database connection failed")
        sys.exit(1)
    
    # List all users
    list_all_users()
    
    # Test specific user (change these if needed)
    test_email = input("\nEnter email to test (or press Enter to skip): ").strip()
    if test_email:
        user = test_user_exists(test_email)
        if user:
            test_password = input("Enter password to test: ").strip()
            if test_password:
                test_password_check(test_email, test_password)
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)









