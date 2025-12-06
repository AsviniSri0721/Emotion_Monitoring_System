# Understanding JWT Tokens

## Important: JWT Tokens Are NOT Stored in the Database

JWT (JSON Web Tokens) are **stateless** authentication tokens. This means:

1. **They are NOT stored in the database** - This is by design and is actually a feature, not a bug!
2. **They contain all necessary information** - The token itself contains the user ID, email, role, etc.
3. **They are validated using a secret key** - The backend uses `JWT_SECRET_KEY` to verify the token is valid

## How JWT Works

### When You Login:
1. Backend validates your email/password against the database
2. Backend creates a JWT token containing: `{id, email, role}`
3. Backend signs the token with `JWT_SECRET_KEY`
4. Backend sends the token to the frontend
5. **Token is NOT saved to database** - it's just sent to the client

### When You Make API Requests:
1. Frontend sends the token in the `Authorization: Bearer <token>` header
2. Backend receives the token
3. Backend validates the token signature using `JWT_SECRET_KEY`
4. If valid, backend extracts user info from the token (no database lookup needed!)
5. If invalid, backend returns 401/422 error

## Why 422 Errors Are Happening

The 422 errors mean the token **validation is failing**. This could be because:

1. **JWT_SECRET_KEY mismatch** - Token was created with one key, but validated with a different key
2. **Token format issue** - Token is malformed or corrupted
3. **Token not being sent correctly** - Authorization header missing or incorrect

## How to Fix

The token validation is failing. Check:

1. **Backend logs** - Look for "Invalid token error" messages
2. **JWT_SECRET_KEY** - Make sure it's the same value in `.env` file
3. **Token in browser** - Check if token exists in localStorage after login

## Testing

Run this to check JWT configuration:
```bash
cd backend
venv\Scripts\activate
python check_jwt.py
```

This will verify:
- JWT_SECRET_KEY is set correctly
- Token creation works
- Token validation works








