# XAMPP Setup Guide

This guide helps you set up the database in XAMPP/phpMyAdmin.

## Step 1: Create Database in phpMyAdmin

1. Open phpMyAdmin: `http://localhost/phpmyadmin`
2. Click on "New" in the left sidebar
3. Database name: `emotiondb`
4. Collation: `utf8mb4_unicode_ci`
5. Click "Create"

## Step 2: Import Database Schema

### Option A: Import SQL File (Recommended)

1. In phpMyAdmin, select the `emotiondb` database
2. Click on the "Import" tab
3. Click "Choose File" and select: `backend/database_schema_mysql.sql`
4. Click "Go" to import

### Option B: Copy-Paste SQL

1. Open `backend/database_schema_mysql.sql` in a text editor
2. Copy all the SQL code
3. In phpMyAdmin, select `emotiondb` database
4. Click on the "SQL" tab
5. Paste the SQL code
6. Click "Go"

## Step 3: Verify Tables Created

After importing, you should see these tables:
- ✅ `users`
- ✅ `videos`
- ✅ `live_sessions`
- ✅ `session_participants`
- ✅ `emotion_data`
- ✅ `engagement_reports`
- ✅ `interventions`

## Step 4: Configure Backend

1. Copy the environment file:
```bash
cd backend
cp .env.example .env
```

2. Edit `backend/.env`:
```env
# Database Configuration for XAMPP
DB_HOST=localhost
DB_NAME=emotiondb
DB_USER=root
DB_PASSWORD=          # Leave empty if no password set
DB_PORT=3306
DB_TYPE=mysql
```

3. Install MySQL driver:
```bash
pip install pymysql
# OR
pip install mysql-connector-python
```

## Step 5: Test Database Connection

1. Start XAMPP (Apache and MySQL)
2. Start the Flask backend:
```bash
cd backend
python app.py
```

3. Check if database connection works:
```bash
curl http://localhost:5000/api/health
```

## Troubleshooting

### "Access denied for user 'root'@'localhost'"

- Check if MySQL password is set in XAMPP
- Update `DB_PASSWORD` in `.env` if needed
- Or reset MySQL password in XAMPP

### "Unknown database 'emotiondb'"

- Make sure you created the database in phpMyAdmin
- Check database name spelling in `.env`

### "Table doesn't exist"

- Import the SQL schema again
- Check if all tables were created successfully

### "Module 'pymysql' not found"

```bash
pip install pymysql
```

### Port 3306 Already in Use

- Check if MySQL is running in XAMPP
- Stop other MySQL services if running
- Restart XAMPP MySQL

## Default XAMPP MySQL Settings

- **Host**: `localhost`
- **Port**: `3306`
- **Username**: `root`
- **Password**: (usually empty, but check your XAMPP setup)

## Next Steps

1. ✅ Database created and schema imported
2. ✅ Backend configured
3. ✅ Test connection
4. Place your ML models in `backend/models/`
5. Start the application!

## Quick Test

After setup, you can test by creating a user:

```sql
INSERT INTO users (id, email, password_hash, first_name, last_name, role)
VALUES (
    UUID(),
    'test@example.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyY5Y5Y5Y5Y5',
    'Test',
    'User',
    'student'
);
```

Or use the registration API endpoint once the backend is running.








