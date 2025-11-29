# Quick Start Guide for XAMPP

## Step 1: Database Setup (5 minutes)

1. **Start XAMPP**
   - Open XAMPP Control Panel
   - Start **Apache** and **MySQL**

2. **Create Database in phpMyAdmin**
   - Go to: `http://localhost/phpmyadmin`
   - Click "New" in left sidebar
   - Database name: `emotiondb`
   - Collation: `utf8mb4_unicode_ci`
   - Click "Create"

3. **Import Schema**
   - Select `emotiondb` database
   - Click "Import" tab
   - Choose file: `backend/database_schema_mysql.sql`
   - Click "Go"

✅ You should now see 7 tables created!

## Step 2: Python Backend Setup (5 minutes)

1. **Install Python Dependencies**
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
pip install pymysql  # For MySQL support
```

2. **Configure Environment**
```bash
# Copy example file
cp .env.example .env

# Edit .env file with these settings:
DB_HOST=localhost
DB_NAME=emotiondb
DB_USER=root
DB_PASSWORD=          # Leave empty if no password
DB_PORT=3306
DB_TYPE=mysql
```

3. **Place Your ML Models**
```bash
mkdir -p models
# Place your models:
# - models/emotion_cnn_model.pkl
# - models/yolov8_face.pt (optional)
```

4. **Start Backend**
```bash
python app.py
```

✅ Backend should be running at `http://localhost:5000`

## Step 3: Test Database Connection

```bash
# Check health
curl http://localhost:5000/api/health

# Check models
curl http://localhost:5000/api/models/status
```

## Step 4: React Frontend Setup

1. **Install Dependencies**
```bash
cd client
npm install
```

2. **Configure Environment**
```bash
# Edit client/.env
REACT_APP_API_URL=http://localhost:5000/api
```

3. **Start Frontend**
```bash
npm start
```

✅ Frontend should be running at `http://localhost:3000`

## Troubleshooting

### "Access denied for user 'root'"
- Check if MySQL has a password in XAMPP
- Update `DB_PASSWORD` in `backend/.env`

### "Unknown database 'emotiondb'"
- Make sure you created the database in phpMyAdmin
- Check spelling in `.env`

### "Module 'pymysql' not found"
```bash
pip install pymysql
```

### Port 5000 already in use
- Change `PORT` in `backend/.env`
- Or stop the process using port 5000

## Next Steps

1. ✅ Database created and imported
2. ✅ Backend running
3. ✅ Frontend running
4. Place your trained models
5. Test the system!

## Default XAMPP Settings

- **Host**: `localhost`
- **Port**: `3306`
- **Username**: `root`
- **Password**: (usually empty)

Your system is now ready! 🎉

