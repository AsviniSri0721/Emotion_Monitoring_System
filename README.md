# Emotion Monitoring System

An advanced e-learning platform with real-time emotion detection, behavioral analytics, and adaptive interventions to improve student engagement in online learning.

## Features

### For Teachers
- Upload recorded lecture videos
- Schedule and host live classes (Google Meet integration)
- View detailed engagement and concentration reports for each student
- Monitor emotional patterns and behavioral insights

### For Students
- Join live classes or watch recorded lectures
- Real-time emotion monitoring during sessions
- Automatic re-engagement activities when disengagement is detected (recorded sessions only)

### Core Capabilities
- **Real-time Emotion Detection**: Continuous monitoring via webcam during live and recorded sessions
- **Engagement Analysis**: Tracks boredom, confusion, frustration, sleepiness, focus, and more
- **Adaptive Interventions**: Automatically redirects to re-engagement activities during recorded sessions
- **Comprehensive Reporting**: Detailed analytics on student emotional states and engagement levels

## Tech Stack

- **Frontend**: React + TypeScript
- **Backend**: Python + Flask
- **Database**: MySQL/MariaDB (XAMPP)
- **ML Models**: 
  - CNN models (from .pkl files) for emotion classification
  - YOLOv8 (Ultralytics) for face detection
- **Video**: HTML5 Video API
- **Live Sessions**: Google Meet API

## Quick Start

See [QUICK_START_XAMPP.md](QUICK_START_XAMPP.md) for detailed Windows/XAMPP setup.

### Quick Setup Steps:

1. **Set up database in XAMPP:**
   - Start XAMPP (Apache and MySQL)
   - Open phpMyAdmin: `http://localhost/phpmyadmin`
   - Create database: `emotiondb`
   - Import: `backend/database_schema_mysql.sql`

2. **Set up Python backend:**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   # Or use: install_requirements.bat
   ```

3. **Configure backend:**
   ```bash
   # Run setup script
   setup_env.bat
   # Edit .env if needed (DB_PASSWORD, JWT_SECRET)
   ```

4. **Place your ML models:**
   - `backend/models/emotion_cnn_model.pkl`
   - `backend/models/yolov8_face.pt` (optional)

5. **Set up React frontend:**
   ```bash
   cd client
   npm install
   ```

6. **Start the application:**
   ```bash
   # Terminal 1: Backend
   cd backend
   venv\Scripts\activate
   python app.py
   
   # Terminal 2: Frontend
   cd client
   npm start
   ```

## Project Structure

```
├── backend/         # Python Flask backend (ML models)
├── client/          # React frontend
└── README.md        # This file
```

## Environment Variables

### Backend (backend/.env)
- `DB_HOST`: Database host (default: localhost)
- `DB_NAME`: Database name (default: emotiondb)
- `DB_USER`: Database user (default: root)
- `DB_PASSWORD`: Database password (usually empty for XAMPP)
- `DB_PORT`: Database port (default: 3306)
- `DB_TYPE`: Database type (mysql)
- `JWT_SECRET`: Secret for JWT tokens
- `CNN_MODEL_PATH`: Path to your CNN model
- `YOLO_MODEL_PATH`: Path to your YOLOv8 model

### Client (client/.env)
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:5000/api)

## License

MIT

