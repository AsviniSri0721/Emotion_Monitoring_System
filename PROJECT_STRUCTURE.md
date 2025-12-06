# Emotion Monitoring System - Project Structure

## Project Overview

The Emotion Monitoring System is an advanced e-learning platform with real-time emotion detection, behavioral analytics, and adaptive interventions to improve student engagement in online learning.

**Tech Stack:**
- **Frontend**: React + TypeScript
- **Backend**: Python + Flask
- **Database**: MySQL/MariaDB (XAMPP)
- **ML Models**: CNN models for emotion classification, YOLOv8 for face detection
- **Video**: HTML5 Video API
- **Live Sessions**: Google Meet API

---

## Complete Folder Structure

```
Emotion_Monitoring_System/
├── backend/                          # Python Flask backend
│   ├── __pycache__/                  # Python bytecode cache
│   ├── backend/                      # Nested backend directory
│   │   ├── models/                   # ML model files
│   │   │   ├── mobilenetv2_best.pt
│   │   │   ├── mobilenetv2_emotion_model.pkl
│   │   │   ├── mobilenetv2_processor/
│   │   │   ├── mobilenetv2.onnx
│   │   │   └── yolov8n-face.onnx
│   │   └── uploads/                  # Temporary upload directory
│   ├── logs/                         # Application logs
│   │   ├── api.log
│   │   ├── app.log
│   │   ├── auth.log
│   │   └── errors.log
│   ├── models/                       # Model directory (placeholder)
│   ├── routes/                        # Flask route blueprints
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── emotions.py
│   │   ├── interventions.py
│   │   ├── reports.py
│   │   ├── sessions.py
│   │   └── videos.py
│   ├── services/                     # Business logic services
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── model_service.py
│   ├── uploads/                      # Uploaded video files
│   ├── utils/                        # Utility functions
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   └── jwt_helpers.py
│   ├── venv/                         # Python virtual environment
│   ├── .env                          # Environment variables
│   ├── .gitignore
│   ├── app.py                        # Main Flask application
│   ├── check_and_migrate_db.py
│   ├── check_database.sql
│   ├── check_jwt.py
│   ├── config.py                     # Application configuration
│   ├── config_logging.py             # Logging configuration
│   ├── DATABASE_MIGRATION.md
│   ├── database_schema_mysql.sql
│   ├── DEBUG_LOGIN.md
│   ├── EXPLAIN_JWT.md
│   ├── fix_jwt_token_structure.py
│   ├── generate_test_reports.py
│   ├── install_requirements.bat
│   ├── LOAD_PYTORCH_MODEL.md
│   ├── migrate_database.sql
│   ├── migrate_database_safe.sql
│   ├── pyrightconfig.json
│   ├── QUICK_FIX_JWT.md
│   ├── README.md
│   ├── README_LOGGING.md
│   ├── requirements.txt
│   ├── setup_env.bat
│   ├── test_jwt_validation.py
│   ├── test_login.py
│   └── yolov8n.pt                    # YOLOv8 model file
├── client/                           # React frontend
│   ├── backend/                      # Temporary backend directory
│   │   └── uploads/
│   ├── node_modules/                 # Node.js dependencies
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── EngagementMeter.css
│   │   │   └── EngagementMeter.tsx
│   │   ├── contexts/                 # React contexts
│   │   │   └── AuthContext.tsx
│   │   ├── hooks/                    # Custom React hooks
│   │   │   └── useEmotionStream.ts
│   │   ├── pages/                    # Page components
│   │   │   ├── Auth.css
│   │   │   ├── Dashboard.css
│   │   │   ├── LiveSession.css
│   │   │   ├── LiveSession.tsx
│   │   │   ├── Login.tsx
│   │   │   ├── Register.tsx
│   │   │   ├── ReportPage.css
│   │   │   ├── ReportPage.tsx
│   │   │   ├── StudentDashboard.tsx
│   │   │   ├── TeacherDashboard.tsx
│   │   │   ├── VideoPlayer.css
│   │   │   └── VideoPlayer.tsx
│   │   ├── services/                 # API and service modules
│   │   │   ├── api.ts
│   │   │   └── emotionDetection.ts
│   │   ├── utils/                    # Utility functions
│   │   │   └── logger.ts
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── index.tsx
│   ├── package-lock.json
│   ├── package.json
│   └── tsconfig.json
├── .gitignore
├── DEBUG_422_ERROR.md
├── package.json
├── pyrightconfig.json
├── QUICK_START_XAMPP.md
├── README.md
├── START_PROJECT.bat
├── TEST_LOGIN_GUIDE.md
├── TEST_VIDEO_UPLOAD.md
├── TROUBLESHOOT_UPLOAD.md
└── XAMPP_SETUP.md
```

---

## Detailed File Information

### Root Directory Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `.gitignore` | 325 B | - | Git ignore rules |
| `DEBUG_422_ERROR.md` | 2.9 KB | 123 | Debugging guide for 422 errors |
| `package.json` | 446 B | 19 | Root package.json |
| `pyrightconfig.json` | 328 B | 26 | Pyright TypeScript configuration |
| `QUICK_START_XAMPP.md` | 2.6 KB | 145 | Quick start guide for XAMPP setup |
| `README.md` | 3.2 KB | 112 | Main project README |
| `START_PROJECT.bat` | 762 B | 27 | Batch script to start the project |
| `TEST_LOGIN_GUIDE.md` | 3.1 KB | 122 | Guide for testing login functionality |
| `TEST_VIDEO_UPLOAD.md` | 3.1 KB | 116 | Guide for testing video upload |
| `TROUBLESHOOT_UPLOAD.md` | 2.1 KB | 100 | Troubleshooting guide for uploads |
| `XAMPP_SETUP.md` | 3.0 KB | 152 | XAMPP setup instructions |

---

### Backend Directory

#### Core Application Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `app.py` | 11.6 KB | 286 | Main Flask application entry point. Initializes Flask app, registers blueprints, sets up JWT, CORS, and model service |
| `config.py` | 1.1 KB | 35 | Application configuration class. Loads environment variables for Flask, JWT, database, uploads, and models |
| `config_logging.py` | 3.4 KB | 110 | Logging configuration. Sets up rotating file handlers for app, errors, API, and auth logs |

#### Database Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `database_schema_mysql.sql` | 5.1 KB | 118 | Complete MySQL database schema. Defines tables for users, videos, sessions, emotions, interventions, and reports |
| `check_database.sql` | 2.4 KB | 72 | SQL queries to check database structure and data |
| `migrate_database.sql` | 1.9 KB | 48 | Database migration script |
| `migrate_database_safe.sql` | 2.9 KB | 78 | Safe database migration script with error handling |
| `check_and_migrate_db.py` | 5.9 KB | 155 | Python script to check and migrate database schema |

#### Routes (Flask Blueprints)

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `routes/__init__.py` | 18 B | 2 | Routes package initialization |
| `routes/auth.py` | 7.0 KB | 200 | Authentication routes: register, login, logout, user profile management |
| `routes/videos.py` | 9.3 KB | 236 | Video management routes: upload, list, get, delete videos |
| `routes/sessions.py` | 8.6 KB | 234 | Session management routes: create, get, update live and recorded sessions |
| `routes/emotions.py` | 13.1 KB | 311 | Emotion detection routes: detect emotions from images, get emotion history |
| `routes/reports.py` | 12.0 KB | 257 | Reporting routes: get engagement reports, dashboard data, student analytics |
| `routes/interventions.py` | 4.5 KB | 123 | Intervention routes: trigger interventions, get intervention history |

#### Services

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `services/__init__.py` | 20 B | 2 | Services package initialization |
| `services/database.py` | 5.3 KB | 159 | Database service. Handles connection pooling, query execution for MySQL/MariaDB |
| `services/model_service.py` | 79.7 KB | 1436 | ML model service. Loads and runs emotion detection models (ONNX, PyTorch), face detection with YOLOv8 |

#### Utils

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `utils/__init__.py` | 51 B | 12 | Utils package initialization |
| `utils/jwt_helpers.py` | 479 B | 27 | JWT helper functions. Extracts user information from JWT tokens |

#### Configuration & Setup Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `.env` | 578 B | - | Environment variables (database, JWT, model paths) |
| `requirements.txt` | 671 B | 30 | Python dependencies: Flask, JWT, OpenCV, NumPy, ONNX Runtime, etc. |
| `pyrightconfig.json` | 303 B | 26 | Pyright configuration for type checking |
| `setup_env.bat` | 1.1 KB | 44 | Batch script to set up environment variables |
| `install_requirements.bat` | 1.8 KB | 64 | Batch script to install Python dependencies |

#### Testing & Debugging Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `test_login.py` | 4.2 KB | 156 | Test script for login functionality |
| `test_jwt_validation.py` | 2.2 KB | 88 | Test script for JWT token validation |
| `check_jwt.py` | 1.7 KB | 73 | Script to check JWT token structure |
| `fix_jwt_token_structure.py` | 1.1 KB | 35 | Script to fix JWT token structure issues |
| `generate_test_reports.py` | 7.8 KB | 206 | Script to generate test engagement reports |

#### Documentation Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `README.md` | 1.7 KB | 75 | Backend README with setup instructions |
| `README_LOGGING.md` | 3.9 KB | 173 | Logging system documentation |
| `DATABASE_MIGRATION.md` | 3.5 KB | 126 | Database migration guide |
| `DEBUG_LOGIN.md` | 3.1 KB | 143 | Debugging guide for login issues |
| `EXPLAIN_JWT.md` | 2.0 KB | 64 | JWT token explanation and usage |
| `QUICK_FIX_JWT.md` | 2.0 KB | 87 | Quick fix guide for JWT issues |
| `LOAD_PYTORCH_MODEL.md` | 1.6 KB | 67 | Guide for loading PyTorch models |

#### Model Files

| File | Size | Description |
|------|------|-------------|
| `yolov8n.pt` | 6.5 MB | YOLOv8 face detection model (PyTorch format) |
| `backend/models/mobilenetv2_best.pt` | 9.2 MB | MobileNetV2 emotion model (PyTorch format) |
| `backend/models/mobilenetv2_emotion_model.pkl` | 9.2 MB | MobileNetV2 emotion model (Pickle format) |
| `backend/models/mobilenetv2.onnx` | 9.1 MB | MobileNetV2 emotion model (ONNX format) |
| `backend/models/yolov8n-face.onnx` | 12.3 MB | YOLOv8 face detection model (ONNX format) |
| `backend/models/mobilenetv2_processor/` | - | MobileNetV2 processor directory |

#### Log Files

| File | Size | Description |
|------|------|-------------|
| `logs/api.log` | 652 KB | API request/response logs |
| `logs/app.log` | 184 KB | General application logs |
| `logs/auth.log` | 133 KB | Authentication-related logs |
| `logs/errors.log` | 31 KB | Error logs |

#### Upload Files

| File | Size | Description |
|------|------|-------------|
| `uploads/1758266b-f7ed-49b0-991f-5ba9840741cd.mp4` | 248.4 MB | Uploaded video file |
| `uploads/c155657f-7209-40ee-a5d8-7f316bbb814c.mp4` | 264 KB | Uploaded video file |

---

### Client Directory

#### Configuration Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `package.json` | 1.1 KB | 50 | Node.js dependencies and scripts. Includes React, TypeScript, Axios, face-api.js, TensorFlow.js, Recharts |
| `package-lock.json` | 714 KB | 19,128 | Locked dependency versions |
| `tsconfig.json` | 539 B | 27 | TypeScript compiler configuration |

#### Public Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `public/index.html` | 456 B | 15 | HTML entry point for React app |

#### Source Files - Core

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/index.tsx` | 274 B | 14 | React application entry point. Renders App component |
| `src/index.css` | 443 B | 20 | Global CSS styles |
| `src/App.tsx` | 4.1 KB | 137 | Main App component. Sets up routing, authentication context, and protected routes |
| `src/App.css` | 1.8 KB | 123 | App component styles |

#### Components

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/components/EngagementMeter.tsx` | 2.4 KB | 81 | Circular engagement meter component. Displays concentration score and current emotion |
| `src/components/EngagementMeter.css` | 594 B | 35 | Engagement meter styles |

#### Contexts

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/contexts/AuthContext.tsx` | 8.4 KB | 243 | Authentication context. Manages user state, login, register, logout, and token management |

#### Hooks

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/hooks/useEmotionStream.ts` | 14.4 KB | 402 | Custom hook for emotion detection stream. Handles video frame capture, emotion detection API calls, and real-time updates |

#### Pages

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/pages/Login.tsx` | 2.4 KB | 76 | Login page component |
| `src/pages/Register.tsx` | 3.6 KB | 116 | Registration page component |
| `src/pages/Auth.css` | 896 B | 56 | Authentication pages styles |
| `src/pages/StudentDashboard.tsx` | 4.6 KB | 149 | Student dashboard. Shows available videos and live sessions |
| `src/pages/TeacherDashboard.tsx` | 14.5 KB | 416 | Teacher dashboard. Manages videos, live sessions, and views reports |
| `src/pages/Dashboard.css` | 2.2 KB | 155 | Dashboard styles |
| `src/pages/VideoPlayer.tsx` | 14.9 KB | 425 | Video player page. Plays recorded videos with real-time emotion detection |
| `src/pages/VideoPlayer.css` | 3.4 KB | 219 | Video player styles |
| `src/pages/LiveSession.tsx` | 9.5 KB | 277 | Live session page. Integrates with Google Meet and monitors emotions |
| `src/pages/LiveSession.css` | 2.3 KB | 149 | Live session styles |
| `src/pages/ReportPage.tsx` | 7.7 KB | 229 | Report page. Displays detailed engagement and emotion analytics |
| `src/pages/ReportPage.css` | 1.7 KB | 118 | Report page styles |

#### Services

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/services/api.ts` | 3.2 KB | 94 | Axios API client. Configures base URL, request/response interceptors, and error handling |
| `src/services/emotionDetection.ts` | 3.5 KB | 122 | Emotion detection service. Handles video frame capture and emotion detection API calls |

#### Utils

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `src/utils/logger.ts` | 3.5 KB | 151 | Frontend logging utility. Logs to console and localStorage with log export functionality |

---

## File Statistics Summary

### Backend
- **Total Python Files**: 21
- **Total Lines of Code**: ~4,500+ lines
- **Largest File**: `services/model_service.py` (1,436 lines, 79.7 KB)
- **Model Files**: 5 files totaling ~46 MB
- **Log Files**: 4 files totaling ~1 MB

### Frontend
- **Total TypeScript/TSX Files**: 15
- **Total CSS Files**: 8
- **Total Lines of Code**: ~3,000+ lines
- **Largest File**: `pages/VideoPlayer.tsx` (425 lines, 14.9 KB)
- **Dependencies**: 19,128 lines in package-lock.json

### Documentation
- **Total Markdown Files**: 12
- **Total Documentation**: ~1,500+ lines

---

## Key Features by Component

### Backend Features
- **Authentication**: JWT-based authentication with role-based access control
- **Video Management**: Upload, store, and serve video files
- **Emotion Detection**: Real-time emotion detection using ML models
- **Session Management**: Live and recorded session tracking
- **Reporting**: Comprehensive engagement and emotion analytics
- **Interventions**: Automatic and manual intervention triggers

### Frontend Features
- **User Authentication**: Login and registration with role-based routing
- **Video Playback**: HTML5 video player with emotion detection overlay
- **Live Sessions**: Google Meet integration with real-time monitoring
- **Dashboards**: Separate dashboards for teachers and students
- **Reports**: Visual analytics with charts and graphs
- **Real-time Updates**: WebSocket-like emotion streaming

---

## Environment Variables

### Backend (.env)
- `DB_HOST`: Database host (default: localhost)
- `DB_NAME`: Database name (default: emotiondb)
- `DB_USER`: Database user (default: root)
- `DB_PASSWORD`: Database password
- `DB_PORT`: Database port (default: 3306)
- `DB_TYPE`: Database type (mysql)
- `JWT_SECRET`: Secret key for JWT tokens
- `CNN_MODEL_PATH`: Path to CNN emotion model
- `YOLO_MODEL_PATH`: Path to YOLOv8 face detection model
- `UPLOAD_DIR`: Directory for uploaded videos
- `GOOGLE_CLIENT_ID`: Google OAuth client ID (for Google Meet)

### Frontend (.env)
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:5000/api)

---

## Database Schema

The database includes the following main tables:
- `users`: User accounts (teachers and students)
- `videos`: Recorded lecture videos
- `live_sessions`: Scheduled and active live sessions
- `session_participants`: Participants in sessions
- `emotion_detections`: Emotion detection results
- `interventions`: Intervention records
- `engagement_reports`: Aggregated engagement reports

See `backend/database_schema_mysql.sql` for complete schema.

---

## Notes

- Model files are stored in `backend/backend/models/` (nested directory structure)
- Log files are automatically rotated (max 10MB, 5 backups)
- Uploaded videos are stored in `backend/uploads/`
- The virtual environment (`venv/`) should not be committed to version control
- Node modules (`node_modules/`) should not be committed to version control

---

*Last Updated: Generated from current project structure*

