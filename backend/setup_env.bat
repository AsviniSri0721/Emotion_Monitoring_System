@echo off
REM Windows batch script to create .env file for XAMPP setup

echo Creating .env file for XAMPP/MySQL setup...

(
echo # Flask Configuration
echo FLASK_ENV=development
echo PORT=5000
echo SECRET_KEY=your-secret-key-change-this
echo.
echo # JWT Configuration
echo JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
echo.
echo # Database Configuration for XAMPP ^(MySQL/MariaDB^)
echo DB_HOST=localhost
echo DB_NAME=emotiondb
echo DB_USER=root
echo DB_PASSWORD=
echo DB_PORT=3306
echo DB_TYPE=mysql
echo.
echo # Upload Configuration
echo UPLOAD_DIR=./uploads
echo.
echo # Model Paths
echo CNN_MODEL_PATH=models/emotion_cnn_model.pkl
echo YOLO_MODEL_PATH=models/yolov8_face.pt
echo ADDITIONAL_MODEL_PATHS=
echo.
echo # Google Meet ^(optional^)
echo GOOGLE_CLIENT_ID=your-google-client-id
) > .env

echo.
echo .env file created successfully!
echo.
echo Please edit .env and update:
echo   - JWT_SECRET: Change to a secure random string
echo   - DB_PASSWORD: Add password if your MySQL has one
echo   - Model paths if your models are in different locations
echo.
pause

