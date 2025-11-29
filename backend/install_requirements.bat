@echo off
REM Windows batch script to install Python requirements with proper setup

echo Installing Python dependencies for Windows...
echo.

REM Upgrade pip, setuptools, and wheel first
echo [1/3] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip/setuptools
    pause
    exit /b 1
)

echo.
echo [2/3] Installing core dependencies...
python -m pip install flask flask-cors flask-jwt-extended python-dotenv werkzeug
if errorlevel 1 (
    echo ERROR: Failed to install Flask dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Installing scientific computing libraries...
python -m pip install numpy opencv-python Pillow scikit-learn
if errorlevel 1 (
    echo ERROR: Failed to install scientific libraries
    pause
    exit /b 1
)

echo.
echo [4/4] Installing database driver...
python -m pip install pymysql
if errorlevel 1 (
    echo ERROR: Failed to install pymysql
    pause
    exit /b 1
)

echo.
echo [Optional] Installing deep learning libraries (large files, may take time)...
echo You can skip this if you don't have your models ready yet.
set /p install_dl="Install PyTorch and Ultralytics? (y/n): "
if /i "%install_dl%"=="y" (
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python -m pip install ultralytics
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure XAMPP MySQL is running
echo 2. Import database_schema_mysql.sql into phpMyAdmin
echo 3. Place your ML models in backend/models/
echo 4. Run: python app.py
echo.
pause

