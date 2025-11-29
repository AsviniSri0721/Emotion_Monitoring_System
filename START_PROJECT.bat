@echo off
REM Start both backend and frontend servers

echo ========================================
echo Starting Emotion Monitoring System
echo ========================================
echo.

echo [1/2] Starting Python Backend...
start "Backend Server" cmd /k "cd backend && venv\Scripts\activate && python app.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting React Frontend...
start "Frontend Server" cmd /k "cd client && npm start"

echo.
echo ========================================
echo Servers are starting...
echo ========================================
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window (servers will keep running)...
pause >nul

