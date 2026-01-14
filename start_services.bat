@echo off
echo Starting AI Research Paper Analyzer Services...

REM Check if GROBID container exists
docker ps -a --format "table {{.Names}}" | findstr grobid >nul
if %errorlevel% neq 0 (
    echo 🚀 Starting GROBID server...
    docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
    echo ⏳ Waiting for GROBID to start...
    timeout /t 30 /nobreak >nul
) else (
    echo 🔄 Starting existing GROBID container...
    docker start grobid
    timeout /t 10 /nobreak >nul
)

REM Check GROBID status
curl -s http://localhost:8070/api/isalive >nul 2>&1
if %errorlevel% eq 0 (
    echo ✅ GROBID server is running at http://localhost:8070
) else (
    echo ❌ GROBID server failed to start
)

echo.
echo 🚀 Starting Backend API Server...
echo 📖 API Documentation will be available at: http://localhost:8002/docs
echo 🔍 Health Check available at: http://localhost:8002/api/v1/health
echo.
echo Press Ctrl+C to stop the backend server
echo.

REM Activate virtual environment and start backend
call .venv\Scripts\activate.bat
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8002

pause