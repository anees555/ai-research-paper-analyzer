@echo off
echo Stopping AI Research Paper Analyzer Services...

echo 🛑 Stopping Backend Server...
REM Kill any uvicorn processes
taskkill /f /im python.exe >nul 2>&1

echo 🛑 Stopping GROBID Server...
docker stop grobid >nul 2>&1

echo ✅ All services stopped.
pause