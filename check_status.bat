@echo off
echo =====================================
echo AI Research Paper Analyzer - Status
echo =====================================
echo.

REM Check GROBID
echo 1. GROBID Server Status:
curl -s http://localhost:8070/api/isalive >nul 2>&1
if %errorlevel% eq 0 (
    echo    [OK] GROBID running at http://localhost:8070
) else (
    echo    [ERROR] GROBID not running - run start_services.bat
)

echo.
echo 2. Backend API Status:
curl -s http://localhost:8002/api/v1/health >nul 2>&1
if %errorlevel% eq 0 (
    echo    [OK] Backend API running at http://localhost:8002
) else (
    echo    [ERROR] Backend not running - run start_services.bat
)

echo.
echo 3. Key URLs:
echo    - API Documentation: http://localhost:8002/docs
echo    - Health Check: http://localhost:8002/api/v1/health
echo    - GROBID Status: http://localhost:8070/api/isalive
echo.

echo 4. Troubleshooting Tips:
echo    - Chat errors usually mean no papers have been analyzed yet
echo    - Upload a PDF via /api/v1/analysis/upload first
echo    - Then use /api/v1/chat/ask with the job_id
echo    - Check logs if services fail to start
echo.

pause