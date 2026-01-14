@echo off
REM Quick script to activate the virtual environment
echo Activating research_env virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo Python location: %VIRTUAL_ENV%
echo.
echo Available commands:
echo   python main.py              - Run main research analyzer
echo   python semantic-document-search/app.py - Run semantic search
echo   cd backend && python main.py - Run FastAPI backend
echo   python -m pip list          - Show installed packages
echo.