# Quick script to activate the virtual environment  
Write-Host "Activating research_env virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1
Write-Host ""
Write-Host "Virtual environment activated!" -ForegroundColor Green  
Write-Host "Python location: $env:VIRTUAL_ENV" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  python main.py                            - Run main research analyzer" -ForegroundColor White
Write-Host "  python semantic-document-search/app.py     - Run semantic search" -ForegroundColor White  
Write-Host "  cd backend; python main.py                - Run FastAPI backend" -ForegroundColor White
Write-Host "  python -m pip list                        - Show installed packages" -ForegroundColor White
Write-Host ""