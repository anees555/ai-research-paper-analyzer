Param(
    [int]$ProxyPort = 8080
)

$ErrorActionPreference = "Stop"

Write-Host "[local-share] Exposing local proxy at http://localhost:$ProxyPort with ngrok..." -ForegroundColor Cyan

$ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrokCmd) {
    Write-Host "[local-share] ngrok is not installed or not in PATH." -ForegroundColor Yellow
    Write-Host "Install: winget install Ngrok.Ngrok" -ForegroundColor Yellow
    exit 1
}

ngrok http $ProxyPort
