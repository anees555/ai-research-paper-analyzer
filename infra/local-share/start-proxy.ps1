Param(
    [int]$ProxyPort = 8080,
    [string]$ConfigPath = "./Caddyfile"
)

$ErrorActionPreference = "Stop"

Write-Host "[local-share] Starting local reverse proxy..." -ForegroundColor Cyan
Write-Host "[local-share] Frontend: http://localhost:3000" -ForegroundColor Gray
Write-Host "[local-share] Backend : http://localhost:8003" -ForegroundColor Gray
Write-Host "[local-share] Proxy   : http://localhost:$ProxyPort" -ForegroundColor Green

function Resolve-CaddyPath {
    $cmd = Get-Command caddy -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles "Caddy\caddy.exe"),
        (Join-Path $env:ProgramFiles "caddy\caddy.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $winGetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $winGetPackages) {
        $fromWinget = Get-ChildItem $winGetPackages -Recurse -Filter caddy.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($fromWinget) {
            return $fromWinget.FullName
        }
    }

    return $null
}

$caddyExe = Resolve-CaddyPath
if (-not $caddyExe) {
    Write-Host "[local-share] Caddy is not installed or not in PATH." -ForegroundColor Yellow
    Write-Host "Install: winget install CaddyServer.Caddy" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $ConfigPath)) {
    Write-Host "[local-share] Config not found: $ConfigPath" -ForegroundColor Red
    exit 1
}

& $caddyExe run --config $ConfigPath --adapter caddyfile
