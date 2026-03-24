Param(
    [int]$ProxyPort = 8080
)

$ErrorActionPreference = "Stop"

Write-Host "[local-share] Exposing local proxy with Cloudflare Tunnel..." -ForegroundColor Cyan
Write-Host "[local-share] Target: http://localhost:$ProxyPort" -ForegroundColor Gray

function Resolve-CloudflaredPath {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles "cloudflared\cloudflared.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "cloudflared\cloudflared.exe"),
        (Join-Path $env:ProgramFiles "Cloudflare\Cloudflared\cloudflared.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $winGetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $winGetPackages) {
        $fromWinget = Get-ChildItem $winGetPackages -Recurse -Filter cloudflared.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($fromWinget) {
            return $fromWinget.FullName
        }
    }

    return $null
}

$cloudflaredExe = Resolve-CloudflaredPath
if (-not $cloudflaredExe) {
    Write-Host "[local-share] cloudflared is not installed or not in PATH." -ForegroundColor Yellow
    Write-Host "Install: winget install Cloudflare.cloudflared" -ForegroundColor Yellow
    exit 1
}

# Quick Tunnel (no account/domain required). Cloudflare will print a public trycloudflare URL.
& $cloudflaredExe tunnel --url "http://localhost:$ProxyPort"
