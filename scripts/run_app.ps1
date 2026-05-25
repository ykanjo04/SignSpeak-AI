# Build the Next.js frontend and start SignSpeak AI on a single port.
param(
    [int]$Port = 8000,
    [string]$ListenHost = "127.0.0.1",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Error "Python venv not found. Run: python -m venv .venv && pip install -r backend\requirements.txt"
}

Write-Host "==> Activating venv"
. .\.venv\Scripts\Activate.ps1

if (-not $SkipBuild) {
    Write-Host "==> Building frontend (static export -> frontend/out)"
    Push-Location frontend
    if (-not (Test-Path node_modules)) {
        npm install
    }
    npm run build
    Pop-Location
}

Write-Host "==> Starting SignSpeak AI at http://${ListenHost}:${Port}"
Write-Host "    Live UI:  http://${ListenHost}:${Port}/live/"
Write-Host "    API docs: http://${ListenHost}:${Port}/docs"
Set-Location backend
uvicorn app.main:app --host $ListenHost --port $Port --reload
