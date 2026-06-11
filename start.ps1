# start.ps1 - Startet Backend (FastAPI) und Frontend (Vite) zusammen.
# Aufruf:  ./start.ps1   (ggf. zuerst:  Set-ExecutionPolicy -Scope Process Bypass)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "==> Python-Abhaengigkeiten (uv sync)..." -ForegroundColor Cyan
uv sync

if (-not (Test-Path (Join-Path $root "web/node_modules"))) {
    Write-Host "==> Frontend-Abhaengigkeiten (npm install)..." -ForegroundColor Cyan
    Push-Location (Join-Path $root "web")
    npm install
    Pop-Location
}

Write-Host "==> Starte Backend auf http://127.0.0.1:8000 ..." -ForegroundColor Green
$backend = Start-Process -FilePath "uv" `
    -ArgumentList "run", "python", "-m", "f1_rl.server" `
    -WorkingDirectory $root -PassThru -NoNewWindow

Write-Host "==> Starte Frontend (Vite) ..." -ForegroundColor Green
$frontend = Start-Process -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $root "web") -PassThru -NoNewWindow

Write-Host ""
Write-Host "Beide laufen. Frontend ueblicherweise auf http://localhost:5173" -ForegroundColor Yellow
Write-Host "Mit STRG+C beenden." -ForegroundColor Yellow

# Aufraeumen: beide Prozesse beenden, wenn das Skript endet (STRG+C)
try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
