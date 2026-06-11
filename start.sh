#!/usr/bin/env bash
# start.sh - Startet Backend (FastAPI) und Frontend (Vite) zusammen.
# Aufruf:  ./start.sh   (ggf. zuerst:  chmod +x start.sh)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

echo "==> Python-Abhaengigkeiten (uv sync)..."
uv sync

if [ ! -d "web/node_modules" ]; then
    echo "==> Frontend-Abhaengigkeiten (npm install)..."
    (cd web && npm install)
fi

# Beide Kindprozesse beenden, wenn das Skript endet (STRG+C)
cleanup() {
    [ -n "${backend:-}" ] && kill "$backend" 2>/dev/null || true
    [ -n "${frontend:-}" ] && kill "$frontend" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starte Backend auf http://127.0.0.1:8000 ..."
uv run python -m f1_rl.server &
backend=$!

echo "==> Starte Frontend (Vite) ..."
(cd web && npm run dev) &
frontend=$!

echo ""
echo "Beide laufen. Frontend ueblicherweise auf http://localhost:5173"
echo "Mit STRG+C beenden."

wait
