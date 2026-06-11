"""pywebview-Fenster für den F1-RL-Simulator.

Simulation und React-UI laufen in EINEM Prozess — kein FastAPI, kein uvicorn,
kein WebSocket. Das gebaute Frontend (web/dist) wird in ein natives Fenster
geladen; JavaScript spricht über die ``Api`` (api.py) direkt mit Python.

    python -m f1_rl.desktop          # gebautes UI (web/dist)
    python -m f1_rl.desktop --dev    # Vite-Dev-Server (http://localhost:5173)

Der ``if __name__ == "__main__"``-Guard in __main__.py ist Pflicht: Das Training
startet Worker-Prozesse, und unter Windows re-importiert jeder dieses Modul.
"""
from __future__ import annotations

import os
import sys

# Trainingslogs nutzen ein paar Unicode-Zeichen (≈, ≥, →). UTF-8 erzwingen,
# damit sie auf einer cp1252-Windows-Konsole nicht abstürzen.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import webview

from f1_rl.desktop.api import Api

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DIST_INDEX = os.path.join(ROOT, "web", "dist", "index.html")
DEV_URL = "http://localhost:5173"


def main() -> None:
    dev = "--dev" in sys.argv
    if dev:
        url = DEV_URL
    elif os.path.exists(DIST_INDEX):
        url = DIST_INDEX
    else:
        sys.exit(
            f"Gebautes UI nicht gefunden unter {DIST_INDEX}.\n"
            f"Erst `npm run build` in web/ ausführen — oder mit --dev gegen den "
            f"Vite-Dev-Server starten (vorher `npm run dev` in web/)."
        )

    webview.create_window(
        "F1 RL Simulator", url=url, js_api=Api(),
        width=1480, height=900, min_size=(1024, 700),
    )
    # debug=True öffnet die DevTools (rechte Maustaste -> Inspect) — beim Dev-Lauf praktisch.
    webview.start(debug=dev)
