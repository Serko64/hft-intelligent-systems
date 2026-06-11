"""Desktop-Frontend des F1-RL-Simulators auf Basis von pywebview.

Die React-Oberfläche (web/) und die Simulation (f1_rl/) laufen hier im SELBEN
Prozess — ohne FastAPI, ohne uvicorn, ohne WebSocket. Das gebaute Web-UI wird
in ein natives Fenster geladen; JavaScript spricht über die ``Api`` (api.py)
direkt mit Python.

Start:  python -m f1_rl.desktop          (gebautes UI aus web/dist)
        python -m f1_rl.desktop --dev    (Vite-Dev-Server auf localhost:5173)
"""
