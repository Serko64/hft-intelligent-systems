# F1 RL — Web-UI (React + three.js + WebSocket)

Browser-Frontend als Alternative zur pygame-UI. Der RL-Kern (`f1_rl/simulation`,
`f1_rl/learning`) bleibt unverändert; das Backend (`f1_rl/server`) streamt
denselben Frame-/Stats-Datenstrom über WebSocket an dieses Frontend.

## Starten (zwei Terminals)

**1) Backend** (FastAPI + WebSocket, Port 8000):
```bash
python -m f1_rl.server
```

**2) Frontend** (Vite Dev-Server, Port 5173):
```bash
cd web
npm run dev
```
Dann http://localhost:5173 öffnen. Der Vite-Dev-Server proxyt `/api` und `/ws`
automatisch zum Backend auf Port 8000 (siehe `vite.config.ts`) — kein CORS,
keine fest verdrahteten Ports.

## Auto-Modell

Lege ein glTF/GLB-Modell unter `web/public/models/f1.glb` ab — es ersetzt den
Platzhalter (farbiger Kegel) automatisch. Details: `web/public/models/README.md`.

## Architektur

```
Browser (React + three.js)  <->  WebSocket /ws  <->  FastAPI (f1_rl/server)
  ControlPanel (shadcn)            JSON-Frames         Session -> trainer.train()
  TrackScene   (three.js)          ~60 fps                      -> F1Env (drive)
  Hud          (shadcn)
```

- **WS-Protokoll:** `f1_rl/server/protocol.py` (Track-Geometrie, Frames, Stats).
- **Kommandos vom Client:** `start_training`, `load_and_drive`, `stop`.
- **Frames** landen in einem React-*ref* und werden direkt von der three.js-
  Render-Schleife gelesen (kein React-Rerender bei 60 fps).

## Charts

Platzhalter im HUD vorgesehen. Apache ECharts wird später ergänzt (Konfiguration
wird separat angelegt) — aktuell bewusst noch nicht eingebaut.

## Build / Typecheck
```bash
cd web && npm run build
```
