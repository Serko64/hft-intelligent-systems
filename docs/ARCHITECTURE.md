# Architektur — F1 RL Simulator (Web UI)

Dieses Dokument beschreibt die Architektur der **Web-Variante** der Anwendung.
Die alte Pygame-Desktop-UI wurde nach [`old_ui/`](../old_ui) archiviert und ist
für den Web-Betrieb nicht mehr nötig.

## Überblick

Die Anwendung trainiert mit Reinforcement Learning (Genetic Double-DQN) einen
F1-Wagen auf realen Rennstrecken und streamt die laufende Simulation live in den
Browser.

Drei Schichten:

| Schicht | Ordner | Aufgabe |
|---|---|---|
| **Frontend** | `web/` | React + Vite + three.js; rendert Strecke & Autos, sendet Befehle |
| **Backend / Transport** | `f1_rl/server/` | FastAPI: REST + WebSocket, `Session` als Brücke zur Simulation |
| **Domäne** | `f1_rl/learning/`, `f1_rl/simulation/` | RL-Training (GA + DQN) und Fahrphysik/Strecken |
| **Persistenz** | `model/`, `circuits/` | Gewichte, Checkpoints, Replays, Streckengeometrie |

---

## Architekturdiagramm

```mermaid
flowchart TB
    subgraph Browser["🌐 Browser — web/"]
        App["App.tsx<br/>State & Layout"]
        Ctrl["ControlPanel.tsx<br/>Strecke / Steps / Gens / Modus"]
        Hud["Hud.tsx<br/>Fitness, Generation, ε"]
        Canvas["SceneCanvas.tsx + three/TrackScene.ts<br/>3D-Rendering ~60 fps"]
        Sock["lib/useSimSocket.ts<br/>WebSocket-Client"]
        App --> Ctrl & Hud & Canvas
        Ctrl -- "send(cmd)" --> Sock
        Sock -- "frames → carsRef" --> Canvas
        Sock -- "stats / status" --> Hud
        Sock -- "track" --> Canvas
    end

    subgraph Server["⚙️ FastAPI Backend — f1_rl/server/"]
        Api["app.py<br/>REST /api/* + WS /ws"]
        Pump["_pump() Task<br/>leert Queues 60×/s → broadcast"]
        Proto["protocol.py<br/>JSON ⇄ Sim-Objekte"]
        Sess["session.py · Session<br/>mode: idle/training/driving<br/>besitzt Worker-Thread + Queues"]
        Api --> Sess
        Pump --> Proto
        Sess --> Pump
    end

    subgraph Domain["🧠 Domäne — f1_rl/learning + simulation"]
        Trainer["learning/trainer.py · train()<br/>Generationsschleife (GA)"]
        Disp["_display_thread<br/>animiert ganze Population"]
        Worker["learning/worker.py · run_worker<br/>Double-DQN pro Individuum"]
        Gen["learning/genetics.py<br/>select / crossover / mutate / packs"]
        Net["learning/network.py · agent.py<br/>NN-Gewichte ⇄ Torch-Netz"]
        Env["simulation/environment.py · F1Env<br/>Gymnasium-Env: Physik, Rays, Reward"]
        Track["simulation/track_loader.py<br/>OSM/GeoJSON → Streckengeometrie"]
        Trainer --> Disp & Worker & Gen
        Worker --> Env & Net
        Disp --> Env & Net
        Trainer --> Track
    end

    subgraph Store["💾 Persistenz"]
        Model[("model/<br/>model.npy · checkpoints/<br/>replays/ · training_state.json")]
        Circuits[("circuits/<br/>*.geojson")]
        Cache[("cache/<br/>OSM-Cache")]
    end

    Sock <-. "WebSocket /ws (JSON)" .-> Api
    Ctrl <-. "REST /api/circuits, /api/track" .-> Api
    Sess -- "startet Thread" --> Trainer
    Sess -- "load_and_drive" --> Env
    Trainer <-- "render_q / stats_q" --> Pump
    Trainer --> Model
    Track --> Circuits & Cache
    Worker -. "ProcessPool (1 Prozess/Individuum)" .- Worker
```

### Datenflüsse

- **Befehle (Client → Server):** `start_training`, `load_and_drive`, `stop` über
  den WebSocket; Strecken-Metadaten über REST.
- **Frames (Server → Client):** Der `_display_thread` schiebt ~60×/s die
  Zustände aller Autos in `render_q`; `_pump()` nimmt jeweils den neuesten Frame
  und broadcastet ihn. Der Client legt ihn in `carsRef` (kein React-Rerender) —
  three.js liest direkt daraus.
- **Stats (Server → Client):** Einmal pro Generation (Fitness, ε, Generation,
  Top-Scores) über `stats_q` → React-State → HUD.
- **Parallelität:** Pro Generation laufen bis zu `N_POP` Worker-Prozesse
  (`ProcessPoolExecutor`) für das DQN-Training; ein separater Thread animiert
  parallel die Live-Population.

---

## Sequenzdiagramm — Trainingslauf

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React UI<br/>(ControlPanel)
    participant WS as useSimSocket
    participant Api as FastAPI app.py
    participant Sess as Session
    participant Tr as trainer.train()<br/>(Thread)
    participant Pool as ProcessPool<br/>(run_worker ×N_POP)
    participant Disp as _display_thread
    participant Pump as _pump() (60 Hz)
    participant Store as model/

    User->>UI: Strecke, Steps/Gen, Gens, Modus wählen, "Start"
    UI->>WS: send({type:"start_training", ...})
    WS->>Api: WS-Nachricht /ws
    Api->>Sess: start_training(circuit, steps, gens, mode, resume)
    Sess->>Sess: stop() vorherige, Track laden,<br/>render_q + stats_q anlegen, mode=training
    Sess->>Tr: Thread starten → train(...)
    Api-->>WS: status "training" + track
    WS-->>UI: Strecke zeichnen, HUD auf "training"

    Tr->>Store: Population init / resume (model.npy)
    Tr->>Disp: Display-Thread starten (animiert Population)

    loop für jede Generation (gen = 0..total_gens)
        Tr->>Tr: ε-greedy-Wert für diese Generation berechnen
        Tr->>Pool: executor.map(run_worker, population)
        activate Pool
        Pool->>Pool: pro Individuum: Double-DQN-Training<br/>(Replay-Buffer, Gradienten)
        Pool->>Pool: 1 Greedy-Eval-Runde → Fitness
        Pool-->>Tr: (fitness, gewichte, end_x, end_y) ×N_POP
        deactivate Pool
        Tr->>Tr: nach Fitness sortieren, Hall-of-Fame,<br/>ggf. Packs zuordnen
        Tr->>Disp: pop_holder[0] = neue Population (Soft-Swap)
        Tr->>Tr: breed (genetics.py): select / crossover / mutate
        Note over Disp,Pump: Disp steppt Autos und legt Frames in render_q
        Disp-->>Pump: render_q (Frames ~60 fps)
        Tr-->>Pump: stats_q (1× pro Generation)
        Pump-->>WS: broadcast frame + stats
        WS-->>UI: carsRef → three.js Render, HUD-Update
        opt alle SAVE_EVERY Generationen
            Tr->>Store: model.npy + training_state.json speichern
        end
        opt neuer Bestwert
            Tr->>Store: Replay (.npz) speichern
        end
    end

    alt User klickt "Stop"
        User->>UI: "Stop"
        UI->>WS: send({type:"stop"})
        WS->>Api: /ws
        Api->>Sess: stop() → cancel_event.set()
        Note over Tr: beendet sauber am nächsten<br/>Generationsende
    end

    Tr->>Store: finale Gewichte speichern
    Tr->>Disp: stop_event → Display-Thread beenden
    Tr->>Sess: mode = "idle"
    Pump-->>WS: status "idle"
    WS-->>UI: HUD zurücksetzen, Szene leeren
```

---

## "Load & Drive" (Kurzform)

Neben dem Training kann ein bereits trainiertes Modell gefahren werden:

1. UI sendet `{type:"load_and_drive", circuit}`.
2. `Session.start_driving()` lädt `model.npy` via `load_policy()` und
   startet einen Thread, der ein einzelnes `F1Env` mit 60 fps fährt.
3. Frames gehen über `render_q` → `_pump()` → WebSocket → three.js.
   (Keine `stats_q`, da kein Training läuft.)

---

## Verzeichnisstruktur (Web-relevant)

```
hft-intelligent-systems/
├─ web/                      # React/Vite/three.js Frontend
│  └─ src/
│     ├─ App.tsx, components/ (ControlPanel, Hud, SceneCanvas, ui/)
│     ├─ three/TrackScene.ts # 3D-Rendering
│     └─ lib/ (useSimSocket.ts, types.ts, utils.ts)
├─ f1_rl/
│  ├─ config.py              # zentrale Konstanten & Strecken-Liste
│  ├─ server/                # FastAPI: app, protocol, session, __main__
│  ├─ learning/              # trainer, worker, agent, network, genetics
│  └─ simulation/            # environment (F1Env), track_loader, track_render*
├─ model/                    # model.npy, checkpoints/, replays/, training_state.json
├─ circuits/                 # *.geojson Streckendaten
├─ cache/                    # OSM-Cache
└─ old_ui/                   # ⛔ archivierte Pygame-Desktop-UI (nicht für Web nötig)

* track_render.py wird nur vom Pygame-Render-Pfad genutzt; environment.py
  importiert es verzögert (deferred) und der Web-Server triggert es nie
  (render_mode=None).
```

## Starten (Web)

```bash
# Backend (Repo-Root)
python -m f1_rl.server          # http://127.0.0.1:8000

# Frontend
cd web && npm install && npm run dev
```
