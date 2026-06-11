# hft-intelligent-systems

F1 RL Simulator – ein Reinforcement-Learning-Agent, der lernt, eine Rennstrecke
zu fahren. Das Projekt besteht aus zwei Teilen:

- **Backend** (`f1_rl/`) – Python/FastAPI, liefert API + WebSocket auf
  `http://127.0.0.1:8000` und führt Simulation/Training aus.
- **Frontend** (`web/`) – React + Vite, verbindet sich mit dem Backend
  (läuft üblicherweise auf `http://localhost:5173`).

## Funktionen

- **Zwei Lern-Backends**
  - **Genetisches DQN** – jede Generation trainiert N_POP parallele Double-DQN-Worker,
    der genetische Algorithmus selektiert/kreuzt/mutiert ihre Gewichte.
  - **Tabellarisches Q-Learning** – klassische Q-Tabelle pro Agent, ebenfalls als
    genetische Population (kein neuronales Netz).
- **Rudel-Evolution (Pack-Modus)** – für *beide* Backends: die Population wird in
  „Packs" geclustert und teils auf Gruppen-Ebene selektiert, sodass schwache DNA
  über ihr Pack überleben kann. Im Live-Bild werden Packs eingefärbt.
- **Modellauswahl beim Fahren** – „Laden & Fahren" lässt das gespeicherte
  **DQN-Netz** oder die **Q-Table** die Strecke abfahren (Auswahl im Bedienfeld).
- **Sensor-Strahlen (Rays)** – 7 Lidar-Strahlen als Fahrzeug-Sicht, im Bedienfeld
  an-/abschaltbar. Klickt man ein Auto an, werden seine Strahlen im 3D-Bild
  gezeichnet (rot = Wand nah, grün = frei) und im Inspector als Balken angezeigt.
- **Auto-Inspector** – Klick auf ein Auto zeigt Telemetrie, Kräfte (g),
  Score-Zusammensetzung, die Sensoren sowie das neuronale Netz bzw. die Q-Tabelle.
- **Live-Ansicht** – frei drehbare 3D-Szene, Racing-Line der besten Runde,
  Performance- und Tempo-Charts, einstellbare/auto-synchrone Geschwindigkeit.

## Bedienung

- **Linksklick** auf ein Auto: auswählen (zeigt Inspector + Sensor-Strahlen).
- **Ziehen**: Kamera drehen · **Rechtsklick ziehen**: verschieben · **Mausrad**: zoomen.
- Strecke, Steps/Generation, Generationen, Backend und Optionen links im Bedienfeld;
  Training mit **Training starten** / **Fortsetzen**, gespeichertes Modell mit
  **Laden & Fahren** abspielen.

## Voraussetzungen

- [uv](https://docs.astral.sh/uv/) (Python-Paketmanager)
- [Node.js](https://nodejs.org/) inkl. npm
- Python 3.12+

## Schnellstart

Ein Skript startet alles (Abhängigkeiten werden bei Bedarf installiert,
Backend und Frontend werden zusammen gestartet):

**Windows (PowerShell):**

```powershell
./start.ps1
```

> Falls die Ausführung blockiert ist, einmalig:
> `Set-ExecutionPolicy -Scope Process Bypass`

**Linux / macOS / Git Bash:**

```bash
chmod +x start.sh   # nur beim ersten Mal
./start.sh
```

Danach das Frontend im Browser öffnen: <http://localhost:5173>

Beenden mit **STRG+C** – dabei werden Backend und Frontend gemeinsam gestoppt.

## Manueller Start

Falls man die Teile getrennt starten möchte:

```bash
# Backend
uv sync
uv run python -m f1_rl.server

# Frontend (in einem zweiten Terminal)
cd web
npm install
npm run dev
```

## Architektur (Kurzüberblick)

- **Simulation** (`f1_rl/simulation/`) – kinematische Fahrphysik als reine Funktionen
  (Auto-Zustand = Dict). Liefert pro Schritt Beobachtung (14 Werte inkl. 7 Rays),
  Reward und Frames.
- **Lernen** (`f1_rl/learning/`) – `trainer.py` (genetisches DQN) und `qtable.py`
  (tabellarisch); gemeinsame Bausteine in `evolution.py`, GA-Operatoren in
  `genetics.py`, Live-Anzeige in `display.py`, Racing-Line in `replay.py`.
- **Session** (`f1_rl/server/session.py`) – hält *die eine* laufende Session
  (Training oder „Fahren"), den Worker-Thread und die Queues. Ein neuer Start
  stoppt den vorherigen Lauf.
- **Bridge/UI** – das Frontend spricht über die **pywebview-Bridge**
  (`f1_rl/desktop/api.py` ↔ `web/src/lib/bridge.ts`) mit der Session und *zieht*
  Frames per `poll()` (~60×/s). Der FastAPI-`server/app.py` bietet denselben
  Datenfluss alternativ über einen WebSocket. Frames landen in einem React-`ref`,
  den die three.js-Schleife (`web/src/three/TrackScene.ts`) direkt liest — so
  rendert React nicht bei jedem Frame neu.
