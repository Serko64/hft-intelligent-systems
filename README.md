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

## Code-Referenz (Backend)

Die Module sind bewusst ohne Docstrings gehalten — die Verantwortlichkeiten stehen
hier, die Detail-Semantik einzelner Felder in den Inline-Kommentaren am Code.

### Module im Detail

**Simulation (`f1_rl/simulation/`)**

- `environment.py` – kinematische Fahrsimulation als reine Funktionen:
  `create_car_env(track)` → env, `reset_env(env)` → obs,
  `step_env(env, action)` → `(obs, reward, terminated, truncated, info)` (gym-Konvention).
  Der Auto-Zustand ist ein Dict (`CarEnv`), das pro Frame ans UI gestreamte Bild ein
  `CarFrame`. **Grip-Kreis:** Quer- und Längsbeschleunigung teilen sich ein
  Reibungsbudget — zu hartes Einlenken führt zu Untersteuern statt magischer Drehung.
  **Curriculum:** die Optimierungs-Rewards (Tempo/Sanftheit/Zeit) zählen erst, nachdem
  ein Auto die erste volle Runde geschafft hat.
- `track_loader.py` – baut das `Track`-Dict aus OSM-Daten (osmnx) oder einer
  GeoJSON-Datei; die kuratierte GeoJSON hat Vorrang vor Live-OSM. Enthält Centerline
  (in Metern), befahrbaren Korridor und vorberechnete Pixel-Koordinaten; OSM-Strecken
  werden als `.pkl` gecacht (`TRACK_CACHE_VERSION` bei Formatänderung hochzählen).
- `track_render.py` – bäckt ein `Track` einmal in eine pygame-Surface (3×
  Supersampling für Antialiasing, pro Track gecacht).

**Lernen (`f1_rl/learning/`)**

- `network.py` – die *einzige* Netz-Definition (kleines PyTorch-MLP
  `N_OBS → NET_HIDDEN → N_ACTIONS`). `flat_to_network`/`network_to_flat` wandeln
  zwischen Modul und flachem float32-Vektor — auf dem Vektor arbeitet der GA. Architektur
  zentral über `config.NET_HIDDEN` ändern.
- `agent.py` – Inferenz: `neural_net_from_weights`, `load_neural_net`, `act` (beste
  Aktion), `forward_trace` (Q-Werte + Post-ReLU-Aktivierungen fürs Inspect-Panel).
- `genetics.py` – GA-Operatoren auf flachen Vektoren: `crossover`, `mutate`, `rank_select`.
- `evolution.py` – geteilte Generations-Bausteine: Epsilon-Schedule, Eval-Budget,
  Hall of Fame, Top-Scoreboard, Stagnations-Boost, Auto-Speed.
- `worker.py` – trainiert *ein* GA-Individuum per Double DQN (vorab allozierter
  numpy-Replay-Buffer + ε-greedy) in einem eigenen Prozess, bewertet es greedy und gibt
  `(fitness, weights, end_x, end_y)` zurück. CUDA wird bei Verfügbarkeit mit größerem
  Batch genutzt.
- `qtable.py` – tabellarisches Q-Learning als genetische Population von Q-Tabellen;
  `q_learning_loop` spiegelt `trainer.py`. **Invariante:** eine fertige Tabelle wird nie
  mehr in place verändert, daher dürfen Crossover/Mutation unveränderte Q-Zeilen
  referenzieren (copy-on-write).
- `trainer.py` – genetische DQN-Generationsschleife: pro Generation `N_POP` Worker
  parallel, nach Fitness sortieren, Hall of Fame pflegen, nächste Generation züchten
  (Rang-Selektion/Crossover/Mutation). Übernimmt auch die Persistenz
  (Checkpoints/Trainingsstand). `cancel_event` stoppt sauber an der Generationsgrenze.
- `display.py` – gemeinsame Live-Anzeige (~60 fps) für beide Backends über die
  Callbacks `build_policy`/`choose_action`/`inspect_trace_fn`. **Soft-Swap:** bei neuer
  Generation fährt jedes Auto mit seiner alten Policy bis zum natürlichen Crash weiter
  (kein Massen-Teleport); `MAX_DISPLAY_LAG` begrenzt den Verzug.
- `replay.py` – zeichnet die beste greedy Fahrt als Racing-Line auf und schickt sie ans
  UI. Nur ein *neuer* Allzeit-Bestwert frischt die Linie auf, damit sie monoton besser wird.

**Server / Desktop (`f1_rl/server/`, `f1_rl/desktop/`)**

- `session.py` – hält *die eine* laufende Session (Training oder „Fahren"), den
  Worker-Thread und die Queues (`SessionState`-Dict). Ein neuer Start stoppt den
  vorherigen Lauf über ein frisches `stop_event`.
- `protocol.py` – Serialisierung zwischen den Sim-Dicts und dem JSON fürs Frontend; alle
  Koordinaten in **Metern** (das Frontend skaliert/zentriert).
- `utils/queues.py` – `put_latest`/`get_latest`: behalten nur den neuesten Eintrag
  (das UI will immer den aktuellsten Stand) und schlucken Fehler bewusst.
- `desktop/api.py`, `desktop/main.py` – pywebview-Bridge: JavaScript ruft `Api.*` direkt
  und zieht Frames per `poll()`.
- `config.py` – zentrale Konstanten, Pfade und UI-Einstellungen (jede Konstante ist am
  Code kommentiert).

### Beobachtung & Aktion

- **Beobachtung** = 14 Werte, normiert auf `[-1, 1]` bzw. `[0, 1]`:
  `[0]` x, `[1]` y, `[2]` Heading-Abweichung/π, `[3]` Tempo, `[4]` Fortschritt,
  `[5]` Abstand links, `[6]` Abstand rechts, `[7]…[13]` 7 Lidar-Strahlen (−75°…+75°).
- **Aktion** = einer von 20 diskreten `(Lenkung, Gas)`-Werten (5 Lenk- × 4 Gas-Stufen).

### Datenstrukturen

Statt Klassen nutzt das Backend einfache Dicts mit `TypedDict`-Beschreibung; die Felder
sind jeweils direkt am `TypedDict` kommentiert:
`Track` (`track_loader.py`), `CarEnv`/`CarFrame` (`environment.py`),
`SessionState` (`session.py`).
