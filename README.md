# hft-intelligent-systems

F1 RL Simulator, ein Reinforcement-Learning-Agent, der lernt, eine Rennstrecke zu
fahren. Das Projekt besteht aus zwei Teilen, die als eine Desktop-Anwendung
zusammenlaufen:

- **Backend** (`f1_rl/`): Python, übernimmt Fahrsimulation, Lernen und Training.
- **Frontend** (`web/`): React, Vite und three.js, rendert Strecke und Autos und
  schickt die Bedienbefehle zurück.

Verbunden werden die beiden über **pywebview**. Statt eines Servers läuft alles in
einem Prozess: ein Desktop-Fenster zeigt das gebaute Web-UI, und JavaScript ruft die
Python-Methoden direkt über die Bridge auf und zieht die Live-Bilder per `poll()`. Es
gibt also keinen Netzwerk-Port und keinen separaten Server, der gestartet werden müsste.

## Funktionen

- **Zwei Lern-Backends**
  - **Genetisches DQN:** jede Generation trainiert `N_POP` parallele Double-DQN-Worker,
    danach selektiert, kreuzt und mutiert der genetische Algorithmus ihre Gewichte.
  - **Tabellarisches Q-Learning:** eine klassische Q-Tabelle pro Agent, ebenfalls als
    genetische Population geführt, ganz ohne neuronales Netz.
- **Modellauswahl beim Fahren:** „Laden und Fahren" lässt entweder das gespeicherte
  **DQN-Netz** oder die **Q-Table** die Strecke abfahren, die Wahl trifft man im Bedienfeld.
- **Sensor-Strahlen (Rays):** sieben Lidar-Strahlen als Fahrzeug-Sicht, im Bedienfeld
  an- und abschaltbar. Klickt man ein Auto an, werden seine Strahlen im 3D-Bild gezeichnet
  (rot heißt Wand nah, grün heißt frei) und im Inspector zusätzlich als Balken angezeigt.
- **Robustere Fitness (nur DQN):** der Multi-Start-Eval mittelt den Score über mehrere
  gleichmäßig verteilte Startpunkte, statt nur ab Start und Ziel zu bewerten. Getrennt davon
  kann das genetische Crossover abgeschaltet werden, dann läuft reine Mutation im Stil einer
  Evolutionsstrategie.
- **Auto-Inspector:** ein Klick auf ein Auto zeigt Telemetrie, die wirkenden Kräfte in g,
  die Score-Zusammensetzung, die Sensoren sowie das neuronale Netz bzw. die Q-Tabelle als Heatmap.
- **Live-Ansicht:** eine frei drehbare 3D-Szene, die Racing-Line der besten Runde, Charts für
  Performance und Tempo, ein Scoreboard der zehn besten Läufe und eine einstellbare oder
  automatisch generationssynchrone Geschwindigkeit.

## Bedienung

- **Linksklick** auf ein Auto wählt es aus und zeigt Inspector und Sensor-Strahlen.
- **Ziehen** dreht die Kamera, **Rechtsklick ziehen** verschiebt sie, das **Mausrad** zoomt.
- Strecke, Steps je Generation, Generationen, Backend und die Optionen liegen links im
  Bedienfeld. Ein Training startet man mit **Training starten** oder **Fortsetzen**, ein
  gespeichertes Modell spielt man mit **Laden und Fahren** ab.

## Voraussetzungen

- [uv](https://docs.astral.sh/uv/) als Python-Paketmanager
- [Node.js](https://nodejs.org/) inklusive npm
- Python 3.12 oder 3.13

## Schnellstart

Zuerst wird das Web-UI einmal gebaut, danach startet die Desktop-App, die das gebaute UI
lädt. Beides läuft vom Repo-Wurzelverzeichnis aus:

```bash
# 1. Frontend bauen (legt web/dist an)
cd web
npm install
npm run build
cd ..

# 2. Python-Abhängigkeiten und Desktop-App
uv sync
uv run python -m f1_rl.desktop.main
```

Danach öffnet sich das Anwendungsfenster direkt, ein Browser wird nicht gebraucht.

> Hinweis: Das Fenster lädt das gebaute UI aus `web/dist`. Nach Änderungen am Frontend
> muss `npm run build` deshalb erneut laufen, damit sie sichtbar werden.

## Architektur (Kurzüberblick)

- **Simulation** (`f1_rl/simulation/`): die kinematische Fahrphysik als reine Funktionen,
  der Auto-Zustand ist ein Dict. Pro Schritt fallen die Beobachtung (14 Werte inklusive
  sieben Rays), der Reward und die Anzeige-Frames an.
- **Lernen** (`f1_rl/learning/`): `trainer.py` für das genetische DQN und `qtable.py` für
  die tabellarische Variante, dazu die geteilten Bausteine in `evolution.py`, die GA-Operatoren
  in `genetics.py`, die Live-Anzeige in `display.py` und die Racing-Line in `replay.py`.
- **Session** (`f1_rl/server/session.py`): hält die eine laufende Session, also entweder das
  Training oder das Fahren, samt Worker-Thread und Queues. Ein neuer Start stoppt den vorigen Lauf.
- **Bridge und UI** (`f1_rl/desktop/`): das Frontend spricht über die pywebview-Bridge
  (`f1_rl/desktop/api.py` und `web/src/lib/bridge.ts`) mit der Session und zieht die Frames per
  `poll()` mit etwa 60 Bildern pro Sekunde. Die Frames landen in einem React-`ref`, den die
  three.js-Schleife (`web/src/three/TrackScene.ts`) direkt liest, sodass React nicht bei jedem
  Bild neu rendert.

## Code-Referenz (Backend)

Die Module sind bewusst ohne Docstrings gehalten, die Verantwortlichkeiten stehen hier und
die Detail-Semantik einzelner Felder in den Inline-Kommentaren am Code.

### Module im Detail

**Simulation (`f1_rl/simulation/`)**

- `environment.py`: die kinematische Fahrsimulation als reine Funktionen, `create_car_env(track)`
  liefert ein env, `reset_env(env)` die erste Beobachtung und `step_env(env, action)` das Tupel
  `(obs, reward, terminated, truncated, info)` nach gym-Konvention. Der Auto-Zustand ist ein Dict
  (`CarEnv`), das pro Frame ans UI gestreamte Bild ein `CarFrame`. **Grip-Kreis:** Quer- und
  Längsbeschleunigung teilen sich ein Reibungsbudget, zu hartes Einlenken führt also zu Untersteuern
  statt zu einer magischen Drehung. **Curriculum:** die Optimierungs-Rewards für Tempo, Sanftheit
  und Zeit zählen erst, nachdem ein Auto die erste volle Runde geschafft hat.
- `track_loader.py`: baut das `Track`-Dict aus OSM-Daten (osmnx) oder einer GeoJSON-Datei, wobei die
  kuratierte GeoJSON Vorrang vor den Live-OSM-Daten hat. Enthält die Centerline in Metern, den befahrbaren
  Korridor und vorberechnete Pixel-Koordinaten, OSM-Strecken werden als `.pkl` gecacht
  (`TRACK_CACHE_VERSION` bei einer Formatänderung hochzählen).
- `track_render.py`: bäckt ein `Track` einmal in eine pygame-Surface, mit dreifachem Supersampling
  fürs Antialiasing und pro Track gecacht.

**Lernen (`f1_rl/learning/`)**

- `network.py`: die einzige Netz-Definition, ein kleines PyTorch-MLP `N_OBS → NET_HIDDEN → N_ACTIONS`.
  `flat_to_network` und `network_to_flat` wandeln zwischen Modul und flachem float32-Vektor, auf diesem
  Vektor arbeitet der GA. Die Architektur ändert man zentral über `config.NET_HIDDEN`.
- `agent.py`: die Inferenz mit `neural_net_from_weights`, `load_neural_net`, `act` für die beste Aktion
  und `forward_trace` für die Q-Werte samt Post-ReLU-Aktivierungen fürs Inspect-Panel.
- `genetics.py`: die GA-Operatoren auf flachen Vektoren, also `crossover`, `mutate` und `rank_select`.
- `evolution.py`: die geteilten Generations-Bausteine, also Epsilon-Schedule, Eval-Budget, Hall of Fame,
  Top-Scoreboard, Stagnations-Boost und Auto-Speed.
- `worker.py`: trainiert ein einzelnes GA-Individuum per Double DQN (vorab allozierter numpy-Replay-Buffer
  und ε-greedy) in einem eigenen Prozess, bewertet es greedy und gibt `(fitness, weights, end_x, end_y)`
  zurück. Eine vorhandene CUDA-GPU wird mit größerem Batch genutzt.
- `qtable.py`: das tabellarische Q-Learning als genetische Population von Q-Tabellen, `q_learning_loop`
  spiegelt dabei `trainer.py`. **Invariante:** eine fertige Tabelle wird nie mehr in place verändert,
  daher dürfen Crossover und Mutation unveränderte Q-Zeilen referenzieren (copy-on-write).
- `trainer.py`: die genetische DQN-Generationsschleife, pro Generation laufen `N_POP` Worker parallel,
  danach wird nach Fitness sortiert, die Hall of Fame gepflegt und die nächste Generation gezüchtet
  (Rang-Selektion, Crossover, Mutation). Übernimmt auch die Persistenz von Checkpoints und Trainingsstand,
  und `cancel_event` stoppt sauber an der Generationsgrenze.
- `display.py`: die gemeinsame Live-Anzeige mit etwa 60 fps für beide Backends, angebunden über die
  Callbacks `build_policy`, `choose_action` und `inspect_trace_fn`. **Soft-Swap:** bei einer neuen
  Generation fährt jedes Auto mit seiner alten Policy bis zum natürlichen Crash weiter, statt dass alle
  auf einmal teleportieren, und `MAX_DISPLAY_LAG` begrenzt den Verzug.
- `replay.py`: zeichnet die beste greedy Fahrt als Racing-Line auf und schickt sie ans UI. Nur ein
  neuer Allzeit-Bestwert frischt die Linie auf, damit sie monoton besser wird.

**Server und Desktop (`f1_rl/server/`, `f1_rl/desktop/`)**

- `session.py`: hält die eine laufende Session, also Training oder Fahren, samt Worker-Thread und Queues
  (`SessionState`-Dict). Ein neuer Start stoppt den vorigen Lauf über ein frisches `stop_event`.
- `protocol.py`: die Serialisierung zwischen den Sim-Dicts und dem JSON fürs Frontend, alle Koordinaten
  in **Metern**, das Frontend skaliert und zentriert selbst.
- `utils/queues.py`: `put_latest` und `get_latest` behalten nur den neuesten Eintrag, weil das UI immer
  den aktuellsten Stand will, und schlucken Fehler bewusst.
- `desktop/api.py`, `desktop/main.py`: die pywebview-Bridge, JavaScript ruft `Api.*` direkt auf und zieht
  die Frames per `poll()`. `main.py` öffnet das Fenster und lädt das gebaute UI aus `web/dist`.
- `config.py`: die zentralen Konstanten, Pfade und UI-Einstellungen, jede Konstante ist am Code kommentiert.

### Beobachtung und Aktion

- **Beobachtung:** 14 Werte, normiert auf `[-1, 1]` bzw. `[0, 1]`:
  `[0]` x, `[1]` y, `[2]` Heading-Abweichung/π, `[3]` Tempo, `[4]` Fortschritt,
  `[5]` Abstand links, `[6]` Abstand rechts, `[7]…[13]` sieben Lidar-Strahlen (−75° bis +75°).
- **Aktion:** einer von 20 diskreten `(Lenkung, Gas)`-Werten, also fünf Lenkstufen mal vier Gasstufen.

### Datenstrukturen

Statt Klassen nutzt das Backend einfache Dicts mit `TypedDict`-Beschreibung, die Felder sind jeweils
direkt am `TypedDict` kommentiert: `Track` (`track_loader.py`), `CarEnv` und `CarFrame`
(`environment.py`) sowie `SessionState` (`session.py`).
