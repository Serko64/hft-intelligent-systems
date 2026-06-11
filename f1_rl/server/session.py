"""Eine einzige Live-Session: entweder ein Trainingslauf oder "Modell laden & fahren".

Statt einer Klasse gibt es hier ein Modul-Dict ``SESSION`` (Typ: SessionState)
plus Funktionen, die es verändern. Die Session besitzt den Worker-Thread und
die Queues, die die Simulation füttert; die FastAPI-App (app.py) liest die
Queues und sendet an die Browser. Es existiert immer nur eine Session
(Ein-Benutzer-Lehrtool) — ein neuer Start stoppt den vorherigen Lauf.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import TypedDict

from f1_rl.config import (
    CIRCUITS, MODEL_DIR, MODEL_PATH, SIM_SPEED_DEFAULT, SIM_SPEED_MAX,
)
from f1_rl.simulation.track_loader import Track, load_track
from f1_rl.utils.queues import put_latest

# Pfad der gespeicherten Q-Tabelle (lokal gebildet, damit der schwere
# qtable-Import nicht schon beim Server-Start gezogen wird).
QTABLE_PATH = os.path.join(MODEL_DIR, "qtable.npz")


class SessionState(TypedDict):
    """Typ-Beschreibung für das SESSION-Dict (zur Laufzeit ein normales Dict)."""
    mode: str                            # "idle" | "training" | "driving"
    track: Track | None
    render_q: queue.Queue | None         # Listen von CarFrames (~60 fps)
    stats_q: queue.Queue | None          # Statistik pro Generation
    line_q: queue.Queue | None           # Racing-Line der besten Runde
    inspect_q: queue.Queue | None        # Netz-/Q-Detail des inspizierten Autos
    table_q: queue.Queue | None          # volle Q-Tabelle (nur Q-Table-Modus)
    speed_holder: list                   # [int] — live verstellbare Sim-Geschwindigkeit
    inspect_holder: list                 # [int | None] — Index des inspizierten Autos
    stop_event: threading.Event
    thread: threading.Thread | None


def _new_session_state() -> SessionState:
    return {
        "mode": "idle",
        "track": None,
        "render_q": None,
        "stats_q": None,
        "line_q": None,
        "inspect_q": None,
        "table_q": None,
        # 1-Element-Listen, die mit den Sim-Threads geteilt werden: Änderungen
        # wirken sofort, ohne den Lauf neu zu starten.
        "speed_holder": [SIM_SPEED_DEFAULT],
        "inspect_holder": [None],
        "stop_event": threading.Event(),
        "thread": None,
    }


SESSION: SessionState = _new_session_state()


def find_circuit(name: str) -> tuple[str | None, float]:
    """Sucht die Strecke in der CIRCUITS-Liste; gibt (GeoJSON-Fallback, Halbbreite) zurück."""
    for circuit_name, geojson_fallback, half_width in CIRCUITS:
        if circuit_name == name:
            return geojson_fallback, half_width
    return None, 10.0


def stop_session() -> None:
    SESSION["stop_event"].set()
    SESSION["mode"] = "idle"


def set_speed(speed: int) -> None:
    """Live-Geschwindigkeit der Anzeige ändern (Sim-Schritte pro Bild)."""
    SESSION["speed_holder"][0] = max(1, min(SIM_SPEED_MAX, int(speed)))


def set_inspect(index) -> None:
    """Welches Auto (per Index) Netz-/Q-Detail streamen soll; None = keins."""
    SESSION["inspect_holder"][0] = None if index is None else int(index)


def _load_session_track(circuit: str) -> Track:
    geojson_fallback, half_width = find_circuit(circuit)
    return load_track(circuit, geojson_fallback_path=geojson_fallback,
                      half_width_m=half_width)


def start_training(circuit: str, steps_per_gen: int, total_gens: int,
                   evolution_mode: str, resume: bool, use_rays: bool = True,
                   auto_speed: bool = False) -> None:
    stop_session()
    # Frisches Event für DIESEN Lauf. Das alte Event bleibt gesetzt und ist in
    # der Closure des vorherigen Laufs gefangen — der beendet sich damit selbst.
    SESSION["stop_event"] = threading.Event()
    SESSION["track"] = _load_session_track(circuit)
    SESSION["render_q"] = queue.Queue(maxsize=4)
    SESSION["stats_q"] = queue.Queue(maxsize=10)
    SESSION["line_q"] = queue.Queue(maxsize=2)
    SESSION["inspect_q"] = queue.Queue(maxsize=2)
    # Nur das Q-Table-Backend streamt eine volle Tabelle; der DQN-Trainer ignoriert sie.
    is_qtable = evolution_mode.startswith("qtable")   # "qtable" oder "qtable_pack"
    SESSION["table_q"] = queue.Queue(maxsize=2) if is_qtable else None
    SESSION["mode"] = "training"
    cancel_event = SESSION["stop_event"]

    def run() -> None:
        # "qtable"/"qtable_pack" wählt das Backend ohne neuronales Netz (klassisches
        # tabellarisches Q-Learning); alles andere nutzt den genetischen DQN-Trainer.
        # Beide haben dieselbe Signatur, der restliche Aufruf ist identisch.
        if is_qtable:
            from f1_rl.learning.qtable import q_learning_loop as train
        else:
            from f1_rl.learning.trainer import train
        try:
            train(track=SESSION["track"], render_queue=SESSION["render_q"],
                  stats_queue=SESSION["stats_q"], line_queue=SESSION["line_q"],
                  speed_holder=SESSION["speed_holder"],
                  inspect_holder=SESSION["inspect_holder"],
                  inspect_queue=SESSION["inspect_q"],
                  auto_speed=auto_speed,
                  steps_per_gen=steps_per_gen, total_gens=total_gens,
                  evolution_mode=evolution_mode, resume=resume,
                  use_rays=use_rays, cancel_event=cancel_event,
                  table_queue=SESSION["table_q"])
        except Exception as e:               # noqa: BLE001
            print(f"[server] training error: {e}")
        finally:
            if SESSION["mode"] == "training":
                SESSION["mode"] = "idle"

    SESSION["thread"] = threading.Thread(target=run, daemon=True)
    SESSION["thread"].start()


def _load_drive_policy(backend: str):
    """Lädt das gespeicherte Modell und liefert (policy, choose_action,
    inspect_trace, make_table_payload) — einheitlich für DQN und Q-Table.

    make_table_payload ist nur beim Q-Table-Backend gesetzt (für die Heatmap),
    beim DQN None. choose_action(policy, obs) wählt die beste Aktion, inspect_
    trace(policy, obs) -> (q_values, hidden) füttert das Detail-Panel.
    """
    if backend == "qtable":
        from f1_rl.learning.qtable import (
            inspect_trace, load_q_table, policy_action_readonly, _table_heatmap_payload,
        )
        table = load_q_table(QTABLE_PATH)
        return table, policy_action_readonly, inspect_trace, _table_heatmap_payload

    from f1_rl.learning.agent import act, forward_trace, neuronal_net_from_weight_file
    net = neuronal_net_from_weight_file(MODEL_PATH)
    return net, act, forward_trace, None


def _drive_loop(track: Track, use_rays: bool, backend: str, stop_event: threading.Event,
                render_q: queue.Queue, line_q: queue.Queue, inspect_q: queue.Queue,
                table_q: queue.Queue | None, inspect_holder: list, speed_holder: list) -> None:
    """Fährt das gewählte Modell (DQN oder Q-Table) live über die Strecke (~60 fps)."""
    from f1_rl.learning.replay import emit_racing_line, record_greedy_replay
    from f1_rl.server.protocol import inspect_to_dict
    from f1_rl.simulation.environment import (
        create_car_env, make_car_frame, reset_env, step_env,
    )

    policy, choose_action, inspect_trace, make_table_payload = _load_drive_policy(backend)
    # Eine Runde vorab aufzeichnen, damit ihre Racing-Line bereitliegt — gestreamt
    # wird sie aber erst NACH dem ersten Live-Frame, sonst erscheint die Linie,
    # bevor das Auto sichtbar ist.
    recorded_line = record_greedy_replay(policy, choose_action, track,
                                         max_steps=6000, use_rays=use_rays)

    env = create_car_env(track, use_rays=use_rays)
    obs = reset_env(env)
    dt = 1.0 / 60.0
    first_frame = True
    frame_no = 0
    INSPECT_EVERY = 6   # Netz-/Q-Detail ~10×/s statt 60×/s
    TABLE_EVERY = 120   # volle Q-Tabelle ~alle 2 s (große Nutzlast)

    while not stop_event.is_set():
        frame_start = time.perf_counter()
        frame_no += 1
        # Mehrere Sim-Schritte pro gerendertem Bild für eine schnellere, lebhaftere
        # Ansicht (Frames kommen trotzdem mit 60 fps). Das Bild zeigt den Endzustand.
        terminated = False
        for _ in range(max(1, int(speed_holder[0]))):
            action = choose_action(policy, obs)
            obs, _, terminated, _, _ = step_env(env, action)
            if terminated:
                break

        put_latest(render_q, [make_car_frame(env)])

        if first_frame:
            emit_racing_line(line_q, recorded_line)
            first_frame = False

        # Netz-/Q-Detail streamen, wenn das (einzige) Auto inspiziert wird.
        if inspect_holder[0] == 0 and frame_no % INSPECT_EVERY == 0:
            q_values, hidden = inspect_trace(policy, obs)
            put_latest(inspect_q,
                       inspect_to_dict(0, obs, q_values, hidden, int(q_values.argmax())))

        # Volle Q-Tabelle als Heatmap (nur Q-Table-Backend).
        if (table_q is not None and make_table_payload is not None
                and inspect_holder[0] == 0 and frame_no % TABLE_EVERY == 0):
            put_latest(table_q, make_table_payload(policy, 0))

        if terminated:
            obs = reset_env(env)

        remaining = dt - (time.perf_counter() - frame_start)
        if remaining > 0:
            time.sleep(remaining)

    if SESSION["mode"] == "driving":
        SESSION["mode"] = "idle"


def start_driving(circuit: str, use_rays: bool = True, backend: str = "dqn") -> None:
    stop_session()
    SESSION["stop_event"] = threading.Event()
    model_path = QTABLE_PATH if backend == "qtable" else MODEL_PATH
    if not os.path.exists(model_path):
        label = "Q-Table" if backend == "qtable" else "DQN-Modell"
        raise FileNotFoundError(f"Kein trainiertes {label} unter {model_path} — erst trainieren.")
    SESSION["track"] = _load_session_track(circuit)
    SESSION["render_q"] = queue.Queue(maxsize=4)
    SESSION["stats_q"] = None
    SESSION["line_q"] = queue.Queue(maxsize=2)
    SESSION["inspect_q"] = queue.Queue(maxsize=2)
    # Nur das Q-Table-Backend streamt eine Heatmap; der DQN-Pfad lässt sie None.
    SESSION["table_q"] = queue.Queue(maxsize=2) if backend == "qtable" else None
    SESSION["mode"] = "driving"

    SESSION["thread"] = threading.Thread(
        target=_drive_loop,
        args=(SESSION["track"], use_rays, backend, SESSION["stop_event"],
              SESSION["render_q"], SESSION["line_q"], SESSION["inspect_q"],
              SESSION["table_q"], SESSION["inspect_holder"], SESSION["speed_holder"]),
        daemon=True,
    )
    SESSION["thread"].start()
