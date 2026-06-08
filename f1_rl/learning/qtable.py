"""Klassisches tabellarisches Q-Learning — die Variante OHNE neuronales Netz.

Diese Datei setzt das Q-Learning der Vorlesung direkt um. Der Code bleibt
unverändert lauffähig; die Benennung ist auf die Folien-Begriffe abgebildet.
Funktionen wurden NICHT umbenannt — der Folien-Begriff steht jeweils als
Kommentar (mit Vorschlag für den englischen Bezeichner zum späteren Refactor).

Begriffs-Mapping  Folie  ->  Code
---------------------------------
    Agent                     (Folie 3)   diese Datei: wählt Aktionen, lernt die Q-Tabelle
    Umgebung (Environment)    (Folie 4)   F1Env aus environment.py
    Zustand  s ∈ S            (Folie 4)   diskretisierter Beobachtungsvektor -> state key
    Aktion   a ∈ A            (Folie 3)   Aktionsindex 0..N_ACTIONS-1
    Reward   r = R(s,a)       (Folie 17)  reward aus env.step(...)
    Folgezustand s' = T(s,a)  (Folie 10)  next_state
    Q-Funktion Q(s,a)         (Folie 24)  erwarteter (abgezinster) Total Reward
    Q-Tabelle                 (Folie 25)  dict: Zustand -> Vektor mit |A| Q-Werten
    Policy  π(s)=argmax Q     (Folie 28)  greedy / argmax über die Zustands-Zeile
    Discount-Faktor γ         (Folie 22)  GAMMA (aus config.py)
    Lernrate α                (Folie 38)  ALPHA
    Temporal-Difference-Update(Folie 38)  update(...)
    Fehler Δ (TD-Fehler)      (Folie 37)  (target - row[a])
    Episode / Lernschleife    (Folie 40)  train_qtable(...)
    Zielzustand s_goal        (Folie 44)  done == True (terminated)

Zwei bewusste Abweichungen von der Folie-40-Pseudo-Schleife (beide praktisch nötig):
  * Folie 40 wählt in Schritt 1 rein greedy; hier wird ε-greedy gewählt, also mit
    Exploration. Das ist die Voraussetzung aus Folie 41 ("jeder Zustand muss
    (unendlich) oft besucht werden").
  * Folie 40 initialisiert die Q-Tabelle "zufallsbasiert"; hier mit Nullen
    (entsteht lazy beim ersten Zugriff). Funktioniert für dieses Reward-Design.

Es ist ein Drop-in-Ersatz für den Deep-Learning-Stack (network.py + agent.py +
worker.py + trainer.py): gleiche Umgebung, gleiches WebSocket-Protokoll, gleiche
Live-Ansicht — aber die Policy ist eine Wertetabelle statt eines MLP.

Warum die Tabelle Zusatzarbeit braucht
--------------------------------------
Eine echte Q-Tabelle ist ``Q[Zustand][Aktion]`` und braucht *diskrete* Zustände.
Die Umgebung liefert 14 kontinuierliche Floats in [-1, 1] (environment._get_obs).
Alle 14 mit echter Auflösung ergäben N_BINS**14 Zustände (Fluch der Dimensionalität,
vgl. Folie 41: "in der Praxis oft sehr große Tabellen, nicht berechenbar"). Daher:

  1. fahrrelevante Merkmale wählen (heading error, speed, die 7 Lidar-Strahlen),
  2. jedes in N_BINS Stufen diskretisieren  ->  kurzer, hashbarer Zustands-Key,
  3. Q-Werte in einem sparse dict halten (nur besuchte Zustände kosten Speicher).

Öffentliche Schnittstelle (spiegelt learning/agent.py):
  bin_values_for_qtable(obs)           -> Zustands-Key (Folie 4 / 43)
  q_values(table, state)               -> die 20 Q-Werte eines Zustands (Folie 25)
  greedy_highest_q_action_for_state    -> beste Aktion = Policy (Folie 28)
  epsilon_greedy(...)                  -> Exploration/Exploitation (Folie 41)
  update(table, s, a, r, ...)          -> Temporal-Difference-Update (Folie 38)
  save_table / load_qtable             -> Persistenz
  act(table, obs)                      -> greedy Aktion = Policy anwenden (Folie 28)
  forward_trace(table, obs)            -> (q_values, hidden) fürs Inspect-Panel
  train_qtable(...)                    -> Lernschleife "FOR EACH Episode" (Folie 40)

Wird etwas davon woanders gebraucht? Importieren — nicht neu implementieren.
"""
from __future__ import annotations

import json
import os
import time
from queue import Full, Queue

import numpy as np

from f1_rl.config import (
    EPSILON_MIN, EPSILON_START, EXPLORE_FRAC, GAMMA,
    MODEL_DIR, SIM_SPEED_AUTO_CAP, STEPS_PER_GEN,
)
from f1_rl.simulation.environment import N_ACTIONS

# ── Konfiguration der Q-Tabelle (das hier IST die ganze "Architektur") ─────────
# Welche der 14 Beobachtungs-Slots den ZUSTAND s bilden (Folie 4). Alle 14 mit echter
# Auflösung sind aussichtslos (Fluch der Dimensionalität: N_BINS**14 Zustände, Folie 41),
# daher fallen die schwachen Positions-/Progress-Signale weg, behalten wird, was lenkt:
#   [2] heading error,  [3] speed,  [7..13] die 7 Lidar-Strahlen.
FEATURE_IDX  = np.array([2, 3, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
# Natürlicher Wertebereich jedes behaltenen Merkmals, damit jede Stufe Information
# trägt (heading error ist [-1, 1]; speed und rays sind [0, 1]). Aligned mit FEATURE_IDX.
FEATURE_LOW  = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
FEATURE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
N_BINS = 6          # Diskretisierungsstufen je Merkmal (Folie 43-47). Höher = feiner, aber langsamer

# Lern-Hyperparameter
ALPHA = 0.15        # Lernrate α des Temporal-Difference-Updates (Folie 38, 0 < α ≤ 1)
# Hinweis: GAMMA = Discount-Faktor γ (Folie 22, 0 < γ < 1) kommt aus config.py.

# Persistenz — getrennt von model.npy des DQN, damit sich beide nie überschreiben.
QTABLE_PATH = os.path.join(MODEL_DIR, "qtable.npz")
QSTATE_PATH = os.path.join(MODEL_DIR, "qtable_state.json")

# Eine Q-Tabelle ist genau das: Zustands-Key (Tupel aus N Merkmals-Bins) -> Q-Wert-Vektor.
QState = tuple                 # = ein Zustand s ∈ S (Folie 4), diskretisiert
QTable = dict                  # = die Q-Tabelle (Folie 25): dict[QState, np.ndarray]

__all__ = [
    "bin_values_for_qtable", "init_q_table", "q_row", "greedy_action", "epsilon_greedy_policy", "temporal_difference_update",
    "save_q_table", "load_q_table", "policy_action", "inspect_trace", "q_learning_loop",
]


# ── Zustands-Diskretisierung (Folie 4 «Zustand s ∈ S» / Folie 43 «State Space») ─


# Folie 4 «Zustand s ∈ S» / Folie 43 «State Space»  → rename: discretize_state(obs)
def bin_values_for_qtable(obs: np.ndarray) -> tuple[int, ...]:
    """Macht aus dem 14-Float-Beobachtungsvektor einen diskreten, hashbaren ZUSTAND s.

    Jedes behaltene Merkmal wird über seinen natürlichen Bereich nach [0, 1] normiert
    und in eine von N_BINS Stufen gelegt, sodass zwei physikalisch ähnliche Situationen
    auf dieselbe Tabellen-Zeile (denselben Zustand) fallen.
    """
    feats = np.asarray(obs, dtype=np.float64)[FEATURE_IDX]
    norm = (feats - FEATURE_LOW) / (FEATURE_HIGH - FEATURE_LOW)
    bins = np.clip((norm * N_BINS).astype(np.int64), 0, N_BINS - 1)
    return tuple(int(b) for b in bins)


# ── Tabellen-Operationen (das dict IST die Q-Tabelle; diese Funktionen wirken darauf) ──

# Folie 40 «Initialisiere Q-Tabelle»  → rename: init_q_table()
def init_q_table() -> QTable:
    """Eine leere Q-Tabelle (Zustände werden beim ersten Zugriff lazy ergänzt).

    Folie 40 initialisiert "zufallsbasiert"; hier mit Nullen (siehe q_values).
    """
    return {}


# Folie 25 «Q-Tabelle» / Folie 24 «Q-Funktion Q(s,·)»  → rename: q_row(table, state)
def q_row(table: QTable, state: tuple[int, ...]) -> np.ndarray:
    # Q-Wert-Vektor Q(s,·) des Zustands; beim ersten Zugriff als Nullen angelegt
    row = table.get(state)
    if row is None:
        row = np.zeros(N_ACTIONS, dtype=np.float32)
        table[state] = row
    return row


# Folie 28 «Optimale Policy  π(s) = argmax_a Q(s,a)»  → rename: greedy_action(table, state)
def greedy_action(table: QTable, state: tuple[int, ...]) -> int:
    # Index der Aktion mit maximalem Q-Wert im Zustand s = die (gierige) Policy π(s)
    return int(np.argmax(q_row(table, state)))


# Folie 41: Damit jeder Zustand besucht werden kann
def epsilon_greedy_policy(table: QTable, state: tuple[int, ...], epsilon: float,
                   rng: np.random.Generator) -> int:
    """Mit Wahrscheinlichkeit 1-ε die Policy-Aktion (argmax, Folie 28), sonst zufällig.

    Die Zufallswahl (Exploration) ist die praktische Ergänzung zu Folie 40 Schritt 1,
    damit — wie auf Folie 41 gefordert — wirklich jeder Zustand besucht werden kann.
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return greedy_action(table, state)


# Folie 38 «Temporal Difference Model» / Folie 40 Schritt 5 «Update von Q(s,a)»
#   → rename: temporal_difference_update(table, s, a, r, s_next, done)
def temporal_difference_update(table: QTable, s: tuple[int, ...], a: int, r: float,
           s_next: tuple[int, ...], done: bool) -> None:
    """Der klassische tabellarische Q-Learning-Schritt (Folie 38, ändert ``table`` in place):

        Q(s,a) ← Q(s,a) + α · [ r + γ · maxₐ′ Q(s′,a′) − Q(s,a) ]

    Zuordnung zur Folie 38:
        ALPHA                         = Lernrate α
        GAMMA                         = Discount-Faktor γ
        np.max(q_values(.., s_next))  = maxₐ′ Q(s′,a′)   (bester Folgewert)
        row[a]                        = Q(s,a)           (aktuelle Zelle)
        target                        = r + γ · maxₐ′ Q(s′,a′)   (Bellman-Ziel, Folie 32)
        (target - row[a])             = Δ, der TD-Fehler (Folie 37)

    Im Zielzustand (done, s = s_goal, Folie 44) gibt es keine Zukunft: kein Bootstrap-
    Term γ·maxₐ′Q(s′,a′), das Ziel ist dann nur ``r`` (entspricht Q(s_goal,·) = 0).
    """
    target = r if done else r + GAMMA * float(np.max(q_row(table, s_next)))
    row = q_row(table, s)
    row[a] += ALPHA * (target - row[a])


# ── Persistenz ──────────────────────────────────────────────────────────────────

# (keine Folie — reine Speicherung)  → rename: save_q_table(...)
def save_q_table(table: QTable, path: str = QTABLE_PATH) -> None:
    """Schreibt die Q-Tabelle als komprimiertes .npz nach ``path`` (Zustände + Werte)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if table:
        states = np.array(list(table.keys()), dtype=np.int16)
        values = np.array(list(table.values()), dtype=np.float32)
    else:
        states = np.empty((0, len(FEATURE_IDX)), dtype=np.int16)
        values = np.empty((0, N_ACTIONS), dtype=np.float32)
    np.savez_compressed(path, states=states, values=values)


# (keine Folie — reine Speicherung)  → rename: load_q_table(...)
def load_q_table(path: str = QTABLE_PATH) -> QTable:
    """Lädt eine zuvor mit :func:`save_table` geschriebene Q-Tabelle."""
    table = init_q_table()
    data = np.load(path)
    for key, row in zip(data["states"], data["values"]):
        table[tuple(int(k) for k in key)] = row.astype(np.float32)
    return table


# ── Inferenz-Helfer (spiegeln learning/agent.py, damit Anzeige/Inspect andocken) ─

# Folie 28 «Policy anwenden»  → rename: policy_action(table, obs)
def policy_action(table: QTable, obs: np.ndarray) -> int:
    """Liefert für eine Beobachtung die Policy-Aktion π(s) = argmax_a Q(s,a) (Folie 28)."""
    return greedy_action(table, bin_values_for_qtable(obs))


# (Inspect-Panel-Helfer, keine Folie)  → rename: inspect_trace(table, obs)
def inspect_trace(table: QTable, obs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Liefert (q_values, hidden) fürs Live-Inspect-Panel.

    Die zurückgegebene q-Zeile ist die echte "Q-Tabellen-Zeile" Q(s,·) dieses Zustands,
    die das UI visualisiert. ``hidden`` zweckentfremdet den Netz-Aktivierungs-Kanal, um
    stattdessen den diskretisierten Zustand zu zeigen: eine Pseudo-Schicht mit dem Bin-
    Index jedes behaltenen Merkmals — so sieht man, in welchen Topf die Situation fällt.
    """
    state = bin_values_for_qtable(obs)
    q = q_row(table, state).copy()
    hidden = [np.asarray(state, dtype=np.float32)]
    return q, hidden


# ── Live-Lernschleife (Folie 40 «FOR EACH Episode …», signaturkompatibel zu trainer.train) ──

def _write_state(circuit: str, steps: int, total: int, epsilon: float,
                 episode: int, best: float, n_states: int) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(QSTATE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "backend": "qtable",
            "timesteps_done": steps,
            "total_timesteps": total,
            "circuit": circuit,
            "epsilon": epsilon,
            "episode": episode,
            "best_fitness": best,
            "n_states": n_states,
        }, f, indent=2)


# Folie 40 «FOR EACH Episode … UNTIL s == s_goal»  → rename: q_learning_loop(...)
def q_learning_loop(
    track_query: str = "Circuit de Monaco",
    geojson_fallback: str | None = None,
    total_timesteps: int = 0,
    save_path: str | None = None,
    render_queue: Queue | None = None,
    stats_queue: Queue | None = None,
    line_queue: Queue | None = None,
    speed_holder: list | None = None,
    inspect_holder: list | None = None,
    inspect_queue: Queue | None = None,
    auto_speed: bool = False,
    track=None,
    resume: bool = False,
    steps_per_gen: int = STEPS_PER_GEN,
    total_gens: int | None = None,
    evolution_mode: str = "qtable",   # angenommen & ignoriert (kein GA im Tabellen-Modus)
    use_rays: bool = True,
    cancel_event=None,
) -> tuple[QTable, object]:
    """Einzel-Agent-Q-Learning mit Live-Ansicht (die Schleife von Folie 40).

    Gleiche Umgebung, gleiches WebSocket-Protokoll und Inspect-Panel wie der DQN-Trainer,
    aber:
      - ein Auto lernt online (keine Population, kein genetischer Algorithmus),
      - die Policy ist eine Q-Tabelle (plain dict) statt eines neuronalen Netzes,
      - ``speed_holder`` steuert Lern- *und* Ansichtsgeschwindigkeit (Sub-Steps pro Frame).

    Nicht genutzte, aber akzeptierte kwargs (evolution_mode, total_timesteps, …) halten die
    Signatur austauschbar mit trainer.train, damit session.py beide aufrufen kann.
    """
    from f1_rl.server.protocol import inspect_to_dict
    from f1_rl.simulation.environment import CarFrame, F1Env, REWARD_PARTS
    from f1_rl.simulation.track_loader import load_track

    os.makedirs(MODEL_DIR, exist_ok=True)
    if save_path is None:
        save_path = QTABLE_PATH

    if track is None:
        track = load_track(track_query, geojson_fallback_path=geojson_fallback)
    print(f"[qtable] Track: {track.name}  {track.total_length_m:.0f} m")

    # Gesamt-Schrittbudget = Horizont fürs ε-Annealing. Wir nutzen die steps×gens-Regler.
    if total_gens is None:
        total_gens = 200
    total_steps = max(1, steps_per_gen * total_gens)
    print(f"[qtable] Tabular Q-learning  features={len(FEATURE_IDX)}  bins={N_BINS}"
          f"  alpha={ALPHA}  gamma={GAMMA}  budget={total_steps:,} steps")

    # ── Initialisiere Q-Tabelle (Folie 40) / Resume ──────────────────────────
    table = init_q_table()
    episode = 0
    best_fitness = -1e9
    if resume and os.path.exists(save_path):
        table = load_q_table(save_path)
        if os.path.exists(QSTATE_PATH):
            with open(QSTATE_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            episode = int(saved.get("episode", 0))
            best_fitness = float(saved.get("best_fitness", -1e9))
        print(f"[qtable] Resumed: {len(table)} states, episode={episode}, best={best_fitness:+.1f}")

    rng = np.random.default_rng()
    env = F1Env(track=track, render_mode=None, use_rays=use_rays)
    obs, _ = env.reset()                       # Umgebung im Initialzustand (Folie 6)
    state = bin_values_for_qtable(obs)         # Initialer Zustand s (Folie 40)

    step_count = 0                             # Iterationszähler t (Folie 12 / 40 Schritt 3)
    fitness_hist: list[float] = []   # letzte Episoden-Scores, für den Mittelwert-Stat
    ep_path: list[tuple] = []        # (x, y, speed, throttle) der laufenden Episode, für die Linie
    best_path: list[tuple] | None = None

    dt = 1.0 / 60.0
    frame_no = 0
    INSPECT_EVERY = 6   # Q-Zeile ~10×/s streamen, nicht 60×/s
    SAVE_EVERY_EPS = 25

    def _emit_frame() -> None:
        frame = [CarFrame(
            x=env.x_m, y=env.y_m, heading=env.heading, speed=env.speed_ms,
            throttle=env._last_throttle, checkpoint=env._checkpoint_idx,
            progress=env.progress, lap=env.lap_count,
            rays=tuple(env._cast_rays()), pack=0, score=env.episode_reward,
            reward_parts=tuple(env.reward_parts[k] for k in REWARD_PARTS),
            generation=episode, a_long=env._a_long, a_lat=env._a_lat,
        )]
        if render_queue is not None:
            try:
                while not render_queue.empty():
                    render_queue.get_nowait()
                render_queue.put_nowait(frame)
            except (Full, Exception):   # noqa: BLE001
                pass

    def _emit_line(path: list[tuple]) -> None:
        if line_queue is None or not path:
            return
        try:
            while not line_queue.empty():
                line_queue.get_nowait()
            line_queue.put_nowait(list(path))
        except (Full, Exception):       # noqa: BLE001
            pass

    while cancel_event is None or not cancel_event.is_set():
        t0 = time.perf_counter()
        frame_no += 1

        # Wie viele Lernschritte in diesen einen gerenderten Frame fallen. Wie beim
        # DQN-Display: größer = schnelleres Lernen und lebendigere Ansicht.
        if auto_speed:
            n_sub = SIM_SPEED_AUTO_CAP
        elif speed_holder is not None:
            n_sub = max(1, int(speed_holder[0]))
        else:
            n_sub = 4

        for _ in range(n_sub):
            # ε-Schedule: linear über die ersten EXPLORE_FRAC des Budgets absenken.
            frac = step_count / total_steps
            epsilon = (EPSILON_START - (EPSILON_START - EPSILON_MIN) * frac / EXPLORE_FRAC
                       if frac < EXPLORE_FRAC else EPSILON_MIN)

            # Folie 40 Schritt 1: Agent wählt Aktion a (hier ε-greedy = mit Exploration).
            action = epsilon_greedy_policy(table, state, epsilon, rng)
            # Folie 40 Schritt 2/4: Aktion ausführen, Reward r und Folgezustand s' erhalten.
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = bin_values_for_qtable(obs)        # s' = T(s,a) diskretisiert (Folie 10)
            done = terminated or truncated                 # Zielzustand erreicht? (Folie 44)
            # Folie 40 Schritt 5: Update von Q(s,a) gemäß Temporal Difference Model (Folie 38).
            temporal_difference_update(table, state, action, reward, next_state, done)
            state = next_state                             # Folie 40 Schritt 6: s = s'
            step_count += 1                                # Folie 40 Schritt 3: t++
            ep_path.append((env.x_m, env.y_m, env.speed_ms, env._last_throttle))

            if done:
                episode += 1
                score = env.episode_reward                 # ~ Total Reward R_t der Episode (Folie 21)
                fitness_hist.append(score)
                if len(fitness_hist) > 50:
                    fitness_hist.pop(0)

                if score > best_fitness:
                    best_fitness = score
                    best_path = list(ep_path)
                    _emit_line(best_path)
                    save_q_table(table, save_path)

                # Periodische Stats ans UI (pro Episode = pro "Generation").
                if stats_queue is not None:
                    try:
                        stats_queue.put_nowait({
                            "timesteps":    step_count,
                            "epsilon":      epsilon,
                            "episode":      episode,
                            "generation":   episode,
                            "best_fitness": best_fitness,
                            "mean_fitness": float(np.mean(fitness_hist)),
                            "n_packs":      len(table),   # Slot zweckentfremdet: zeigt Tabellengröße
                            "top_scores":   [],
                        })
                    except Full:
                        pass

                if episode % SAVE_EVERY_EPS == 0:
                    save_table(table, save_path)
                    _write_state(track.name, step_count, total_steps, epsilon,
                                 episode, best_fitness, len(table))

                obs, _ = env.reset()                       # neue Episode: Initialzustand (Folie 40)
                state = bin_values_for_qtable(obs)
                ep_path = []

        _emit_frame()

        # Inspect: aktuelle Q-Zeile Q(s,·) + diskretisierten Zustand für Auto 0 streamen.
        if (inspect_queue is not None and inspect_holder is not None
                and inspect_holder[0] == 0 and frame_no % INSPECT_EVERY == 0):
            q, hidden = inspect_trace(table, obs)
            try:
                while not inspect_queue.empty():
                    inspect_queue.get_nowait()
                inspect_queue.put_nowait(
                    inspect_to_dict(0, obs, q, hidden, int(q.argmax())))
            except Exception:           # noqa: BLE001
                pass

        # Stoppen, sobald das Schrittbudget aufgebraucht ist.
        if step_count >= total_steps:
            print(f"[qtable] Budget reached ({step_count:,} steps).")
            break

        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

    env.close()
    save_q_table(table, save_path)
    _write_state(track.name, step_count, total_steps, EPSILON_MIN,
                 episode, best_fitness, len(table))
    print(f"[qtable] Done. {len(table)} states learned, best={best_fitness:+.1f} -> {save_path}")
    return table, track


# ── Standalone-Runner (ohne Web-UI) ──────────────────────────────────────────────

if __name__ == "__main__":
    # Headless-Konsolentraining, z. B.:  python -m f1_rl.learning.qtable
    q_learning_loop(steps_per_gen=5_000, total_gens=200, resume=False)