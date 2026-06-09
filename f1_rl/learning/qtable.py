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
    Episode / Lernschleife    (Folie 40)  q_learning_loop(...)
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
  q_learning_loop(...)                 -> Lernschleife "FOR EACH Episode" (Folie 40)

Wird etwas davon woanders gebraucht? Importieren — nicht neu implementieren.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from queue import Full, Queue

import numpy as np

from f1_rl.config import (
    EPSILON_MIN, EPSILON_START, EXPLORE_FRAC, GAMMA,
    HALL_OF_FAME_K, MODEL_DIR, MUTATION_NOISE, MUTATION_RATE,
    N_BEST_CLONES, QTABLE_N_POP, SAVE_EVERY, SIM_SPEED_AUTO_CAP,
    STAGNATION_GENS, STEPS_PER_GEN,
)
from f1_rl.learning.genetics import rank_select
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
    "crossover_tables", "mutate_table", "run_qtable_worker",
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


# ── Genetische Operatoren auf Q-Tabellen (Pendant zu genetics.py für Vektoren) ──
# genetics.crossover/mutate wirken auf flache Gewichtsvektoren; eine Q-Tabelle ist
# ein sparse dict[Zustand -> Q-Zeile], daher diese Tabellen-Varianten. rank_select
# aus genetics.py ist generisch über Listen und wird direkt wiederverwendet.

def _copy_table(t: QTable) -> QTable:
    """Tiefe Kopie einer Q-Tabelle (Zeilen werden kopiert, nicht geteilt)."""
    return {k: v.copy() for k, v in t.items()}


def crossover_tables(ta: QTable, tb: QTable) -> QTable:
    """Uniformes Crossover je Zustand: jede Q-Zeile stammt von einem zufälligen Elternteil.

    Spiegelt genetics.crossover (gen-weise Wahl) auf Zeilen-Ebene. Zustände, die nur ein
    Elternteil kennt, werden unverändert übernommen — so geht gelerntes Wissen nicht verloren.
    """
    child: QTable = {}
    for k in set(ta) | set(tb):
        ra, rb = ta.get(k), tb.get(k)
        if ra is None:
            child[k] = rb.copy()
        elif rb is None:
            child[k] = ra.copy()
        else:
            child[k] = (ra if np.random.random() < 0.5 else rb).copy()
    return child


def mutate_table(t: QTable, rate: float, noise: float) -> QTable:
    """Addiert Gauß-Rauschen auf einen Bruchteil (`rate`) aller Q-Werte (Pendant zu genetics.mutate)."""
    child: QTable = {}
    for k, row in t.items():
        r = row.copy()
        mask = np.random.rand(len(r)) < rate
        if mask.any():
            r[mask] += np.random.normal(0, noise, int(mask.sum())).astype(np.float32)
        child[k] = r
    return child


# ── Worker: trainiert EINE Tabelle online, bewertet sie greedy (läuft im Prozess-Pool) ──

def run_qtable_worker(args: tuple) -> tuple[float, QTable, float, float]:
    """Tabellen-Pendant zu worker.run_worker: online Q-Learning + greedy Eval.

    1. trainiert die geerbte Tabelle ``n_steps`` Schritte ε-greedy (Folie 40/38),
    2. spielt eine greedy Eval-Episode (Policy π = argmax) und misst die Fitness,
    3. gibt ``(fitness, table, end_x, end_y)`` zurück — der GA in q_learning_loop selektiert/kreuzt/mutiert.
    """
    inherited_table, track, n_steps, epsilon, eval_steps, use_rays = args

    # Deferred import (spawn-sicher auf Windows, wie worker.py).
    from f1_rl.simulation.environment import F1Env

    rng = np.random.default_rng()
    table = _copy_table(inherited_table)        # die geerbte Tabelle nicht in place verändern

    env = F1Env(track=track, render_mode=None, use_rays=use_rays)
    obs, _ = env.reset()
    state = bin_values_for_qtable(obs)

    # ── Lernphase (Folie 40 «FOR EACH Episode …») ────────────────────────────
    for _ in range(n_steps):
        action = epsilon_greedy_policy(table, state, epsilon, rng)
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = bin_values_for_qtable(obs)
        done = terminated or truncated
        temporal_difference_update(table, state, action, reward, next_state, done)
        state = next_state
        if done:
            obs, _ = env.reset()
            state = bin_values_for_qtable(obs)

    # ── Greedy-Eval (eine Episode, Fitness = aufsummierter Reward) ───────────
    eval_reward = 0.0
    obs, _ = env.reset()
    for _ in range(eval_steps):
        a = policy_action(table, obs)
        obs, r, term, trunc, _ = env.step(a)
        eval_reward += r
        if term or trunc:
            break
    end_x, end_y = env.x_m, env.y_m
    env.close()
    return eval_reward, table, end_x, end_y


# ── Greedy-Replay einer Tabelle (für die Racing-Line; Pendant zu trainer._record_replay) ──

def _record_table_replay(table: QTable, track, max_steps: int, use_rays: bool) -> np.ndarray:
    from f1_rl.simulation.environment import F1Env
    env = F1Env(track=track, render_mode=None, use_rays=use_rays)
    obs, _ = env.reset()
    frames: list[tuple] = []
    for _ in range(max_steps):
        a = policy_action(table, obs)
        obs, _, terminated, _, _ = env.step(a)
        frames.append((env.x_m, env.y_m, env.heading, env.speed_ms,
                       env._last_throttle, env.progress))
        if terminated:
            break
    env.close()
    return np.array(frames, dtype=np.float32)


# ── Live-Anzeige der ganzen Population (Pendant zu trainer._display_thread) ──

def _qtable_display_thread(pop_holder: list, stop_event: threading.Event, track,
                           render_queue: Queue, use_rays: bool = True,
                           speed_holder: list | None = None, gen_holder: list | None = None,
                           inspect_holder: list | None = None,
                           inspect_queue: Queue | None = None,
                           table_queue: Queue | None = None) -> None:
    """Animiert alle Population-Tabellen ~60 fps (Policy = argmax) und streamt Frames.

    Soft-Swap wie beim DQN-Display: bei neuer Generation fährt jedes Auto mit seiner
    aktuellen Tabelle weiter bis zum nächsten Crash und übernimmt dann die neue Tabelle
    (kein Massen-Teleport). MAX_DISPLAY_LAG verhindert zu großes Nachhinken.

    Für das inspizierte Auto werden zwei Dinge gestreamt: die Q-Zeile des aktuellen
    Zustands (inspect_queue, ~10×/s) und die VOLLE Tabelle als Heatmap (table_queue, ~1×/s).
    """
    from f1_rl.server.protocol import inspect_to_dict, qtable_to_dict
    from f1_rl.simulation.environment import CarFrame, F1Env, REWARD_PARTS

    current_pop = pop_holder[0]
    n_cars = len(current_pop)
    envs = [F1Env(track=track, render_mode=None, use_rays=use_rays) for _ in range(n_cars)]
    tables: list[QTable] = list(current_pop)
    obses = [env.reset()[0] for env in envs]
    next_tables: list = [None] * n_cars

    if speed_holder is None:
        speed_holder = [1]
    if gen_holder is None:
        gen_holder = [0]
    car_gens = [int(gen_holder[0])] * n_cars
    next_gen = [int(gen_holder[0])] * n_cars
    dt = 1.0 / 60.0
    frame_no = 0
    INSPECT_EVERY = 6      # Q-Zeile ~10×/s
    TABLE_EVERY = 60       # volle Tabelle ~1×/s (große Nutzlast)
    MAX_DISPLAY_LAG = 2

    while not stop_event.is_set():
        t0 = time.perf_counter()
        frame_no += 1
        n_sub = max(1, int(speed_holder[0]))

        new_pop = pop_holder[0]
        if new_pop is not current_pop:
            current_pop = new_pop
            arriving_gen = int(gen_holder[0])
            for i in range(n_cars):
                next_tables[i] = current_pop[i]
                next_gen[i] = arriving_gen

        arriving = int(gen_holder[0])
        frames: list = []
        for i in range(n_cars):
            if next_tables[i] is not None and arriving - car_gens[i] > MAX_DISPLAY_LAG:
                tables[i] = next_tables[i]
                next_tables[i] = None
                car_gens[i] = next_gen[i]
                obses[i] = envs[i].reset()[0]
            term = False
            for _ in range(n_sub):
                a = policy_action(tables[i], obses[i])
                obses[i], _, term, _, _ = envs[i].step(a)
                if term:
                    break
            env = envs[i]
            frames.append(CarFrame(
                x=env.x_m, y=env.y_m, heading=env.heading,
                speed=env.speed_ms, throttle=env._last_throttle,
                checkpoint=env._checkpoint_idx, progress=env.progress,
                lap=env.lap_count, rays=tuple(env._cast_rays()),
                pack=0, score=env.episode_reward,
                reward_parts=tuple(env.reward_parts[k] for k in REWARD_PARTS),
                generation=car_gens[i], a_long=env._a_long, a_lat=env._a_lat,
            ))
            if term:
                obses[i] = envs[i].reset()[0]
                if next_tables[i] is not None:
                    tables[i] = next_tables[i]
                    next_tables[i] = None
                    car_gens[i] = next_gen[i]

        while not render_queue.empty():
            try:
                render_queue.get_nowait()
            except Exception:            # noqa: BLE001
                break
        try:
            render_queue.put_nowait(frames)
        except Full:
            pass

        sel = inspect_holder[0] if inspect_holder is not None else None
        valid_sel = isinstance(sel, int) and 0 <= sel < n_cars

        # Q-Zeile + diskretisierter Zustand des inspizierten Autos (~10×/s).
        if inspect_queue is not None and valid_sel and frame_no % INSPECT_EVERY == 0:
            q, hidden = inspect_trace(tables[sel], obses[sel])
            try:
                while not inspect_queue.empty():
                    inspect_queue.get_nowait()
                inspect_queue.put_nowait(
                    inspect_to_dict(sel, obses[sel], q, hidden, int(q.argmax())))
            except Exception:            # noqa: BLE001
                pass

        # Volle Tabelle des inspizierten Autos als Heatmap (~1×/s, viele Zustände).
        if table_queue is not None and valid_sel and frame_no % TABLE_EVERY == 0:
            tbl = tables[sel]
            keys = sorted(tbl.keys())            # stabile Zeilen-Reihenfolge über Updates
            payload = qtable_to_dict(sel, keys, [tbl[k] for k in keys], FEATURE_IDX, N_BINS)
            try:
                while not table_queue.empty():
                    table_queue.get_nowait()
                table_queue.put_nowait(payload)
            except Exception:            # noqa: BLE001
                pass

        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

    for env in envs:
        env.close()


# ── Generations-Bausteine (Pendant zu trainer._init_population/_update_hall_of_fame/_breed) ──

def _init_table_population(resume: bool, save_path: str, n_pop: int) -> tuple[list, int, float]:
    """Start-Population. Mit ``resume`` aus der gespeicherten Best-Tabelle geseedet
    (plus mutierte Kopien) inkl. wiederhergestellter Generation/Fitness, sonst N leere
    Tabellen ab Generation 0 (Zustände entstehen lazy beim Lernen in den Workern)."""
    if resume and os.path.exists(save_path):
        best = load_q_table(save_path)
        saved: dict = {}
        if os.path.exists(QSTATE_PATH):
            with open(QSTATE_PATH, encoding="utf-8") as f:
                saved = json.load(f)
        start_gen = int(saved.get("generation", saved.get("episode", 0)))
        best_fitness = float(saved.get("best_fitness", -1e9))
        population = [_copy_table(best)] + [
            mutate_table(best, MUTATION_RATE * 3, MUTATION_NOISE * 2) for _ in range(n_pop - 1)
        ]
        print(f"[qtable] Resumed gen={start_gen}  best_fitness={best_fitness:+.1f}  ({len(best)} states)")
        return population, start_gen, best_fitness
    return [init_q_table() for _ in range(n_pop)], 0, -1e9


def _update_table_hof(hall_of_fame: list, fitnesses: list, tables_out: list) -> None:
    """Hält die besten HALL_OF_FAME_K Tabellen aller Zeiten (best first); Eingabe best-first."""
    for fit, t in zip(fitnesses, tables_out):
        if len(hall_of_fame) < HALL_OF_FAME_K:
            hall_of_fame.append((fit, _copy_table(t)))
            hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        elif fit > hall_of_fame[-1][0]:
            hall_of_fame[-1] = (fit, _copy_table(t))
            hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        else:
            break


def _breed_table_generation(tables_out: list, fitnesses: list, hall_of_fame: list,
                            best_ever_table: QTable, stagnation_count: int,
                            stagnation_boost: float, n_pop: int) -> list:
    """Nächste Generation (Tier-Ansatz wie trainer._breed_next_generation, „classic"):
    HOF-Eliten überleben unverändert, leicht mutierte Klone des Allzeit-Besten, der Rest
    sind Crossover-Kinder mit rang-gestufter Mutation. Bei Stagnation eine frische (leere)
    Tabelle injizieren (lernt sich in den Workern neu auf)."""
    hof_tables = [t for _, t in hall_of_fame]

    new_pop = [_copy_table(t) for _, t in hall_of_fame]          # Tier 1: Eliten
    for _ in range(N_BEST_CLONES):                               # Tier 2: Best-Klone
        if len(new_pop) < n_pop:
            new_pop.append(mutate_table(best_ever_table, MUTATION_RATE * 0.3, MUTATION_NOISE * 0.3))

    n_protected = len(new_pop)                                   # Tier 3: Crossover-Kinder
    while len(new_pop) < n_pop:
        pa = (rank_select(hof_tables)
              if (hof_tables and np.random.random() < 0.5)
              else rank_select(tables_out))
        pb = rank_select(tables_out)
        child = crossover_tables(pa, pb)
        rank_factor = (len(new_pop) - n_protected) / max(n_pop - n_protected, 1)
        child = mutate_table(
            child,
            MUTATION_RATE * stagnation_boost * (0.3 + rank_factor * 0.7),
            MUTATION_NOISE * stagnation_boost * (0.3 + rank_factor * 0.7),
        )
        new_pop.append(child)

    if stagnation_count > 0 and stagnation_count % STAGNATION_GENS == 0:
        new_pop[-1] = init_q_table()
    return new_pop[:n_pop]


# ── Persistenz des Trainingsstands ────────────────────────────────────────────

def _write_state(circuit: str, steps: int, total: int, epsilon: float,
                 gen: int, best: float, n_states: int) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(QSTATE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "backend": "qtable",
            "timesteps_done": steps,
            "total_timesteps": total,
            "circuit": circuit,
            "epsilon": epsilon,
            "generation": gen,
            "best_fitness": best,
            "n_states": n_states,
        }, f, indent=2)


# ── Haupt-Generationsschleife (genetische Multi-Thread-Evolution von Q-Tabellen) ──

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
    evolution_mode: str = "qtable",   # angenommen & ignoriert (Pack-Modus gibt es hier nicht)
    use_rays: bool = True,
    cancel_event=None,
    table_queue: Queue | None = None,  # Live-Stream der vollen Tabelle (nur Q-Table-Modus)
) -> tuple[QTable, object]:
    """Genetische, parallele Q-Tabellen-Evolution (Pendant zu trainer.train).

    QTABLE_N_POP Agenten, je in einem eigenen Prozess: jeder lernt seine Tabelle online
    (Q-Learning, Folie 40/38) und wird greedy bewertet. Danach selektiert/kreuzt/mutiert
    der GA die Tabellen pro Generation. Gleiche Umgebung, gleiches WebSocket-Protokoll und
    dieselbe Live-Population-Ansicht wie der DQN-Trainer — nur die Policy ist eine Q-Tabelle.

    Nicht genutzte, aber akzeptierte kwargs (evolution_mode, …) halten die Signatur
    austauschbar mit trainer.train, damit session.py beide aufrufen kann.
    """
    from f1_rl.learning.trainer import _emit_racing_line, _eval_step_budget
    from f1_rl.simulation.track_loader import load_track

    os.makedirs(MODEL_DIR, exist_ok=True)
    if save_path is None:
        save_path = QTABLE_PATH
    if track is None:
        track = load_track(track_query, geojson_fallback_path=geojson_fallback)
    print(f"[qtable] Track: {track.name}  {track.total_length_m:.0f} m")

    if total_gens is None:
        total_gens = 200
    n_pop = QTABLE_N_POP

    # ── Population initialisieren / Resume ───────────────────────────────────
    population, start_gen, best_ever_fitness = _init_table_population(resume, save_path, n_pop)
    best_ever_table = _copy_table(population[0])
    hall_of_fame: list[tuple[float, QTable]] = []
    top_10: list[tuple[float, int]] = []
    stagnation_count = 0
    prev_best_eval = -1e9
    steps_so_far = 0

    pop_holder = [list(population)]    # Display-Thread liest pop_holder[0]
    gen_holder = [start_gen]
    stop_event = threading.Event()

    print(f"[qtable] Genetic tabular Q-learning  {n_pop} agents x {steps_per_gen} steps/gen "
          f"x {total_gens} gens  features={len(FEATURE_IDX)} bins={N_BINS} alpha={ALPHA} gamma={GAMMA}")

    if render_queue is not None:
        disp = threading.Thread(
            target=_qtable_display_thread,
            args=(pop_holder, stop_event, track, render_queue, use_rays, speed_holder,
                  gen_holder, inspect_holder, inspect_queue, table_queue),
            daemon=True,
        )
        disp.start()
    else:
        disp = None

    n_workers = min(n_pop, os.cpu_count() or 4)
    eval_steps = _eval_step_budget(track)
    print(f"[qtable] Spawning {n_workers} worker processes  eval={eval_steps} steps/episode")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for gen in range(start_gen, start_gen + total_gens):
            if cancel_event is not None and cancel_event.is_set():
                print(f"[qtable] Stop requested — ending at generation {gen}")
                break

            gen_t0 = time.perf_counter()
            frac = (gen - start_gen) / total_gens
            epsilon = (EPSILON_START - (EPSILON_START - EPSILON_MIN) * frac / EXPLORE_FRAC
                       if frac < EXPLORE_FRAC else EPSILON_MIN)

            # 1. Population auswerten (ein Worker-Prozess je Tabelle), best first.
            results = list(executor.map(
                run_qtable_worker,
                [(t, track, steps_per_gen, epsilon, eval_steps, use_rays) for t in population],
                chunksize=1,
            ))
            fitnesses = [r[0] for r in results]
            tables_out = [r[1] for r in results]
            order = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            fitnesses = [fitnesses[i] for i in order]
            tables_out = [tables_out[i] for i in order]

            # 2. Allzeit-Bestenliste (Hall of Fame) über Tabellen.
            _update_table_hof(hall_of_fame, fitnesses, tables_out)
            if hall_of_fame and hall_of_fame[0][0] > best_ever_fitness:
                best_ever_fitness = hall_of_fame[0][0]
                best_ever_table = _copy_table(hall_of_fame[0][1])
                _emit_racing_line(line_queue, _record_table_replay(
                    best_ever_table, track, eval_steps, use_rays))

            # Sortierte Population + Generation an die Live-Anzeige übergeben.
            gen_holder[0] = gen
            pop_holder[0] = list(tables_out)

            # 3. Leaderboard (Top-10).
            if len(top_10) < 10 or fitnesses[0] > top_10[-1][0]:
                top_10.append((fitnesses[0], gen))
                top_10.sort(key=lambda x: x[0], reverse=True)
                if len(top_10) > 10:
                    top_10.pop()

            # 4. Stagnation → Mutation hochregeln.
            if fitnesses[0] > prev_best_eval + 0.5:
                prev_best_eval = fitnesses[0]
                stagnation_count = 0
            else:
                stagnation_count += 1
            stagnation_boost = 1.0 + min(2.0, stagnation_count / STAGNATION_GENS)

            # 5. Nächste Generation züchten.
            population = _breed_table_generation(
                tables_out, fitnesses, hall_of_fame, best_ever_table,
                stagnation_count, stagnation_boost, n_pop)

            # 6. Stats + Checkpoint.
            best_states = len(tables_out[0])
            steps_so_far = (gen - start_gen + 1) * n_pop * steps_per_gen
            if stats_queue is not None:
                try:
                    stats_queue.put_nowait({
                        "timesteps":    steps_so_far,
                        "epsilon":      epsilon,
                        "episode":      gen,
                        "generation":   gen,
                        "best_fitness": fitnesses[0],
                        "mean_fitness": float(np.mean(fitnesses)),
                        "n_packs":      best_states,   # Slot zweckentfremdet: Zustände der besten Tabelle
                        "top_scores":   list(top_10),
                    })
                except Full:
                    pass

            if (gen - start_gen + 1) % SAVE_EVERY == 0 or gen == start_gen + total_gens - 1:
                save_q_table(best_ever_table, save_path)
                _write_state(track.name, steps_so_far, total_timesteps or steps_so_far,
                             epsilon, gen, best_ever_fitness, len(best_ever_table))
                stag_str = f"  stag={stagnation_count}×{stagnation_boost:.1f}" if stagnation_count > 0 else ""
                print(f"[gen {gen:>4d}/{start_gen + total_gens - 1}]  ε={epsilon:.3f}"
                      f"  eval={fitnesses[0]:+.1f}  mean={float(np.mean(fitnesses)):+.1f}"
                      f"  hof_best={best_ever_fitness:+.1f}  states_best={best_states}{stag_str}")

            # 7. Auto-Speed: Animation an die Generations-Rechenzeit koppeln.
            if auto_speed and speed_holder is not None:
                gen_seconds = time.perf_counter() - gen_t0
                if gen_seconds > 0:
                    target = math.ceil(eval_steps / (60.0 * gen_seconds))
                    speed_holder[0] = max(1, min(SIM_SPEED_AUTO_CAP, target))

    stop_event.set()
    if disp is not None:
        disp.join(timeout=2.0)

    save_q_table(best_ever_table, save_path)
    _write_state(track.name, steps_so_far, total_timesteps or steps_so_far,
                 EPSILON_MIN, start_gen + total_gens - 1, best_ever_fitness, len(best_ever_table))
    print(f"[qtable] Done. best={best_ever_fitness:+.1f}  {len(best_ever_table)} states -> {save_path}")
    return best_ever_table, track


# ── Standalone-Runner (ohne Web-UI) ──────────────────────────────────────────────

if __name__ == "__main__":
    # Headless-Konsolentraining, z. B.:  python -m f1_rl.learning.qtable
    q_learning_loop(steps_per_gen=5_000, total_gens=200, resume=False)