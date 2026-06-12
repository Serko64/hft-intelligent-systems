from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from queue import Full, Queue

import numpy as np

from f1_rl.config import (
    EPSILON_MIN, GAMMA, MODEL_DIR, MUTATION_NOISE, MUTATION_RATE,
    N_BEST_CLONES, QTABLE_N_POP, SAVE_EVERY,
    STAGNATION_GENS, STEPS_PER_GEN,
)
from f1_rl.learning.genetics import rank_select
from f1_rl.simulation.environment import N_ACTIONS

# Features die gebinned werden sollen (ihre indexe)
FEATURE_IDX  = np.array([2, 3, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64)

# Wertebereiche der Features für Bins
FEATURE_LOW  = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
FEATURE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
N_BINS = 6

LEARNING_RATE = 0.15

# Speicherung
QTABLE_PATH = os.path.join(MODEL_DIR, "qtable.npz")
QSTATE_PATH = os.path.join(MODEL_DIR, "qtable_state.json")

# Map von tuple und dict zu var für übersicht
QState = tuple
QTable = dict

__all__ = [
    "bin_values_for_qtable", "init_q_table", "q_row", "greedy_action", "epsilon_greedy_policy", "temporal_difference_update",
    "save_q_table", "load_q_table", "policy_action", "inspect_trace", "q_learning_loop",
    "crossover_tables_biased", "mutate_table", "run_qtable_worker",
]

def bin_values_for_qtable(obs: np.ndarray) -> tuple[int, ...]:
    feats = np.asarray(obs, dtype=np.float64)[FEATURE_IDX]
    norm = (feats - FEATURE_LOW) / (FEATURE_HIGH - FEATURE_LOW)
    bins = np.clip((norm * N_BINS).astype(np.int64), 0, N_BINS - 1)
    return tuple(int(b) for b in bins)

# Basis qTable
def init_q_table() -> QTable:
    return {}


# QTable Row generation und getter
def q_row(table: QTable, state: tuple[int, ...]) -> np.ndarray:
    row = table.get(state)
    if row is None:
        row = np.zeros(N_ACTIONS, dtype=np.float32)
        table[state] = row
    return row


# Verwendung der Aktion die den maximalen QWert hat über den state der rauskam => greedy
def greedy_action(table: QTable, state: tuple[int, ...]) -> int:
    return int(np.argmax(q_row(table, state)))


_ZERO_ROW = np.zeros(N_ACTIONS, dtype=np.float32)


def q_row_readonly(table: QTable, state: tuple[int, ...]) -> np.ndarray:
    row = table.get(state)
    return _ZERO_ROW if row is None else row


def greedy_action_readonly(table: QTable, state: tuple[int, ...]) -> int:
    return int(np.argmax(q_row_readonly(table, state)))


def policy_action_readonly(table: QTable, obs: np.ndarray) -> int:
    """Policy-Aktion π(s) OHNE die Tabelle zu verändern (für Anzeige/Eval)."""
    return greedy_action_readonly(table, bin_values_for_qtable(obs))


# Damit jeder Zustand besucht werden kann
def epsilon_greedy_policy(table: QTable, state: tuple[int, ...], epsilon: float,
                   rng: np.random.Generator) -> int:
    #Mit Wahrscheinlichkeit 1-ε die Policy-Aktion, sonst zufällig.
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return greedy_action(table, state)


# Q Learning Schritt
def temporal_difference_update(table: QTable, state: tuple[int, ...], action: int,
                               reward: float, next_state: tuple[int, ...],
                               done: bool) -> None:
    # Der klassische tabellarische Q-Learning-Schritt: Q(s,a) = Q(s,a) + α · [ r + γ · maxₐ′ Q(s′,a′) − Q(s,a) ]

    if done:
        target = reward  # Episode vorbei → kein zukünftiger Wert mehr
    else:
        best_next_value = float(np.max(q_row(table, next_state)))
        target = reward + GAMMA * best_next_value
    row = q_row(table, state)
    row[action] += LEARNING_RATE * (target - row[action])


# ── Persistenz ──────────────────────────────────────────────────────────────────

# Speicherung
def save_q_table(table: QTable, path: str = QTABLE_PATH) -> None:
    #Schreibt die Q-Tabelle als komprimiertes .npz nach path (Zustände + Werte)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if table:
        states = np.array(list(table.keys()), dtype=np.int16)
        values = np.array(list(table.values()), dtype=np.float32)
    else:
        states = np.empty((0, len(FEATURE_IDX)), dtype=np.int16)
        values = np.empty((0, N_ACTIONS), dtype=np.float32)
    np.savez_compressed(path, states=states, values=values)


# Speicherung
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
    q = q_row_readonly(table, state).copy()   # nur lesen — Inspect darf nicht „lernen"
    hidden = [np.asarray(state, dtype=np.float32)]
    return q, hidden


# ── Genetische Operatoren auf Q-Tabellen (Pendant zu genetics.py für Vektoren) ──
# genetics.crossover/mutate wirken auf flache Gewichtsvektoren; eine Q-Tabelle ist
# ein sparse dict[Zustand -> Q-Zeile], daher diese Tabellen-Varianten. rank_select
# aus genetics.py ist generisch über Listen und wird direkt wiederverwendet.

def _copy_table(table: QTable) -> QTable:
    """Tiefe Kopie einer Q-Tabelle (Zeilen werden kopiert, nicht geteilt)."""
    return {state: row.copy() for state, row in table.items()}


# Wichtige Invariante (Performance, vermeidet Lag am Generationsübergang):
# Eine fertige Tabelle wird NIE mehr in place verändert — nur Worker mutieren Q-Zeilen,
# und die kopieren ihre Eingabe vorab per _copy_table. Anzeige/Inspect/Eval/Speichern
# lesen nur (read-only Helfer). Deshalb dürfen Crossover/Mutation unveränderte Q-Zeilen
# einfach *referenzieren* statt zu kopieren (copy-on-write). Das spart pro Generation
# tausende winzige NumPy-Allokationen — genau die Pure-Python-Arbeit, die sonst kurz das
# GIL hält und die Frame-Broadcasts blockiert.

def crossover_tables_biased(table_a: QTable, table_b: QTable,
                            score_a: float, score_b: float) -> QTable:
    """Uniformes gewichtetes Crossover: Bessere Eltern vererben mehr Zeilen."""

    # 1. Softmax-Berechnung für das Gewicht von A
    # (Der Abzug von max_score verhindert numerische Overflows bei sehr großen Scores)
    max_score = max(score_a, score_b)
    exp_a = np.exp(score_a - max_score)
    exp_b = np.exp(score_b - max_score)
    weight_a = exp_a / (exp_a + exp_b)

    child: QTable = {}

    # 2. Crossover-Logik: pro Zustand entscheidet der Zufall, von welchem
    #    Elternteil die Q-Zeile kommt — der bessere gewinnt öfter.
    for state in set(table_a) | set(table_b):
        row_a, row_b = table_a.get(state), table_b.get(state)
        if row_a is None:
            child[state] = row_b      # nur B kennt diesen Zustand
        elif row_b is None:
            child[state] = row_a      # nur A kennt diesen Zustand
        else:
            # Der Zufallswert wird gegen weight_a geprüft statt gegen 0.5
            child[state] = row_a if np.random.random() < weight_a else row_b

    return child


def mutate_table(table: QTable, rate: float, noise: float) -> QTable:
    """Addiert Gauß-Rauschen auf einen Bruchteil (`rate`) der Q-Werte (Pendant zu genetics.mutate).

    Copy-on-write: nur Zeilen, die wirklich Rauschen abbekommen, werden kopiert; alle
    anderen werden mit dem Elternteil geteilt (siehe Invariante oben). Bei kleinem `rate`
    bleibt so die große Mehrheit der Zeilen unkopiert.
    """
    child: QTable = {}
    for state, row in table.items():
        mutation_mask = np.random.rand(len(row)) < rate   # welche Q-Werte bekommen Rauschen?
        if mutation_mask.any():
            mutated_row = row.copy()
            mutated_row[mutation_mask] += np.random.normal(
                0, noise, int(mutation_mask.sum())).astype(np.float32)
            child[state] = mutated_row
        else:
            child[state] = row    # unverändert → Zeile mit dem Elternteil teilen
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
    from f1_rl.simulation.environment import create_car_env, reset_env, step_env

    rng = np.random.default_rng()
    table = _copy_table(inherited_table)        # die geerbte Tabelle nicht in place verändern

    env = create_car_env(track, use_rays=use_rays)
    obs = reset_env(env)
    state = bin_values_for_qtable(obs)

    # ── Lernphase (Folie 40 «FOR EACH Episode …») ────────────────────────────
    for _ in range(n_steps):
        action = epsilon_greedy_policy(table, state, epsilon, rng)
        obs, reward, terminated, truncated, _ = step_env(env, action)
        next_state = bin_values_for_qtable(obs)
        done = terminated or truncated
        temporal_difference_update(table, state, action, reward, next_state, done)
        state = next_state
        if done:
            obs = reset_env(env)
            state = bin_values_for_qtable(obs)

    # ── Greedy-Eval (eine Episode, Fitness = aufsummierter Reward) ───────────
    eval_reward = 0.0
    obs = reset_env(env)
    for _ in range(eval_steps):
        action = policy_action_readonly(table, obs)   # Eval lernt nicht → keine Phantom-Zustände
        obs, reward, terminated, truncated, _ = step_env(env, action)
        eval_reward += reward
        if terminated or truncated:
            break
    end_x, end_y = env["x_m"], env["y_m"]
    return eval_reward, table, end_x, end_y


# ── Anzeige-Helfer (die eigentliche Anzeige-Schleife lebt in display.py) ──────

def _table_as_policy(table: QTable) -> QTable:
    """build_policy-Callback fürs gemeinsame Display: eine Q-Tabelle IST schon
    die Policy (beim DQN wird hier stattdessen ein Netz aus Gewichten gebaut)."""
    return table


def _keep_table_reference(table: QTable) -> QTable:
    """copy_individual-Callback für die Hall of Fame: Tabellen werden per
    REFERENZ aufbewahrt statt kopiert — fertige Tabellen werden nie mehr
    verändert (Invariante oben), Kopieren wäre nur unnötige Arbeit."""
    return table


TABLE_MAX_ROWS = 400   # mehr Zeilen als Canvas-Pixel bringen visuell nichts → downsamplen


def _table_heatmap_payload(table: QTable, selected_car: int) -> dict:
    """Bereitet die volle Q-Tabelle des inspizierten Autos für die UI-Heatmap auf.

    Die Zeilen werden stabil sortiert und bei sehr vielen Zuständen gleichmäßig
    auf TABLE_MAX_ROWS heruntergerechnet — die Canvas zeigt ohnehin nur ~320
    Pixelzeilen, und eine zu große Nutzlast würde die Server-Event-Loop blockieren.
    """
    from f1_rl.server.protocol import qtable_to_dict

    n_states = len(table)
    keys = sorted(table.keys())          # stabile Zeilen-Reihenfolge über Updates
    if n_states > TABLE_MAX_ROWS:
        step_size = n_states / TABLE_MAX_ROWS
        keys = [keys[int(i * step_size)] for i in range(TABLE_MAX_ROWS)]
    return qtable_to_dict(selected_car, [table[k] for k in keys], n_states)


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


def _breed_table_generation(tables_out: list, fitnesses: list, hall_of_fame: list,
                            best_ever_table: QTable, stagnation_count: int,
                            stagnation_boost: float, n_pop: int) -> list:
    """Nächste Generation mit BIASED Crossover:
    HOF-Eliten, mutierte Best-Klone und gewichtete Crossover-Kinder.
    """
    # hall_of_fame hat das Format [(score, table), (score, table), ...]
    new_pop = [table for _, table in hall_of_fame]  # Tier 1: Eliten

    for _ in range(N_BEST_CLONES):  # Tier 2: Best-Klone
        if len(new_pop) < n_pop:
            new_pop.append(mutate_table(best_ever_table, MUTATION_RATE * 0.3, MUTATION_NOISE * 0.3))

    n_protected = len(new_pop)  # Tier 3: Crossover-Kinder

    # Wir kombinieren Tabellen und Scores, damit rank_select beides zurückgeben kann
    # Format: [(score_1, table_1), (score_2, table_2), ...]
    current_pop_scored = list(zip(fitnesses, tables_out))

    while len(new_pop) < n_pop:
        # Elternteil A auswählen (Score und Tabelle holen)
        if hall_of_fame and np.random.random() < 0.5:
            score_a, parent_a = rank_select(hall_of_fame)
        else:
            score_a, parent_a = rank_select(current_pop_scored)

        # Elternteil B auswählen (Score und Tabelle holen)
        score_b, parent_b = rank_select(current_pop_scored)

        # Biased Crossover anwenden (besseres Elternteil vererbt mehr)
        child = crossover_tables_biased(parent_a, parent_b, score_a, score_b)

        # Mutieren des Kindes (mit Rang-Faktor wie bisher)
        rank_factor = (len(new_pop) - n_protected) / max(n_pop - n_protected, 1)
        child = mutate_table(
            child,
            MUTATION_RATE * stagnation_boost * (0.3 + rank_factor * 0.7),
            MUTATION_NOISE * stagnation_boost * (0.3 + rank_factor * 0.7),
        )
        new_pop.append(child)

    # Stagnations-Behandlung: Frische Tabelle injizieren
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
    use_rays: bool = True,
    cancel_event=None,
    table_queue: Queue | None = None,  # Live-Stream der vollen Tabelle (nur Q-Table-Modus)
) -> tuple[QTable, object]:
    """Genetische, parallele Q-Tabellen-Evolution.

    QTABLE_N_POP Agenten, je in einem eigenen Prozess: jeder lernt seine Tabelle online
    (Q-Learning) und wird greedy bewertet. Danach selektiert/kreuzt/mutiert
    der GA die Tabellen pro Generation. Gleiche Umgebung, gleiches WebSocket-Protokoll und
    dieselbe Live-Population-Ansicht wie der DQN-Trainer — nur die Policy ist eine Q-Tabelle.
    """
    from f1_rl.learning.display import run_population_display
    from f1_rl.learning.evolution import (
        epsilon_for_generation, eval_step_budget, update_auto_speed,
        update_hall_of_fame, update_stagnation, update_top_scores,
    )
    from f1_rl.learning.replay import emit_racing_line, record_greedy_replay
    from f1_rl.simulation.track_loader import load_track

    os.makedirs(MODEL_DIR, exist_ok=True)
    if save_path is None:
        save_path = QTABLE_PATH
    if track is None:
        track = load_track(track_query, geojson_fallback_path=geojson_fallback)
    print(f"[qtable] Track: {track['name']}  {track['total_length_m']:.0f} m")

    if total_gens is None:
        total_gens = 200
    n_pop = QTABLE_N_POP

    # ── Population initialisieren / Resume ───────────────────────────────────
    population, start_gen, best_ever_fitness = _init_table_population(resume, save_path, n_pop)
    best_ever_table = population[0]
    hall_of_fame: list[tuple[float, QTable]] = []
    top_scores: list[tuple[float, int]] = []   # Scoreboard: (Score, Generation), best first
    stagnation_count = 0
    prev_best_eval = -1e9
    steps_so_far = 0
    last_line_t = 0.0   # drosselt das Racing-Line-Replay (pure-Python, ~Tausende Steps)

    pop_holder = [list(population)]    # Display-Thread liest pop_holder[0]
    gen_holder = [start_gen]
    stop_event = threading.Event()

    print(f"[qtable] Genetic tabular Q-learning  {n_pop} agents x {steps_per_gen} steps/gen "
          f"x {total_gens} gens  features={len(FEATURE_IDX)} bins={N_BINS} alpha={LEARNING_RATE} gamma={GAMMA}")

    if render_queue is not None:
        display_thread = threading.Thread(
            target=run_population_display,
            args=(pop_holder, stop_event, track, render_queue),
            kwargs=dict(
                build_policy=_table_as_policy,            # Tabelle ist schon die Policy
                choose_action=policy_action_readonly,     # Anzeige verändert die Tabelle nicht
                inspect_trace_fn=inspect_trace,
                use_rays=use_rays,
                speed_holder=speed_holder,
                gen_holder=gen_holder,
                inspect_holder=inspect_holder,
                inspect_queue=inspect_queue,
                table_queue=table_queue,
                make_table_payload=_table_heatmap_payload,
            ),
            daemon=True,
        )
        display_thread.start()
    else:
        display_thread = None

    n_workers = min(n_pop, os.cpu_count() or 4)
    eval_steps = eval_step_budget(track)
    print(f"[qtable] Spawning {n_workers} worker processes  eval={eval_steps} steps/episode")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for gen in range(start_gen, start_gen + total_gens):
            if cancel_event is not None and cancel_event.is_set():
                print(f"[qtable] Stop requested — ending at generation {gen}")
                break

            gen_start_time = time.perf_counter()
            # Stetiges Sinken des Epsilon-Wertes (erst viel erkunden, später greedy)
            epsilon = epsilon_for_generation(gen, start_gen, total_gens)

            # 1. Population auswerten (ein Worker-Prozess je Tabelle), best first.
            results = list(executor.map(
                run_qtable_worker,
                [(table, track, steps_per_gen, epsilon, eval_steps, use_rays)
                 for table in population],
                chunksize=1,
            ))
            fitnesses = [result[0] for result in results]
            tables_out = [result[1] for result in results]
            ghost_positions = [(result[2], result[3]) for result in results]   # Endpositionen
            best_first = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            fitnesses = [fitnesses[i] for i in best_first]
            tables_out = [tables_out[i] for i in best_first]
            ghost_positions = [ghost_positions[i] for i in best_first]

            # 2. Allzeit-Bestenliste (Hall of Fame) über Tabellen.
            # copy_individual = Identität: fertige Tabellen werden nie mehr verändert
            # (Invariante oben), Referenzen aufzubewahren ist also sicher und billig.
            update_hall_of_fame(hall_of_fame, fitnesses, tables_out,
                                copy_individual=_keep_table_reference)
            if hall_of_fame and hall_of_fame[0][0] > best_ever_fitness:
                best_ever_fitness = hall_of_fame[0][0]
                best_ever_table = hall_of_fame[0][1]   # Referenz (wird nur read-only genutzt)
                # Replay ist ein langer pure-Python-Lauf (hält das GIL). Bei den schnellen
                # Tabellen-Generationen kommen neue Bestwerte im Sekundentakt → höchstens
                # alle 2 s neu aufzeichnen, sonst hungert die Event-Loop (ruckelndes UI).
                now = time.perf_counter()
                if now - last_line_t > 2.0:
                    last_line_t = now
                    emit_racing_line(line_queue, record_greedy_replay(
                        best_ever_table, policy_action_readonly, track,
                        eval_steps, use_rays=use_rays))

            # 3. Sortierte Population + Generation an die Live-Anzeige übergeben.
            gen_holder[0] = gen
            pop_holder[0] = list(tables_out)

            # 4. Stagnation → Mutation hochregeln.
            prev_best_eval, stagnation_count, stagnation_boost = update_stagnation(
                prev_best_eval, stagnation_count, fitnesses[0])

            # 5. Nächste Generation züchten.
            population = _breed_table_generation(
                tables_out, fitnesses, hall_of_fame, best_ever_table,
                stagnation_count, stagnation_boost, n_pop)

            # 6. Stats + Checkpoint.
            update_top_scores(top_scores, fitnesses[0], gen)
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
                        "top_scores":   list(top_scores),
                    })
                except Full:
                    pass

            if (gen - start_gen + 1) % SAVE_EVERY == 0 or gen == start_gen + total_gens - 1:
                save_q_table(best_ever_table, save_path)
                _write_state(track["name"], steps_so_far, total_timesteps or steps_so_far,
                             epsilon, gen, best_ever_fitness, len(best_ever_table))
                stag_str = f"  stag={stagnation_count}×{stagnation_boost:.1f}" if stagnation_count > 0 else ""
                print(f"[gen {gen:>4d}/{start_gen + total_gens - 1}]  ε={epsilon:.3f}"
                      f"  eval={fitnesses[0]:+.1f}  mean={float(np.mean(fitnesses)):+.1f}"
                      f"  hof_best={best_ever_fitness:+.1f}  states_best={best_states}{stag_str}")

            # 7. Auto-Speed: Animation an die Generations-Rechenzeit koppeln.
            if auto_speed and speed_holder is not None:
                update_auto_speed(speed_holder, eval_steps,
                                  time.perf_counter() - gen_start_time)

    stop_event.set()
    if display_thread is not None:
        display_thread.join(timeout=2.0)

    save_q_table(best_ever_table, save_path)
    _write_state(track["name"], steps_so_far, total_timesteps or steps_so_far,
                 EPSILON_MIN, start_gen + total_gens - 1, best_ever_fitness, len(best_ever_table))
    print(f"[qtable] Done. best={best_ever_fitness:+.1f}  {len(best_ever_table)} states -> {save_path}")
    return best_ever_table, track


# ── Standalone-Runner (ohne Web-UI) ──────────────────────────────────────────────

if __name__ == "__main__":
    # Headless-Konsolentraining, z. B.:  python -m f1_rl.learning.qtable
    q_learning_loop(steps_per_gen=5_000, total_gens=200, resume=False)