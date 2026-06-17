import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from queue import Full, Queue

import numpy as np

from f1_rl.config import (
    EPSILON_MIN, MODEL_DIR, MODEL_PATH,
    MUTATION_NOISE, MUTATION_RATE,
    N_ACTIONS, N_BEST_CLONES, N_POP,
    SAVE_EVERY,
    STATE_PATH, STEPS_PER_GEN, STAGNATION_GENS,
)
from f1_rl.learning.evolution import (
    epsilon_for_generation, eval_step_budget, update_auto_speed,
    update_hall_of_fame, update_stagnation, update_top_scores,
)
from f1_rl.learning.genetics import crossover, mutate, rank_select
from f1_rl.learning.replay import emit_racing_line


def load_training_state() -> dict | None:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_state(circuit: str, steps: int, total: int, epsilon: float,
                 gen: int = 0, best_fitness: float = 0.0) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timesteps_done": steps,
                "total_timesteps": total,
                "circuit": circuit,
                "epsilon": epsilon,
                "generation": gen,
                "best_fitness": best_fitness,
            },
            f, indent=2,
        )


def _record_replay(weights: np.ndarray, track, max_steps: int = 6000,
                   use_rays: bool = True) -> np.ndarray:
    from f1_rl.learning.agent import act, neural_net_from_weights
    from f1_rl.learning.replay import record_greedy_replay
    net = neural_net_from_weights(weights)
    return record_greedy_replay(net, act, track, max_steps, use_rays=use_rays)


def _init_population(resume: bool, save_path: str, amount_of_weights: int) -> tuple[list, int, float]:
    from f1_rl.learning.network import random_weights

    if resume and os.path.exists(save_path):
        loaded = np.load(save_path)
        # Abgespeicherte Modell muss gleiche gewichtsanzahl haben wie gebraucht
        if loaded.ndim == 1 and len(loaded) == amount_of_weights:
            best_w = loaded.astype(np.float32)
            saved = load_training_state() or {}
            start_gen = int(saved.get("generation", 0))
            best_ever_fitness = float(saved.get("best_fitness", -1e9))
            population = [best_w.copy()] + [
                mutate(best_w, MUTATION_RATE * 3, MUTATION_NOISE * 2)
                for _ in range(N_POP - 1)
            ]
            print(
                f"[train] Resumed gen={start_gen}  best_fitness={best_ever_fitness:.1f}")
            return population, start_gen, best_ever_fitness
        print(f"[train] Incompatible checkpoint (shape {loaded.shape}, "
              f"expected ({amount_of_weights},)) — starting fresh")

    return [random_weights() for _ in range(N_POP)], 0, -1e9


def _evaluate_population(executor, population, track, steps_per_gen, epsilon,
                         eval_steps, use_rays, multi_start=False) -> tuple[list, list, list]:
    from f1_rl.learning.worker import run_worker

    results = list(executor.map(
        run_worker,
        [(w.copy(), track, steps_per_gen, epsilon, eval_steps, use_rays, multi_start)
         for w in population],
        chunksize=1,
    ))
    fitnesses = [r[0] for r in results]
    weights_out = [r[1] for r in results]
    ghost_positions = [(r[2], r[3]) for r in results]

    order = sorted(range(len(fitnesses)),
                   key=lambda i: fitnesses[i], reverse=True)
    return (
        [fitnesses[i] for i in order],
        [weights_out[i] for i in order],
        [ghost_positions[i] for i in order],
    )


def _copy_weight_vector(weights: np.ndarray) -> np.ndarray:
    return weights.copy()


def _breed_next_generation(weights_out,
                           hall_of_fame, best_ever_w, stagnation_count,
                           stagnation_boost, use_crossover=True) -> list:
    from f1_rl.learning.network import random_weights

    hof_weights = [w for _, w in hall_of_fame]

    # Stufe 1: HOF-Eliten überleben unverändert
    new_pop = [w.copy() for _, w in hall_of_fame]

    # Stufe 2: leicht mutierte Kopien des Allzeit-Besten
    for _ in range(N_BEST_CLONES):
        if len(new_pop) < N_POP:
            new_pop.append(mutate(best_ever_w,
                                  MUTATION_RATE * 0.3,
                                  MUTATION_NOISE * 0.3))

    # Stufe 3: Crossover-Kinder mit rang-abgestufter Mutation
    n_protected = len(new_pop)
    while len(new_pop) < N_POP:
        pa = (rank_select(hof_weights)
              if (hof_weights and np.random.random() < 0.5)
              else rank_select(weights_out))
        pb = rank_select(weights_out)
        # Crossover im Gewichtsraum ist bei neuronalen Netzen umstritten (competing
        # conventions). Schalter aus -> reine Mutation eines selektierten Elternteils
        # (Evolutionsstrategie-Stil) statt zwei Eltern zu mischen.
        child = crossover(pa, pb) if use_crossover else pa.copy()
        rank_factor = (len(new_pop) - n_protected) / \
            max(N_POP - n_protected, 1)
        child = mutate(
            child,
            rate=MUTATION_RATE * stagnation_boost * (0.3 + rank_factor * 0.7),
            noise=MUTATION_NOISE * stagnation_boost *
            (0.3 + rank_factor * 0.7),
        )
        new_pop.append(child)

    # Bei starker Stagnation ein frisches Zufalls-Individuum injizieren
    if stagnation_count > 0 and stagnation_count % STAGNATION_GENS == 0:
        new_pop[-1] = random_weights()

    return new_pop


# ── Haupt-Trainingsschleife ─────────────────────────────────────────────────────

def train(
    track_query: str = "Circuit de Monaco",
    geojson_fallback: str | None = None,
    total_timesteps: int = 10_000_000_000,
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
    multi_start_eval: bool = False,
    use_crossover: bool = True,
    # accepted & ignored (q-table backend only)
    table_queue: Queue | None = None,
) -> tuple[object, object]:
    # torch-lastige Imports erst hier, damit Menü/UI schnell starten.
    from f1_rl.learning.agent import neural_net_from_weights
    from f1_rl.learning.network import n_params
    from f1_rl.simulation.track_loader import load_track

    os.makedirs(MODEL_DIR, exist_ok=True)
    if save_path is None:
        save_path = MODEL_PATH

    if track is None:
        print(f"[train] Loading track: {track_query}")
        track = load_track(track_query, geojson_fallback_path=geojson_fallback)
    print(f"[train] Track: {track['name']}  {track['total_length_m']:.0f} m")

    if total_gens is None:
        total_gens = max(500, total_timesteps // (N_POP * steps_per_gen))

    network_params_amount = n_params()
    print(
        f"[train] Genetic DQN  {N_POP} workers x {steps_per_gen} steps/gen x {total_gens} gens"
        f"  ~ {N_POP * steps_per_gen * total_gens:,} total env-steps"
        f"  network params={network_params_amount:,}  actions={N_ACTIONS}"
    )
    print(f"[train] multi_start_eval={multi_start_eval}  use_crossover={use_crossover}")

    # Population initialisieren/fortsetzen
    population, start_gen, best_ever_fitness = _init_population(
        resume, save_path, network_params_amount)
    best_ever_weight_vector: np.ndarray = population[0].copy()

    hall_of_fame: list[tuple[float, np.ndarray]] = []
    # Scoreboard: (Score, Generation), best first
    top_scores: list[tuple[float, int]] = []
    stagnation_count = 0
    prev_best_eval = -1e9

    pop_holder = [list(population)]        # Display-Thread liest pop_holder[0]
    # aktuelle Generation, für die Auto-Färbung
    gen_holder = [start_gen]
    stop_event = threading.Event()

    if render_queue is not None:
        from f1_rl.learning.agent import act, forward_trace
        from f1_rl.learning.display import run_population_display
        display_thread = threading.Thread(
            target=run_population_display,
            args=(pop_holder, stop_event, track, render_queue),
            kwargs=dict(
                build_policy=neural_net_from_weights,   # Gewichtsvektor -> fertiges Netz
                choose_action=act,
                inspect_trace_fn=forward_trace,
                use_rays=use_rays,
                speed_holder=speed_holder,
                gen_holder=gen_holder,
                inspect_holder=inspect_holder,
                inspect_queue=inspect_queue,
            ),
            daemon=True,
        )
        display_thread.start()
    else:
        display_thread = None

    n_workers = min(N_POP, os.cpu_count() or 4)
    print(f"[train] Spawning {n_workers} worker processes")

    eval_steps = eval_step_budget(track)
    print(f"[train] Eval budget: {eval_steps} steps/episode "
          f"(≥1 lap of {track['total_length_m']:.0f} m)")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for gen in range(start_gen, start_gen + total_gens):

            if cancel_event is not None and cancel_event.is_set():
                print(f"[train] Stop requested — ending at generation {gen}")
                break

            gen_start_time = time.perf_counter()
            epsilon = epsilon_for_generation(gen, start_gen, total_gens)

            # 1. Ganze Population auswerten (je ein DDQN-Worker), best first.
            fitnesses, weights_out, ghost_positions = _evaluate_population(
                executor, population, track, steps_per_gen, epsilon, eval_steps,
                use_rays, multi_start_eval)

            # 2. Allzeit-Besten über die Hall of Fame verfolgen.
            update_hall_of_fame(hall_of_fame, fitnesses, weights_out,
                                copy_individual=_copy_weight_vector)
            if hall_of_fame and hall_of_fame[0][0] > best_ever_fitness:
                best_ever_fitness = hall_of_fame[0][0]
                best_ever_weight_vector = hall_of_fame[0][1].copy()
                # Nur ein NEUER Allzeit-Bestwert frischt die Racing-Line auf,
                # aufgezeichnet aus den Allzeit-Best-Gewichten. So verbessert sich
                # die angezeigte Linie monoton, statt bei jedem guten Lauf zu flackern.
                emit_racing_line(line_queue, _record_replay(
                    best_ever_weight_vector, track,
                    max_steps=eval_steps, use_rays=use_rays))

            # 3. Sortierte Population + Generation an die Anzeige geben.
            gen_holder[0] = gen
            pop_holder[0] = list(weights_out)

            # 4. Stagnation: je länger der Bestwert feststeckt, desto stärker wird
            #    die Mutation hochgeregelt (stagnation_boost), um das Plateau zu verlassen.
            prev_best_eval, stagnation_count, stagnation_boost = update_stagnation(
                prev_best_eval, stagnation_count, fitnesses[0])

            # 5. Nächste Generation aus den Ergebnissen dieser züchten.
            population = _breed_next_generation(
                weights_out,
                hall_of_fame, best_ever_weight_vector, stagnation_count, stagnation_boost,
                use_crossover=use_crossover)

            # 6. Stats melden + Checkpoint ─────────────────────────────────
            update_top_scores(top_scores, fitnesses[0], gen)
            steps_so_far = (gen - start_gen + 1) * N_POP * steps_per_gen
            if stats_queue is not None:
                try:
                    stats_queue.put_nowait({
                        "timesteps":       steps_so_far,
                        "epsilon":         epsilon,
                        "episode":         gen,
                        "generation":      gen,
                        "best_fitness":    fitnesses[0],
                        "mean_fitness":    float(np.mean(fitnesses)),
                        "ghost_positions": ghost_positions,
                        "top_scores":      list(top_scores),
                    })
                except Full:
                    pass

            # ── Checkpoint speichern ──────────────────────────────────────
            if (gen - start_gen + 1) % SAVE_EVERY == 0 or gen == start_gen + total_gens - 1:
                np.save(save_path, best_ever_weight_vector)
                _write_state(track["name"], steps_so_far, total_timesteps,
                             epsilon, gen, best_ever_fitness)
                stag_str = (
                    f"  stag={stagnation_count}×{stagnation_boost:.1f}"
                    if stagnation_count > 0 else ""
                )
                print(
                    f"[gen {gen:>4d}/{start_gen + total_gens - 1}]"
                    f"  ε={epsilon:.3f}"
                    f"  eval={fitnesses[0]:+.1f}"
                    f"  mean={float(np.mean(fitnesses)):+.1f}"
                    f"  hof_best={best_ever_fitness:+.1f}"
                    f"{stag_str}"
                )

            # ── Automatische Anzeige-Geschwindigkeit ───────────────────────
            if auto_speed and speed_holder is not None:
                update_auto_speed(speed_holder, eval_steps,
                                  time.perf_counter() - gen_start_time)

    stop_event.set()
    if display_thread is not None:
        display_thread.join(timeout=2.0)

    np.save(save_path, best_ever_weight_vector)
    _write_state(track["name"], total_timesteps, total_timesteps,
                 EPSILON_MIN, start_gen + total_gens - 1, best_ever_fitness)
    print(f"[train] Done. Best weights → {save_path}")
    return neural_net_from_weights(best_ever_weight_vector), track


if __name__ == "__main__":
    train()
