"""Genetic DQN trainer: the generation loop that ties DQN + GA together.

Each generation:
  1. Run N_POP workers in parallel — every worker does Double-DQN training on
     its inherited weights, then plays one evaluation lap for a fitness score
     (see worker.run_worker).
  2. Sort by fitness, keep a hall-of-fame of elites, save replays of new bests.
  3. Breed the next generation with the genetic operators (classic rank-based
     selection, or pack/group selection).

This module also handles persistence (checkpoints, training state) and the
live display thread that animates the whole population for the UI.
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
    EPSILON_MIN, EPSILON_START, EXPLORE_FRAC,
    HALL_OF_FAME_K, MAX_REPLAYS, MIGRATION_RATE, MODEL_DIR, MODEL_PATH,
    MUTATION_NOISE, MUTATION_RATE,
    N_ACTIONS, N_BEST_CLONES, N_PACKS, N_POP, PACK_MIN_SURVIVORS, PACK_SUPPORT,
    REPLAY_DIR, SAVE_EVERY,
    STATE_PATH, STEPS_PER_GEN, STAGNATION_GENS,
)
from f1_rl.learning.genetics import (
    assign_packs, breed_packs, crossover, mutate, rank_select,
)


# ── Persistence ───────────────────────────────────────────────────────────────

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


# ── Replay recording ──────────────────────────────────────────────────────────

def _eval_step_budget(track) -> int:
    """Eval-episode length scaled to the track so a full lap can be completed.

    A fixed cap (e.g. 2000) is too short for most circuits, so the lap/goal bonus
    would never register in fitness.  Estimate steps for one lap at an assumed avg
    speed, add a margin, and clamp between EVAL_STEPS and EVAL_STEP_CAP.
    """
    from f1_rl.config import (
        EVAL_AVG_SPEED_MS, EVAL_STEP_BUFFER, EVAL_STEP_CAP, EVAL_STEPS, LAPS_PER_EPISODE,
    )
    # Budget for the whole multi-lap episode so all LAPS_PER_EPISODE laps fit.
    steps_for_lap = track.total_length_m / EVAL_AVG_SPEED_MS * 60.0 * EVAL_STEP_BUFFER
    return int(min(EVAL_STEP_CAP, max(EVAL_STEPS, steps_for_lap * LAPS_PER_EPISODE)))


def _record_replay(weights: np.ndarray, track, max_steps: int = 6000,
                   use_rays: bool = True) -> np.ndarray:
    from f1_rl.learning.agent import act, neuronal_net_from_weights
    from f1_rl.simulation.environment import F1Env
    net = neuronal_net_from_weights(weights)
    env = F1Env(track=track, render_mode=None, use_rays=use_rays)
    obs, _ = env.reset()
    frames: list[tuple] = []
    for _ in range(max_steps):
        a = act(net, obs)
        obs, _, terminated, _, info = env.step(a)
        frames.append((env.x_m, env.y_m, env.heading, env.speed_ms,
                       env._last_throttle, env.progress))
        # Record the full multi-lap episode (ends on the final lap or a crash).
        if terminated:
            break
    env.close()
    return np.array(frames, dtype=np.float32)


def _emit_racing_line(line_queue, frames: np.ndarray | None) -> None:
    """Push the best lap's path (x, y, speed) to the UI queue, newest only.

    Drops silently if there is no queue / no frames, so it is safe to call
    unconditionally from the training loop.
    """
    if line_queue is None or frames is None or len(frames) == 0:
        return
    # x, y, speed, throttle  (frame tuple is x,y,heading,speed,throttle,progress)
    line = [(float(f[0]), float(f[1]), float(f[3]), float(f[4])) for f in frames]
    try:
        while not line_queue.empty():
            line_queue.get_nowait()
    except Exception:                    # noqa: BLE001
        pass
    try:
        line_queue.put_nowait(line)
    except Full:
        pass


def _save_replay(weights: np.ndarray, track, fitness: float, gen: int,
                 max_steps: int = 6000, use_rays: bool = True,
                 frames: np.ndarray | None = None) -> None:
    import glob as _glob
    os.makedirs(REPLAY_DIR, exist_ok=True)
    if frames is None:
        frames = _record_replay(weights, track, max_steps=max_steps, use_rays=use_rays)
    path = os.path.join(REPLAY_DIR, f"replay_g{gen:04d}.npz")
    np.savez_compressed(
        path,
        frames=frames,
        fitness=np.array([fitness], dtype=np.float32),
        generation=np.array([gen], dtype=np.int32),
        circuit=np.array([track.name]),
    )
    all_paths = _glob.glob(os.path.join(REPLAY_DIR, "replay_*.npz"))
    if len(all_paths) > MAX_REPLAYS:
        scored: list[tuple[float, str]] = []
        for p in all_paths:
            try:
                d = np.load(p, allow_pickle=True)
                scored.append((float(d["fitness"][0]), p))
            except Exception:
                scored.append((-1e9, p))
        scored.sort(reverse=True)
        for _, old_path in scored[MAX_REPLAYS:]:
            try:
                os.remove(old_path)
            except OSError:
                pass
    print(f"[replay] Saved {len(frames)} frames  fitness={fitness:+.1f}  → {os.path.basename(path)}")


# ── Display thread ────────────────────────────────────────────────────────────

def _display_thread(pop_holder: list, stop_event: threading.Event,
                    track, render_queue: Queue, pack_holder: list,
                    use_rays: bool = True, speed_holder: list | None = None,
                    gen_holder: list | None = None,
                    inspect_holder: list | None = None,
                    inspect_queue: Queue | None = None) -> None:
    """Step the full population at ~60 fps, pushing a list of frames to render_queue.

    When a new generation arrives, each car is queued for a soft swap: it keeps
    running with its current weights until it crashes naturally, then picks up the
    new generation's weights on reset.  This lets ongoing runs finish before the
    transition, so there are no mass teleports and good runs are not cut short.
    """
    from f1_rl.learning.agent import act, forward_trace, neuronal_net_from_weights
    from f1_rl.server.protocol import inspect_to_dict
    from f1_rl.simulation.environment import CarFrame, F1Env, REWARD_PARTS

    current_pop = pop_holder[0]
    n_cars      = len(current_pop)
    envs        = [F1Env(track=track, render_mode=None, use_rays=use_rays) for _ in range(n_cars)]
    agents      = [neuronal_net_from_weights(w) for w in current_pop]
    obses       = [env.reset()[0] for env in envs]

    # next_agents[i]: agent waiting to replace agents[i] on next crash; None = no pending swap
    next_agents: list = [None] * n_cars

    if speed_holder is None:
        speed_holder = [1]
    if gen_holder is None:
        gen_holder = [0]
    # car_gens[i]: which generation the agent currently driving car i belongs to.
    # next_gen[i]: generation of the queued replacement (adopted on the next crash),
    # so the "colour by generation" view shows the true mix during soft swaps.
    car_gens: list[int] = [int(gen_holder[0])] * n_cars
    next_gen: list[int] = [int(gen_holder[0])] * n_cars
    dt = 1.0 / 60.0
    frame_no = 0
    INSPECT_EVERY = 6     # stream net/Q detail ~10×/s, not 60×/s (keeps the UI snappy)
    MAX_DISPLAY_LAG = 2   # never let the animation drift more than this many gens behind

    while not stop_event.is_set():
        t0 = time.perf_counter()
        frame_no += 1
        n_sub = max(1, int(speed_holder[0]))   # sim sub-steps this frame (live speed)

        new_pop = pop_holder[0]
        if new_pop is not current_pop:
            current_pop = new_pop
            # Build new agents and queue them — each car picks up its new agent
            # the next time it crashes, so current runs finish uninterrupted.
            new_agent_list = [neuronal_net_from_weights(w) for w in current_pop]
            arriving_gen = int(gen_holder[0])
            for i in range(n_cars):
                next_agents[i] = new_agent_list[i]
                next_gen[i] = arriving_gen

        packs = pack_holder[0]
        arriving = int(gen_holder[0])
        frames: list[tuple] = []
        for i in range(n_cars):
            # Bound the display lag: if this car is still running an old generation
            # while training has moved on, snap it to the latest NOW instead of
            # waiting for a natural crash. Without this the animation drifts tens of
            # generations behind (worse since multi-lap runs last much longer).
            if next_agents[i] is not None and arriving - car_gens[i] > MAX_DISPLAY_LAG:
                agents[i] = next_agents[i]
                next_agents[i] = None
                car_gens[i] = next_gen[i]
                obses[i] = envs[i].reset()[0]
            # Advance n_sub steps; the frame shows the final (possibly crashed) state.
            term = False
            for _ in range(n_sub):
                a = act(agents[i], obses[i])
                obses[i], _, term, _, _ = envs[i].step(a)
                if term:
                    break
            env = envs[i]
            frames.append(CarFrame(
                x=env.x_m, y=env.y_m, heading=env.heading,
                speed=env.speed_ms, throttle=env._last_throttle,
                checkpoint=env._checkpoint_idx, progress=env.progress,
                lap=env.lap_count,
                rays=tuple(env._cast_rays()),
                pack=int(packs[i]) if i < len(packs) else 0,  # swarm colouring
                score=env.episode_reward,                     # live cumulative score
                reward_parts=tuple(env.reward_parts[k] for k in REWARD_PARTS),
                generation=car_gens[i],
                a_long=env._a_long, a_lat=env._a_lat,
            ))
            if term:
                obses[i] = envs[i].reset()[0]
                if next_agents[i] is not None:
                    agents[i]    = next_agents[i]
                    next_agents[i] = None
                    car_gens[i]    = next_gen[i]

        while not render_queue.empty():
            try:
                render_queue.get_nowait()
            except Exception:
                break
        try:
            render_queue.put_nowait(frames)
        except Full:
            pass

        # Net/Q-value detail for the inspected car (one car only, throttled).
        if inspect_queue is not None and inspect_holder is not None and frame_no % INSPECT_EVERY == 0:
            sel = inspect_holder[0]
            if isinstance(sel, int) and 0 <= sel < n_cars:
                q, hidden = forward_trace(agents[sel], obses[sel])
                try:
                    while not inspect_queue.empty():
                        inspect_queue.get_nowait()
                    inspect_queue.put_nowait(
                        inspect_to_dict(sel, obses[sel], q, hidden, int(q.argmax())))
                except Exception:            # noqa: BLE001
                    pass

        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

    for env in envs:
        env.close()


# ── Generation building blocks ──────────────────────────────────────────────────

def _init_population(resume: bool, save_path: str, amount_of_weights: int) -> tuple[list, int, float]:
    """Build the starting population.

    Returns (population, start_gen, best_ever_fitness). With ``resume`` and a
    compatible checkpoint, the population is seeded from the saved best (plus
    mutated copies) and the saved generation/fitness are restored; otherwise a
    fresh random population starts from generation 0.
    """
    from f1_rl.learning.network import random_weights

    if resume and os.path.exists(save_path):
        loaded = np.load(save_path)
        if loaded.ndim == 1 and len(loaded) == amount_of_weights:
            best_w = loaded.astype(np.float32)
            saved  = load_training_state() or {}
            start_gen = int(saved.get("generation", 0))
            best_ever_fitness = float(saved.get("best_fitness", -1e9))
            population = [best_w.copy()] + [
                mutate(best_w, MUTATION_RATE * 3, MUTATION_NOISE * 2)
                for _ in range(N_POP - 1)
            ]
            print(f"[train] Resumed gen={start_gen}  best_fitness={best_ever_fitness:.1f}")
            return population, start_gen, best_ever_fitness
        print(f"[train] Incompatible checkpoint (shape {loaded.shape}, "
              f"expected ({amount_of_weights},)) — starting fresh")

    return [random_weights() for _ in range(N_POP)], 0, -1e9


def _evaluate_population(executor, population, track, steps_per_gen, epsilon,
                        eval_steps, use_rays) -> tuple[list, list, list]:
    """Run one DDQN worker per individual, then return results sorted best-first.

    Returns (fitnesses, weights_out, ghost_positions); each list is aligned by
    rank, so index 0 is the highest-fitness individual.
    """
    from f1_rl.learning.worker import run_worker

    results = list(executor.map(
        run_worker,
        [(w.copy(), track, steps_per_gen, epsilon, eval_steps, use_rays) for w in population],
        chunksize=1,
    ))
    fitnesses       = [r[0] for r in results]
    weights_out     = [r[1] for r in results]
    ghost_positions = [(r[2], r[3]) for r in results]

    order = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    return (
        [fitnesses[i]       for i in order],
        [weights_out[i]     for i in order],
        [ghost_positions[i] for i in order],
    )


def _update_hall_of_fame(hall_of_fame: list, fitnesses: list, weights_out: list) -> None:
    """Merge this generation's results into ``hall_of_fame`` (edited in place).

    Keeps the best HALL_OF_FAME_K weight vectors of all time, best first. Inputs
    are sorted best-first, so we can stop early once an individual can no longer
    displace the current worst elite.
    """
    for fit, w in zip(fitnesses, weights_out):
        if len(hall_of_fame) < HALL_OF_FAME_K:
            hall_of_fame.append((fit, w.copy()))
            hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        elif fit > hall_of_fame[-1][0]:
            hall_of_fame[-1] = (fit, w.copy())
            hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        else:
            break


def _breed_next_generation(evolution_mode, weights_out, fitnesses, pack_ids,
                          hall_of_fame, best_ever_w, stagnation_count,
                          stagnation_boost) -> list:
    """Produce the next generation's population from this generation's results.

    "classic" mode: HOF elites survive unchanged, light-mutation clones of the
    all-time best compound it, and the rest are crossover children with mutation
    tapered from near-zero (elite-adjacent) to full (the tail). "pack" mode
    delegates to genetics.breed_packs (group selection). When badly stuck, one
    slot is replaced by a fresh random individual.
    """
    from f1_rl.learning.network import random_weights

    hof_weights = [w for _, w in hall_of_fame]

    if evolution_mode == "pack":
        new_pop = breed_packs(
            weights_out, fitnesses, pack_ids, hof_weights, best_ever_w, N_POP,
            pack_support  = PACK_SUPPORT,
            migration_rate= MIGRATION_RATE,
            min_survivors = PACK_MIN_SURVIVORS,
            mutation_rate = MUTATION_RATE  * stagnation_boost,
            mutation_noise= MUTATION_NOISE * stagnation_boost,
        )
    else:
        # Tier 1: HOF elites survive unchanged
        new_pop = [w.copy() for _, w in hall_of_fame]

        # Tier 2: light-mutation copies of the all-time best
        for _ in range(N_BEST_CLONES):
            if len(new_pop) < N_POP:
                new_pop.append(mutate(best_ever_w,
                                      MUTATION_RATE * 0.3,
                                      MUTATION_NOISE * 0.3))

        # Tier 3: crossover children with rank-tapered mutation
        n_protected = len(new_pop)
        while len(new_pop) < N_POP:
            pa = (rank_select(hof_weights)
                  if (hof_weights and np.random.random() < 0.5)
                  else rank_select(weights_out))
            pb = rank_select(weights_out)
            child = crossover(pa, pb)
            rank_factor = (len(new_pop) - n_protected) / max(N_POP - n_protected, 1)
            child = mutate(
                child,
                rate  = MUTATION_RATE  * stagnation_boost * (0.3 + rank_factor * 0.7),
                noise = MUTATION_NOISE * stagnation_boost * (0.3 + rank_factor * 0.7),
            )
            new_pop.append(child)

    # Inject a fresh random individual when badly stuck
    if stagnation_count > 0 and stagnation_count % STAGNATION_GENS == 0:
        new_pop[-1] = random_weights()

    return new_pop


# ── Main training loop ────────────────────────────────────────────────────────

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
    evolution_mode: str = "classic",
    use_rays: bool = True,
    cancel_event=None,
    table_queue: Queue | None = None,  # accepted & ignored (q-table backend only)
) -> tuple[object, object]:
    """Genetic DQN: N_POP parallel DDQN workers evolved by the GA each generation.

    evolution_mode: "classic" = individual rank-based selection;
                    "pack"    = pack/group selection (weak DNA survives via its pack).
    cancel_event: optional threading.Event — when set, training stops cleanly at
                  the next generation boundary (a running generation finishes first).
    """
    # Imports that pull in torch are deferred to here so the menu/UI starts fast.
    from f1_rl.learning.agent import neuronal_net_from_weights
    from f1_rl.learning.network import n_params
    from f1_rl.simulation.track_loader import load_track

    os.makedirs(MODEL_DIR, exist_ok=True)
    if save_path is None:
        save_path = MODEL_PATH

    if track is None:
        print(f"[train] Loading track: {track_query}")
        track = load_track(track_query, geojson_fallback_path=geojson_fallback)
    print(f"[train] Track: {track.name}  {track.total_length_m:.0f} m")

    if total_gens is None:
        total_gens = max(500, total_timesteps // (N_POP * steps_per_gen))

    n_w = n_params()
    print(
        f"[train] Genetic DQN  {N_POP} workers x {steps_per_gen} steps/gen x {total_gens} gens"
        f"  ~ {N_POP * steps_per_gen * total_gens:,} total env-steps"
        f"  network params={n_w:,}  actions={N_ACTIONS}"
    )

    # ── Initialise or resume population ──────────────────────────────────────
    population, start_gen, best_ever_fitness = _init_population(resume, save_path, n_w)
    best_ever_weight_vector: np.ndarray = population[0].copy()

    hall_of_fame: list[tuple[float, np.ndarray]] = []
    top_10: list[tuple[float, int]] = []
    stagnation_count  = 0
    prev_best_eval    = -1e9

    pop_holder  = [list(population)]        # display thread reads pop_holder[0]
    pack_holder = [[0] * len(population)]    # parallel pack ids for swarm colouring
    gen_holder  = [start_gen]                # current generation, for per-car colouring
    stop_event  = threading.Event()
    print(f"[train] Evolution mode: {evolution_mode}")

    if render_queue is not None:
        disp = threading.Thread(
            target=_display_thread,
            args=(pop_holder, stop_event, track, render_queue, pack_holder, use_rays,
                  speed_holder, gen_holder, inspect_holder, inspect_queue),
            daemon=True,
        )
        disp.start()
    else:
        disp = None

    n_workers = min(N_POP, os.cpu_count() or 4)
    print(f"[train] Spawning {n_workers} worker processes")

    eval_steps = _eval_step_budget(track)
    print(f"[train] Eval budget: {eval_steps} steps/episode "
          f"(≥1 lap of {track.total_length_m:.0f} m)")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for gen in range(start_gen, start_gen + total_gens):

            if cancel_event is not None and cancel_event.is_set():
                print(f"[train] Stop requested — ending at generation {gen}")
                break

            gen_t0 = time.perf_counter()
            frac = (gen - start_gen) / total_gens
            epsilon = (
                EPSILON_START - (EPSILON_START - EPSILON_MIN) * frac / EXPLORE_FRAC
                if frac < EXPLORE_FRAC else EPSILON_MIN
            )

            # 1. Evaluate the whole population (one DDQN worker each), best first.
            fitnesses, weights_out, ghost_positions = _evaluate_population(
                executor, population, track, steps_per_gen, epsilon, eval_steps, use_rays)

            # 2. Track the all-time best via the hall of fame.
            _update_hall_of_fame(hall_of_fame, fitnesses, weights_out)
            if hall_of_fame and hall_of_fame[0][0] > best_ever_fitness:
                best_ever_fitness = hall_of_fame[0][0]
                best_ever_weight_vector = hall_of_fame[0][1].copy()
                # Only a NEW all-time best refreshes the racing line, recorded
                # from the all-time-best weights. This keeps the displayed line
                # monotonically improving instead of flickering / regressing with
                # every top-10 entry of a merely-good generation.
                _emit_racing_line(line_queue, _record_replay(
                    best_ever_weight_vector, track,
                    max_steps=eval_steps, use_rays=use_rays))

            # 3. Pack assignment (swarm mode): cluster cars by where they ended
            #    up + their score, forming "packs" of similarly-driving cars.
            pack_ids = [0] * N_POP
            if evolution_mode == "pack":
                descriptors = np.array(
                    [[gx, gy, fit] for (gx, gy), fit in zip(ghost_positions, fitnesses)],
                    dtype=np.float64,
                )
                pack_ids = assign_packs(descriptors, N_PACKS).tolist()

            # Hand the sorted population + pack colours + generation to the display.
            gen_holder[0]  = gen
            pop_holder[0]  = list(weights_out)
            pack_holder[0] = list(pack_ids)

            # 4. Leaderboard + replay: a generation that beats the current 10th
            #    best enters the top-10 and gets its replay saved, so every
            #    clickable top score has a matching replay to watch.
            if len(top_10) < 10 or fitnesses[0] > top_10[-1][0]:
                top_10.append((fitnesses[0], gen))
                top_10.sort(key=lambda x: x[0], reverse=True)
                if len(top_10) > 10:
                    top_10.pop()
                _save_replay(weights_out[0], track, fitnesses[0], gen,
                             max_steps=eval_steps, use_rays=use_rays)

            # 5. Stagnation: the longer the best score is stuck, the more we
            #    ramp mutation up (stagnation_boost) to escape the plateau.
            if fitnesses[0] > prev_best_eval + 0.5:
                prev_best_eval = fitnesses[0]
                stagnation_count = 0
            else:
                stagnation_count += 1
            stagnation_boost = 1.0 + min(2.0, stagnation_count / STAGNATION_GENS)

            # 6. Breed the next generation from this one's results.
            population = _breed_next_generation(
                evolution_mode, weights_out, fitnesses, pack_ids,
                hall_of_fame, best_ever_weight_vector, stagnation_count, stagnation_boost)

            # 7. Report stats + checkpoint ─────────────────────────────────
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
                        "top_scores":      list(top_10),
                        "n_packs":         len(set(pack_ids)) if evolution_mode == "pack" else 0,
                    })
                except Full:
                    pass

            # ── Checkpoint save ───────────────────────────────────────────
            if (gen - start_gen + 1) % SAVE_EVERY == 0 or gen == start_gen + total_gens - 1:
                np.save(save_path, best_ever_weight_vector)
                _write_state(track.name, steps_so_far, total_timesteps,
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

            # ── Auto display speed ────────────────────────────────────────
            # Make the live animation keep pace with training: in the time one
            # generation took to compute, the display should advance ~one full
            # episode, so this generation's cars finish before the next arrives.
            #   steps/sec on display = 60 × sub_steps  →  sub_steps = eval_steps/(60·T)
            if auto_speed and speed_holder is not None:
                from f1_rl.config import SIM_SPEED_AUTO_CAP
                gen_seconds = time.perf_counter() - gen_t0
                if gen_seconds > 0:
                    target = math.ceil(eval_steps / (60.0 * gen_seconds))
                    speed_holder[0] = max(1, min(SIM_SPEED_AUTO_CAP, target))

    stop_event.set()
    if disp is not None:
        disp.join(timeout=2.0)

    np.save(save_path, best_ever_weight_vector)
    _write_state(track.name, total_timesteps, total_timesteps,
                 EPSILON_MIN, start_gen + total_gens - 1, best_ever_fitness)
    print(f"[train] Done. Best weights → {save_path}")
    return neuronal_net_from_weights(best_ever_weight_vector), track


if __name__ == "__main__":
    train()
