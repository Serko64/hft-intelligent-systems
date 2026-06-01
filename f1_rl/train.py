"""Genetic DQN trainer: population loop, display thread, replay I/O."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from queue import Full, Queue

import numpy as np

from .agent import NeuralAgent, n_params, random_weights
from .config import (
    EPSILON_MIN, EPSILON_START, EXPLORE_FRAC,
    HALL_OF_FAME_K, MAX_REPLAYS, MIGRATION_RATE, MODEL_DIR, MODEL_PATH,
    MUTATION_NOISE, MUTATION_RATE,
    N_ACTIONS, N_BEST_CLONES, N_PACKS, N_POP, PACK_MIN_SURVIVORS, PACK_SUPPORT,
    QTABLE_PATH, REPLAY_DIR, SAVE_EVERY,
    STATE_PATH, STEPS_PER_GEN, STAGNATION_GENS,
)
from .genetics import (
    _worker, assign_packs, breed_packs, crossover, mutate, rank_select,
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
    from .config import EVAL_AVG_SPEED_MS, EVAL_STEP_BUFFER, EVAL_STEP_CAP, EVAL_STEPS
    steps_for_lap = track.total_length_m / EVAL_AVG_SPEED_MS * 60.0 * EVAL_STEP_BUFFER
    return int(min(EVAL_STEP_CAP, max(EVAL_STEPS, steps_for_lap)))


def _record_replay(weights: np.ndarray, track, max_steps: int = 6000) -> np.ndarray:
    from f1_rl.env import F1Env
    from f1_rl.agent import NeuralAgent
    agent = NeuralAgent(weights)
    env   = F1Env(track=track, render_mode=None)
    obs, _ = env.reset()
    frames: list[tuple] = []
    for _ in range(max_steps):
        a, _ = agent.predict(obs)
        obs, _, terminated, _, info = env.step(a)
        frames.append((env.x_m, env.y_m, env.heading, env.speed_ms,
                       env._last_throttle, env.progress))
        if info.get("lap_complete") or terminated:
            break
    env.close()
    return np.array(frames, dtype=np.float32)


def _save_replay(weights: np.ndarray, track, fitness: float, gen: int,
                 max_steps: int = 6000) -> None:
    import glob as _glob
    os.makedirs(REPLAY_DIR, exist_ok=True)
    frames = _record_replay(weights, track, max_steps=max_steps)
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
                    track, render_queue: Queue, pack_holder: list) -> None:
    """Step the full population at ~60 fps, pushing a list of frames to render_queue.

    When a new generation arrives, each car is queued for a soft swap: it keeps
    running with its current weights until it crashes naturally, then picks up the
    new generation's weights on reset.  This lets ongoing runs finish before the
    transition, so there are no mass teleports and good runs are not cut short.
    """
    from f1_rl.env import F1Env, REWARD_PARTS
    from f1_rl.agent import NeuralAgent

    current_pop = pop_holder[0]
    n_cars      = len(current_pop)
    envs        = [F1Env(track=track, render_mode=None) for _ in range(n_cars)]
    agents      = [NeuralAgent(w) for w in current_pop]
    obses       = [env.reset()[0] for env in envs]

    # next_agents[i]: agent waiting to replace agents[i] on next crash; None = no pending swap
    next_agents: list = [None] * n_cars

    dt = 1.0 / 60.0

    while not stop_event.is_set():
        t0 = time.perf_counter()

        new_pop = pop_holder[0]
        if new_pop is not current_pop:
            current_pop = new_pop
            # Build new agents and queue them — each car picks up its new agent
            # the next time it crashes, so current runs finish uninterrupted.
            new_agent_list = [NeuralAgent(w) for w in current_pop]
            for i in range(n_cars):
                next_agents[i] = new_agent_list[i]

        packs = pack_holder[0]
        frames: list[tuple] = []
        for i in range(n_cars):
            a, _ = agents[i].predict(obses[i])
            obses[i], _, term, _, _ = envs[i].step(a)
            frames.append((
                envs[i].x_m, envs[i].y_m, envs[i].heading,
                envs[i].speed_ms, envs[i]._last_throttle,
                envs[i]._checkpoint_idx, envs[i].progress,
                envs[i].lap_count,
                tuple(envs[i]._cast_rays()),
                int(packs[i]) if i < len(packs) else 0,   # pack id (for swarm colouring)
                envs[i].episode_reward,                    # [10] live cumulative score
                tuple(envs[i].reward_parts[k] for k in REWARD_PARTS),  # [11] reward breakdown
            ))
            if term:
                obses[i] = envs[i].reset()[0]
                if next_agents[i] is not None:
                    agents[i]    = next_agents[i]
                    next_agents[i] = None

        while not render_queue.empty():
            try:
                render_queue.get_nowait()
            except Exception:
                break
        try:
            render_queue.put_nowait(frames)
        except Full:
            pass

        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

    for env in envs:
        env.close()


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    track_query: str = "Circuit de Monaco",
    geojson_fallback: str | None = None,
    total_timesteps: int = 10_000_000_000,
    save_path: str | None = None,
    render_queue: Queue | None = None,
    stats_queue: Queue | None = None,
    track=None,
    resume: bool = False,
    steps_per_gen: int = STEPS_PER_GEN,
    total_gens: int | None = None,
    evolution_mode: str = "classic",
) -> tuple[NeuralAgent, object]:
    """Genetic DQN: N_POP parallel DDQN workers evolved by the GA each generation.

    evolution_mode: "classic" = individual rank-based selection;
                    "pack"    = pack/group selection (weak DNA survives via its pack).
    """
    from f1_rl.track import load_track

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
        f"[train] Genetic DQN  {N_POP} workers × {steps_per_gen} steps/gen × {total_gens} gens"
        f"  ≈ {N_POP * steps_per_gen * total_gens:,} total env-steps"
        f"  network params={n_w:,}  actions={N_ACTIONS}"
    )

    # ── Initialise or resume population ──────────────────────────────────────
    start_gen         = 0
    best_ever_fitness = -1e9

    if resume and os.path.exists(save_path):
        loaded = np.load(save_path)
        if loaded.ndim == 1 and len(loaded) == n_w:
            best_w = loaded.astype(np.float32)
            saved  = load_training_state() or {}
            start_gen = int(saved.get("generation", 0))
            best_ever_fitness = float(saved.get("best_fitness", -1e9))
            population = [best_w.copy()] + [
                mutate(best_w, MUTATION_RATE * 3, MUTATION_NOISE * 2)
                for _ in range(N_POP - 1)
            ]
            print(f"[train] Resumed gen={start_gen}  best_fitness={best_ever_fitness:.1f}")
        else:
            print(f"[train] Incompatible checkpoint (shape {loaded.shape}, "
                  f"expected ({n_w},)) — starting fresh")
            population = [random_weights() for _ in range(N_POP)]
    else:
        population = [random_weights() for _ in range(N_POP)]

    best_ever_w: np.ndarray = population[0].copy()

    hall_of_fame: list[tuple[float, np.ndarray]] = []
    top_10: list[tuple[float, int]] = []
    stagnation_count  = 0
    prev_best_eval    = -1e9

    pop_holder  = [list(population)]        # display thread reads pop_holder[0]
    pack_holder = [[0] * len(population)]    # parallel pack ids for swarm colouring
    stop_event  = threading.Event()
    print(f"[train] Evolution mode: {evolution_mode}")

    if render_queue is not None:
        disp = threading.Thread(
            target=_display_thread,
            args=(pop_holder, stop_event, track, render_queue, pack_holder),
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

            frac = (gen - start_gen) / total_gens
            epsilon = (
                EPSILON_START - (EPSILON_START - EPSILON_MIN) * frac / EXPLORE_FRAC
                if frac < EXPLORE_FRAC else EPSILON_MIN
            )

            results = list(executor.map(
                _worker,
                [(w.copy(), track, steps_per_gen, epsilon, eval_steps) for w in population],
                chunksize=1,
            ))

            fitnesses       = [r[0] for r in results]
            weights_out     = [r[1] for r in results]
            ghost_positions = [(r[2], r[3]) for r in results]

            order           = sorted(range(N_POP), key=lambda i: fitnesses[i], reverse=True)
            fitnesses       = [fitnesses[i]       for i in order]
            weights_out     = [weights_out[i]     for i in order]
            ghost_positions = [ghost_positions[i] for i in order]

            # ── Hall of fame ──────────────────────────────────────────────
            for fit, w in zip(fitnesses, weights_out):
                if len(hall_of_fame) < HALL_OF_FAME_K:
                    hall_of_fame.append((fit, w.copy()))
                    hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                elif fit > hall_of_fame[-1][0]:
                    hall_of_fame[-1] = (fit, w.copy())
                    hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                else:
                    break

            if hall_of_fame and hall_of_fame[0][0] > best_ever_fitness:
                best_ever_fitness = hall_of_fame[0][0]
                best_ever_w = hall_of_fame[0][1].copy()

            # ── Pack assignment (swarm mode) ──────────────────────────────
            # Behavioural descriptor per individual: where it ended up + score.
            # Clustering these forms "packs" of similarly-driving cars.
            pack_ids = [0] * N_POP
            if evolution_mode == "pack":
                descriptors = np.array(
                    [[gx, gy, fit] for (gx, gy), fit in zip(ghost_positions, fitnesses)],
                    dtype=np.float64,
                )
                pack_ids = assign_packs(descriptors, N_PACKS).tolist()

            # Update live display with current sorted population + pack colours
            pop_holder[0]  = list(weights_out)
            pack_holder[0] = list(pack_ids)

            # ── Top-10 leaderboard + replay (kept consistent) ─────────────
            # A generation enters the leaderboard iff it beats the current 10th
            # best; when it does we also save its replay, so every clickable top
            # score has a matching replay to watch.
            if len(top_10) < 10 or fitnesses[0] > top_10[-1][0]:
                top_10.append((fitnesses[0], gen))
                top_10.sort(key=lambda x: x[0], reverse=True)
                if len(top_10) > 10:
                    top_10.pop()
                _save_replay(weights_out[0], track, fitnesses[0], gen,
                             max_steps=eval_steps)

            # ── Stagnation detection ──────────────────────────────────────
            if fitnesses[0] > prev_best_eval + 0.5:
                prev_best_eval = fitnesses[0]
                stagnation_count = 0
            else:
                stagnation_count += 1
            stagnation_boost = 1.0 + min(2.0, stagnation_count / STAGNATION_GENS)

            # ── Breed next generation ─────────────────────────────────────
            hof_weights = [w for _, w in hall_of_fame]

            if evolution_mode == "pack":
                # Pack / group selection: weak DNA survives by belonging to a
                # strong pack.  See genetics.breed_packs.
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

                # Tier 2: light-mutation copies of the all-time best — let the
                # best result compound without crossover destroying it
                for _ in range(N_BEST_CLONES):
                    if len(new_pop) < N_POP:
                        new_pop.append(mutate(best_ever_w,
                                              MUTATION_RATE * 0.3,
                                              MUTATION_NOISE * 0.3))

                # Tier 3: crossover children — HOF-biased parent selection
                # keeps exploitation pressure; rank_factor tapers mutation up
                # smoothly from near-zero for elite-adjacent slots to full for
                # the tail, so good results are not immediately blown apart.
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

            population = new_pop

            # ── Stats ─────────────────────────────────────────────────────
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
                np.save(save_path, best_ever_w)
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

    stop_event.set()
    if disp is not None:
        disp.join(timeout=2.0)

    np.save(save_path, best_ever_w)
    _write_state(track.name, total_timesteps, total_timesteps,
                 EPSILON_MIN, start_gen + total_gens - 1, best_ever_fitness)
    print(f"[train] Done. Best weights → {save_path}")
    return NeuralAgent(best_ever_w), track


if __name__ == "__main__":
    train()
