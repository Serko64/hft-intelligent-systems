"""Gemeinsame Live-Anzeige der ganzen Population (~60 fps) für beide Backends.

DQN-Trainer und Q-Table-Backend zeigen ihre Population identisch an — nur WIE
eine Aktion gewählt wird unterscheidet sich. Deshalb gibt es hier EINE
Anzeige-Schleife, der die Backends drei kleine Funktionen mitgeben:

  build_policy(individual)      -> policy   (DQN: Gewichte → Netz; Q-Table: Tabelle bleibt Tabelle)
  choose_action(policy, obs)    -> int      (beste Aktion laut Policy)
  inspect_trace_fn(policy, obs) -> (q_values, hidden)   (Detail-Ansicht fürs UI)

Soft-Swap: Kommt eine neue Generation an, fährt jedes Auto mit seiner alten
Policy weiter, bis es natürlich crasht, und übernimmt erst dann die neue —
so gibt es keine Massen-Teleports und gute Läufe werden nicht abgewürgt.
"""
from __future__ import annotations

import threading
import time
from queue import Queue

from f1_rl.utils.queues import put_latest

INSPECT_EVERY = 6      # Netz-/Q-Detail ~10×/s statt 60×/s (hält das UI flüssig)
TABLE_EVERY = 120      # volle Q-Tabelle ~alle 2 s (große Nutzlast)
MAX_DISPLAY_LAG = 2    # Anzeige darf höchstens so viele Generationen hinterherhinken


def run_population_display(
    pop_holder: list,
    stop_event: threading.Event,
    track,
    render_queue: Queue,
    *,
    build_policy,
    choose_action,
    inspect_trace_fn,
    use_rays: bool = True,
    speed_holder: list | None = None,
    gen_holder: list | None = None,
    inspect_holder: list | None = None,
    inspect_queue: Queue | None = None,
    table_queue: Queue | None = None,           # nur Q-Table-Backend
    make_table_payload=None,                    # (policy, car_index) -> dict, nur Q-Table
) -> None:
    """Animiert alle Autos der Population und streamt Frames ans Web-UI.

    Läuft als Daemon-Thread, bis ``stop_event`` gesetzt wird. Liest live aus den
    1-Element-Listen (``pop_holder``, ``speed_holder``, …), damit Änderungen aus
    dem UI ohne Neustart wirken.
    """
    # Imports erst hier, damit der Server schnell startet und Worker-Prozesse
    # (Windows spawn) dieses Modul ohne torch/protocol-Kosten laden können.
    from f1_rl.server.protocol import inspect_to_dict
    from f1_rl.simulation.environment import create_car_env, make_car_frame, reset_env, step_env

    current_pop = pop_holder[0]
    n_cars = len(current_pop)
    envs = [create_car_env(track, use_rays=use_rays) for _ in range(n_cars)]
    policies = [build_policy(individual) for individual in current_pop]
    observations = [reset_env(env) for env in envs]

    # next_policies[i]: wartet darauf, policies[i] beim nächsten Crash zu ersetzen.
    next_policies: list = [None] * n_cars

    if speed_holder is None:
        speed_holder = [1]
    if gen_holder is None:
        gen_holder = [0]
    # car_gens[i]: aus welcher Generation die gerade fahrende Policy von Auto i ist.
    # next_gen[i]: Generation der wartenden Ersatz-Policy (für die Färbung im UI).
    car_gens = [int(gen_holder[0])] * n_cars
    next_gen = [int(gen_holder[0])] * n_cars
    dt = 1.0 / 60.0
    frame_no = 0

    while not stop_event.is_set():
        frame_start = time.perf_counter()
        frame_no += 1
        sim_steps_per_frame = max(1, int(speed_holder[0]))   # Live-Geschwindigkeit

        # Neue Generation angekommen? Policies bauen und zum Soft-Swap einreihen.
        new_pop = pop_holder[0]
        if new_pop is not current_pop:
            current_pop = new_pop
            arriving_gen = int(gen_holder[0])
            for i in range(n_cars):
                next_policies[i] = build_policy(current_pop[i])
                next_gen[i] = arriving_gen

        arriving = int(gen_holder[0])
        frames: list = []
        for i in range(n_cars):
            # Anzeige-Verzug begrenzen: hängt dieses Auto zu viele Generationen
            # zurück, wird es SOFORT auf die neueste Policy gesetzt statt auf den
            # nächsten natürlichen Crash zu warten.
            if next_policies[i] is not None and arriving - car_gens[i] > MAX_DISPLAY_LAG:
                policies[i] = next_policies[i]
                next_policies[i] = None
                car_gens[i] = next_gen[i]
                observations[i] = reset_env(envs[i])

            # Mehrere Sim-Schritte pro Bild; das Bild zeigt den Endzustand.
            terminated = False
            for _ in range(sim_steps_per_frame):
                action = choose_action(policies[i], observations[i])
                observations[i], _, terminated, _, _ = step_env(envs[i], action)
                if terminated:
                    break

            frames.append(make_car_frame(envs[i], generation=car_gens[i]))

            if terminated:
                observations[i] = reset_env(envs[i])
                if next_policies[i] is not None:   # Soft-Swap beim natürlichen Crash
                    policies[i] = next_policies[i]
                    next_policies[i] = None
                    car_gens[i] = next_gen[i]

        put_latest(render_queue, frames)

        selected_car = inspect_holder[0] if inspect_holder is not None else None
        selection_valid = isinstance(selected_car, int) and 0 <= selected_car < n_cars

        # Q-Werte + Detail des inspizierten Autos (~10×/s).
        if inspect_queue is not None and selection_valid and frame_no % INSPECT_EVERY == 0:
            q_values, hidden = inspect_trace_fn(policies[selected_car], observations[selected_car])
            put_latest(inspect_queue, inspect_to_dict(
                selected_car, observations[selected_car], q_values, hidden,
                int(q_values.argmax())))

        # Volle Q-Tabelle des inspizierten Autos als Heatmap (nur Q-Table-Backend).
        if (table_queue is not None and make_table_payload is not None
                and selection_valid and frame_no % TABLE_EVERY == 0):
            put_latest(table_queue, make_table_payload(policies[selected_car], selected_car))

        # Auf 60 fps einbremsen.
        remaining = dt - (time.perf_counter() - frame_start)
        if remaining > 0:
            time.sleep(remaining)
