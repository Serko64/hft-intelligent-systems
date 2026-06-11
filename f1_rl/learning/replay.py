"""Aufzeichnung der besten Fahrt — gemeinsam für DQN- und Q-Table-Backend.

Eine Policy fährt einmal greedy (immer die beste Aktion) über die Strecke; die
Positionen werden mitgeschrieben und als Racing-Line ans Web-UI geschickt
(emit_racing_line).
"""
from __future__ import annotations

import numpy as np

from f1_rl.utils.queues import put_latest


def record_greedy_replay(policy, choose_action, track, max_steps: int,
                         use_rays: bool = True) -> np.ndarray:
    """Fährt EINE greedy Episode und gibt die Fahrspur als Array zurück.

    ``policy`` ist je nach Backend ein neuronales Netz oder eine Q-Tabelle;
    ``choose_action(policy, obs) -> int`` wählt die jeweils beste Aktion.
    Jede Zeile des Ergebnisses: (x, y, heading, speed, throttle, progress).
    """
    from f1_rl.simulation.environment import create_car_env, reset_env, step_env

    env = create_car_env(track, use_rays=use_rays)
    obs = reset_env(env)
    frames: list[tuple] = []
    for _ in range(max_steps):
        action = choose_action(policy, obs)
        obs, _, terminated, _, _ = step_env(env, action)
        frames.append((env["x_m"], env["y_m"], env["heading"], env["speed_ms"],
                       env["last_throttle"], env["progress"]))
        # Die ganze Mehrrunden-Episode aufzeichnen (endet mit letzter Runde oder Crash).
        if terminated:
            break
    return np.array(frames, dtype=np.float32)


def emit_racing_line(line_queue, frames: np.ndarray | None) -> None:
    """Schiebt die Fahrspur der besten Runde (x, y, speed, throttle) ans UI.

    Tut nichts, wenn es keine Queue oder keine Frames gibt — kann also
    bedenkenlos aus der Trainingsschleife aufgerufen werden.
    """
    if line_queue is None or frames is None or len(frames) == 0:
        return
    # Frame-Zeile ist (x, y, heading, speed, throttle, progress) — wir brauchen 4 davon.
    line = [(float(f[0]), float(f[1]), float(f[3]), float(f[4])) for f in frames]
    put_latest(line_queue, line)
