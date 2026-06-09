"""A single live session: either a training run or a 'load & drive' run.

The session owns the worker thread and the queues the simulation feeds. The
FastAPI app (app.py) reads `render_q` / `stats_q` and broadcasts to clients.
Only one session exists at a time (single-user student tool); starting a new
one stops the previous.
"""
from __future__ import annotations

import os
import queue
import threading
import time

from f1_rl.config import CIRCUITS, MODEL_PATH, SIM_SPEED_DEFAULT, SIM_SPEED_MAX
from f1_rl.simulation.track_loader import load_track


def _find_circuit(name: str) -> tuple[str | None, float]:
    for cname, fallback, hw in CIRCUITS:
        if cname == name:
            return fallback, hw
    return None, 10.0


class Session:
    def __init__(self) -> None:
        self.mode: str = "idle"          # "idle" | "training" | "driving"
        self.track = None
        self.render_q: queue.Queue | None = None
        self.stats_q: queue.Queue | None = None
        self.line_q: queue.Queue | None = None   # best-lap racing line (x, y, speed)
        self.inspect_q: queue.Queue | None = None  # selected car's net/Q-value state
        self.table_q: queue.Queue | None = None  # selected car's full Q-table (q-table mode)
        # Live-adjustable sim sub-steps per frame (shared with the sim threads via
        # a 1-element list so changes take effect without restarting the run).
        self.speed_holder: list[int] = [SIM_SPEED_DEFAULT]
        # Index of the car to stream network/Q-value detail for (None = nobody).
        self.inspect_holder: list = [None]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        self.mode = "idle"

    def set_speed(self, speed: int) -> None:
        """Change the live-view speed (sub-steps per frame) on the fly."""
        self.speed_holder[0] = max(1, min(SIM_SPEED_MAX, int(speed)))

    def set_inspect(self, index) -> None:
        """Pick which car (by index) to stream net/Q-value detail for; None = off."""
        self.inspect_holder[0] = None if index is None else int(index)

    def _load(self, circuit: str):
        fallback, hw = _find_circuit(circuit)
        return load_track(circuit, geojson_fallback_path=fallback, half_width_m=hw)

    # ------------------------------------------------------------------
    def start_training(self, circuit: str, steps_per_gen: int, total_gens: int,
                       evolution_mode: str, resume: bool, use_rays: bool = True,
                       auto_speed: bool = False) -> None:
        self.stop()
        self._stop = threading.Event()
        self.track = self._load(circuit)
        self.render_q = queue.Queue(maxsize=4)
        self.stats_q = queue.Queue(maxsize=10)
        self.line_q = queue.Queue(maxsize=2)
        self.inspect_q = queue.Queue(maxsize=2)
        # Only the q-table backend streams a full table; the DQN trainer ignores it.
        self.table_q = queue.Queue(maxsize=2) if evolution_mode == "qtable" else None
        self.mode = "training"
        cancel = self._stop

        def run() -> None:
            # "qtable" selects the no-neural-network backend (classic tabular
            # Q-learning); everything else uses the genetic DQN trainer. Both share
            # the same signature, so the rest of the call is identical.
            if evolution_mode == "qtable":
                from f1_rl.learning.qtable import q_learning_loop as train
            else:
                from f1_rl.learning.trainer import train
            try:
                train(track=self.track, render_queue=self.render_q, stats_queue=self.stats_q,
                      line_queue=self.line_q, speed_holder=self.speed_holder,
                      inspect_holder=self.inspect_holder, inspect_queue=self.inspect_q,
                      auto_speed=auto_speed,
                      steps_per_gen=steps_per_gen, total_gens=total_gens,
                      evolution_mode=evolution_mode, resume=resume,
                      use_rays=use_rays, cancel_event=cancel, table_queue=self.table_q)
            except Exception as e:               # noqa: BLE001
                print(f"[server] training error: {e}")
            finally:
                if self.mode == "training":
                    self.mode = "idle"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def start_driving(self, circuit: str, use_rays: bool = True) -> None:
        self.stop()
        self._stop = threading.Event()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No trained model at {MODEL_PATH} — train one first.")
        self.track = self._load(circuit)
        self.render_q = queue.Queue(maxsize=4)
        self.stats_q = None
        self.line_q = queue.Queue(maxsize=2)
        self.inspect_q = queue.Queue(maxsize=2)
        self.table_q = None
        self.mode = "driving"
        stop = self._stop
        track = self.track
        render_q = self.render_q
        line_q = self.line_q
        inspect_q = self.inspect_q
        inspect_holder = self.inspect_holder
        speed_holder = self.speed_holder

        def run() -> None:
            from f1_rl.learning.agent import act, forward_trace, neuronal_net_from_weight_file
            from f1_rl.learning.trainer import _emit_racing_line, _record_replay
            from f1_rl.server.protocol import inspect_to_dict
            from f1_rl.simulation.environment import CarFrame, F1Env
            net = neuronal_net_from_weight_file(MODEL_PATH)
            # Record one lap up front so its racing line is ready, but only stream
            # it AFTER the first live frame — otherwise the line shows before the
            # car appears.
            import numpy as np
            recorded_line = _record_replay(np.load(MODEL_PATH), track, use_rays=use_rays)
            env = F1Env(track=track, render_mode=None, use_rays=use_rays)
            obs, _ = env.reset()
            dt = 1.0 / 60.0
            first_frame = True
            frame_no = 0
            INSPECT_EVERY = 6   # ~10×/s net/Q detail, not 60×/s
            while not stop.is_set():
                t0 = time.perf_counter()
                frame_no += 1
                # Advance several sim steps per rendered frame for a faster, livelier
                # view (frames still emit at 60 fps). The frame shows the final state.
                terminated = False
                for _ in range(max(1, int(speed_holder[0]))):
                    action = act(net, obs)
                    obs, _, terminated, _, _ = env.step(action)
                    if terminated:
                        break
                frame = [CarFrame(
                    x=env.x_m, y=env.y_m, heading=env.heading, speed=env.speed_ms,
                    throttle=env._last_throttle, checkpoint=env._checkpoint_idx,
                    progress=env.progress, lap=env.lap_count,
                    rays=tuple(env._cast_rays()), pack=0, score=env.episode_reward,
                    reward_parts=tuple(env.reward_parts.values()),
                    generation=0, a_long=env._a_long, a_lat=env._a_lat,
                )]
                try:
                    while not render_q.empty():
                        render_q.get_nowait()
                    render_q.put_nowait(frame)
                except Exception:                # noqa: BLE001
                    pass
                if first_frame:
                    _emit_racing_line(line_q, recorded_line)
                    first_frame = False
                # Stream net/Q-value detail for the single car when it's inspected.
                if inspect_holder[0] == 0 and frame_no % INSPECT_EVERY == 0:
                    q, hidden = forward_trace(net, obs)
                    try:
                        while not inspect_q.empty():
                            inspect_q.get_nowait()
                        inspect_q.put_nowait(inspect_to_dict(0, obs, q, hidden, int(q.argmax())))
                    except Exception:            # noqa: BLE001
                        pass
                if terminated:
                    obs, _ = env.reset()
                remaining = dt - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)
            env.close()
            if self.mode == "driving":
                self.mode = "idle"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
