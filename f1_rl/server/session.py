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

from f1_rl.config import CIRCUITS, MODEL_PATH
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        self.mode = "idle"

    def _load(self, circuit: str):
        fallback, hw = _find_circuit(circuit)
        return load_track(circuit, geojson_fallback_path=fallback, half_width_m=hw)

    # ------------------------------------------------------------------
    def start_training(self, circuit: str, steps_per_gen: int, total_gens: int,
                       evolution_mode: str, resume: bool) -> None:
        self.stop()
        self._stop = threading.Event()
        self.track = self._load(circuit)
        self.render_q = queue.Queue(maxsize=4)
        self.stats_q = queue.Queue(maxsize=10)
        self.mode = "training"
        cancel = self._stop

        def run() -> None:
            from f1_rl.learning.trainer import train
            try:
                train(track=self.track, render_queue=self.render_q, stats_queue=self.stats_q,
                      steps_per_gen=steps_per_gen, total_gens=total_gens,
                      evolution_mode=evolution_mode, resume=resume, cancel_event=cancel)
            except Exception as e:               # noqa: BLE001
                print(f"[server] training error: {e}")
            finally:
                if self.mode == "training":
                    self.mode = "idle"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def start_driving(self, circuit: str) -> None:
        self.stop()
        self._stop = threading.Event()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No trained model at {MODEL_PATH} — train one first.")
        self.track = self._load(circuit)
        self.render_q = queue.Queue(maxsize=4)
        self.stats_q = None
        self.mode = "driving"
        stop = self._stop
        track = self.track
        render_q = self.render_q

        def run() -> None:
            from f1_rl.learning.agent import NeuralAgent
            from f1_rl.simulation.environment import F1Env
            agent = NeuralAgent.load(MODEL_PATH)
            env = F1Env(track=track, render_mode=None)
            obs, _ = env.reset()
            dt = 1.0 / 60.0
            while not stop.is_set():
                t0 = time.perf_counter()
                action, _ = agent.predict(obs)
                obs, _, terminated, _, _ = env.step(action)
                frame = [(
                    env.x_m, env.y_m, env.heading, env.speed_ms, env._last_throttle,
                    env._checkpoint_idx, env.progress, env.lap_count,
                    tuple(env._cast_rays()), 0, env.episode_reward,
                    tuple(env.reward_parts.values()),
                )]
                try:
                    while not render_q.empty():
                        render_q.get_nowait()
                    render_q.put_nowait(frame)
                except Exception:                # noqa: BLE001
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
