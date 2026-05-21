"""PPO training setup using stable-baselines3."""
from __future__ import annotations

import glob
import json
import math
import os
from queue import Full, Queue

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "model")
STATE_PATH = os.path.join(MODEL_DIR, "training_state.json")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")


def load_training_state() -> dict | None:
    """Return saved training state or None if no state exists."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _latest_checkpoint() -> str | None:
    """Return path to the most recent checkpoint zip, or None."""
    pattern = os.path.join(CHECKPOINT_DIR, "ppo_f1_*.zip")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[-1] if ckpts else None


# ── Callbacks ─────────────────────────────────────────────────────────────────

class _SaveStateCallback:
    """Writes training_state.json after every checkpoint interval and at end."""

    def __init__(self, track_name: str, total_timesteps: int, save_freq: int):
        self.track_name = track_name
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.n_calls = 0
        self.model = None
        self.training_env = None
        self.locals: dict = {}
        self.globals: dict = {}
        self.parent = None

    def init_callback(self, model) -> None:
        self.model = model

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_step(self) -> bool:
        self.n_calls += 1
        if self.n_calls % self.save_freq == 0:
            self._write()
        return True

    def on_rollout_end(self) -> None:
        pass

    def on_training_end(self) -> None:
        self._write()

    def update_locals(self, locals_: dict) -> None:
        self.locals.update(locals_)

    def update_globals(self, globals_: dict) -> None:
        self.globals.update(globals_)

    def _write(self) -> None:
        state = {
            "timesteps_done": self.model.num_timesteps,
            "total_timesteps": self.total_timesteps,
            "circuit": self.track_name,
        }
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


class _LiveRenderCallback:
    """Minimal SB3-compatible callback that ships car state to the render queue."""

    def __init__(self, render_queue: Queue, stats_queue: Queue, track):
        self.render_queue = render_queue
        self.stats_queue = stats_queue
        self.track = track
        self.n_calls = 0
        self.model = None
        self.training_env = None
        self.locals: dict = {}
        self.globals: dict = {}
        self.parent = None

    def init_callback(self, model) -> None:
        self.model = model
        self.training_env = model.get_env()

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_step(self) -> bool:
        self.n_calls += 1
        if self.n_calls % 10 == 0:
            locals_src = self.parent.locals if self.parent is not None else self.locals
            new_obs = locals_src.get("new_obs")
            if new_obs is not None:
                obs0 = new_obs[0]
                cx, cy = self.track.bounds_center
                hd = self.track.half_diag_m
                x_m = float(obs0[0]) * hd + cx
                y_m = float(obs0[1]) * hd + cy
                heading = float(obs0[2]) * math.pi
                speed_ms = float(obs0[3]) * 80.0
                actions = locals_src.get("actions")
                throttle = float(actions[0][1]) if actions is not None else 0.0
                try:
                    self.render_queue.put_nowait((x_m, y_m, heading, speed_ms, throttle))
                except Full:
                    pass

        if self.n_calls % 500 == 0:
            try:
                self.stats_queue.put_nowait({
                    "timesteps": self.model.num_timesteps,
                })
            except Full:
                pass
        return True

    def on_rollout_end(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def update_locals(self, locals_: dict) -> None:
        self.locals.update(locals_)

    def update_globals(self, globals_: dict) -> None:
        self.globals.update(globals_)


# ── Env factory ──────────────────────────────────────────────────────────────

def make_env(track, _rank: int):
    """Factory for SubprocVecEnv workers. Inner import avoids pickling the class."""
    def _init():
        from f1_rl.env import F1Env
        return F1Env(track=track, render_mode=None)
    return _init


# ── Main train function ───────────────────────────────────────────────────────

def train(
    track_query: str = "Circuit de Monaco",
    geojson_fallback: str | None = None,
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    save_path: str | None = None,
    render_queue: Queue | None = None,
    stats_queue: Queue | None = None,
    track=None,
    resume: bool = False,
):
    """Load track, train PPO with SubprocVecEnv, save model. Returns (model, track)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from f1_rl.track import load_track

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(MODEL_DIR, "ppo_f1")

    if track is None:
        print(f"[train] Loading track: {track_query}")
        track = load_track(track_query, geojson_fallback_path=geojson_fallback)
    print(f"[train] Track loaded: {track.name}, {track.total_length_m:.0f} m")

    env_fns = [make_env(track, _rank=i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    save_freq = max(50_000 // n_envs, 1)

    if resume:
        ckpt = _latest_checkpoint()
        resume_path = ckpt or (save_path + ".zip")
        if os.path.exists(resume_path):
            print(f"[train] Resuming from {resume_path}")
            model = PPO.load(
                resume_path,
                env=vec_env,
                device="cuda",
                tensorboard_log=os.path.join(MODEL_DIR, "tb_logs"),
            )
            # Ensure we train at least until total_timesteps from current point
            total_timesteps = max(total_timesteps, model.num_timesteps + 1_000_000)
        else:
            print("[train] No checkpoint found — starting fresh")
            resume = False

    if not resume:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
            device="cuda",
            tensorboard_log=os.path.join(MODEL_DIR, "tb_logs"),
        )

    callbacks = []
    callbacks.append(CheckpointCallback(
        save_freq=save_freq,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_f1",
    ))
    callbacks.append(_SaveStateCallback(
        track_name=track.name,
        total_timesteps=total_timesteps,
        save_freq=save_freq,
    ))
    if render_queue is not None and stats_queue is not None:
        callbacks.append(_LiveRenderCallback(render_queue, stats_queue, track))

    print(f"[train] {'Resuming' if resume else 'Starting'} PPO — target {total_timesteps:,} timesteps ...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(save_path)
    vec_env.close()
    print(f"[train] Model saved to {save_path}.zip")
    return model, track


if __name__ == "__main__":
    train()
