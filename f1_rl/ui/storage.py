"""Loading and saving for the UI: best lap times, the replay list, and the
ReplayState object used by the playback view.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

from f1_rl.config import BEST_LAPS_PATH, CIRCUITS, REPLAY_DIR


# ── Best lap times ────────────────────────────────────────────────────────────

def load_best_laps() -> dict[str, float]:
    if os.path.exists(BEST_LAPS_PATH):
        with open(BEST_LAPS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_best_lap(circuit_name: str, seconds: float) -> None:
    laps = load_best_laps()
    laps[circuit_name] = seconds
    os.makedirs(os.path.dirname(BEST_LAPS_PATH), exist_ok=True)
    with open(BEST_LAPS_PATH, "w", encoding="utf-8") as f:
        json.dump(laps, f, indent=2)


# ── Replay list ───────────────────────────────────────────────────────────────

def load_replay_list() -> list[dict]:
    if not os.path.exists(REPLAY_DIR):
        return []
    replays = []
    for path in glob.glob(os.path.join(REPLAY_DIR, "replay_*.npz")):
        try:
            d = np.load(path, allow_pickle=True)
            replays.append({
                "path":       path,
                "fitness":    float(d["fitness"][0]),
                "generation": int(d["generation"][0]),
                "circuit":    str(d["circuit"][0]),
                "n_frames":   int(d["frames"].shape[0]),
            })
        except Exception:
            pass
    replays.sort(key=lambda r: r["fitness"], reverse=True)
    return replays


def find_circuit(name: str) -> tuple[str | None, float]:
    for cname, fallback, hw in CIRCUITS:
        if cname == name:
            return fallback, hw
    return None, 20.0


# ── Replay playback state ──────────────────────────────────────────────────────

@dataclass
class ReplayState:
    frames: np.ndarray | None = None
    track: object = None
    idx: int = 0
    paused: bool = False
    speed: int = 1
    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0
    dragging: bool = False
    drag_start: tuple = field(default_factory=lambda: (0, 0))
    drag_start_pan: tuple = field(default_factory=lambda: (0, 0))
    meta: dict = field(default_factory=dict)
    return_state: str = "replays"

    def reset_view(self) -> None:
        self.pan_x = self.pan_y = 0
        self.zoom = 1.0

    def reset_pan(self) -> None:
        self.pan_x = self.pan_y = 0


def open_replay_file(path: str, meta: dict, replay: ReplayState, return_state: str) -> bool:
    """Load a .npz replay into *replay* in-place. Returns True on success."""
    from f1_rl.simulation.track_loader import load_track
    try:
        d = np.load(path, allow_pickle=True)
        circuit_name = str(d["circuit"][0])
        fallback, hw = find_circuit(circuit_name)
        replay.track  = load_track(circuit_name, geojson_fallback_path=fallback, half_width_m=hw)
        replay.frames = d["frames"]
        replay.meta   = meta or {
            "fitness":    float(d["fitness"][0]),
            "generation": int(d["generation"][0]),
            "circuit":    circuit_name,
        }
        replay.idx          = 0
        replay.paused       = False
        replay.speed        = 1
        replay.dragging     = False
        replay.return_state = return_state
        replay.reset_view()
        return True
    except Exception as e:
        print(f"[main] Could not open replay {path}: {e}")
        return False


def try_open_training_replay(gen: int, replay: ReplayState) -> bool:
    """Load a generation's saved replay for side-watching during training."""
    path = os.path.join(REPLAY_DIR, f"replay_g{gen:04d}.npz")
    if not os.path.exists(path):
        print(f"[main] Replay for gen {gen} not saved yet.")
        return False
    return open_replay_file(path, {}, replay, return_state="training")
