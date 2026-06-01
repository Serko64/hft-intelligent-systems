"""Serialization between the Python simulation objects and the JSON the browser
front-end consumes over the WebSocket.

All coordinates are in **metres** (the simulation's native units); the front-end
handles scaling/centering for display. The frame `car` tuple layout matches what
`trainer._display_thread` produces:

    (x, y, heading, speed, throttle, checkpoint, progress, lap,
     rays, pack_id, episode_reward, reward_parts)
"""
from __future__ import annotations

from f1_rl.simulation.environment import REWARD_PARTS


def track_to_dict(track) -> dict:
    """Geometry the browser needs to draw the circuit once (centerline + walls)."""
    corr = track.corridor
    poly = max(corr.geoms, key=lambda g: g.area) if corr.geom_type == "MultiPolygon" else corr
    minx, miny, maxx, maxy = corr.bounds
    return {
        "name": track.name,
        "total_length_m": float(track.total_length_m),
        "half_width_m": float(track.half_width_m),
        "centerline": [[float(x), float(y)] for x, y in track.centerline_m.coords],
        "corridor_exterior": [[float(x), float(y)] for x, y in poly.exterior.coords],
        "corridor_interiors": [
            [[float(x), float(y)] for x, y in ring.coords] for ring in poly.interiors
        ],
        "bounds": {"minx": float(minx), "miny": float(miny),
                   "maxx": float(maxx), "maxy": float(maxy)},
    }


def car_to_dict(c: tuple) -> dict:
    """One car's per-frame state."""
    d = {
        "x": float(c[0]), "y": float(c[1]), "heading": float(c[2]),
        "speed": float(c[3]), "throttle": float(c[4]),
        "checkpoint": int(c[5]), "progress": float(c[6]), "lap": int(c[7]),
        "rays": [float(r) for r in c[8]] if len(c) > 8 else [],
        "pack": int(c[9]) if len(c) > 9 else 0,
        "score": float(c[10]) if len(c) > 10 else 0.0,
    }
    if len(c) > 11:
        d["reward_parts"] = {k: float(v) for k, v in zip(REWARD_PARTS, c[11])}
    return d


def stats_to_dict(stats: dict) -> dict:
    """Slim, JSON-safe copy of the training stats (drops the heavy ghost array)."""
    return {
        "timesteps": int(stats.get("timesteps", 0)),
        "generation": int(stats.get("generation", 0)),
        "epsilon": float(stats.get("epsilon", 1.0)),
        "best_fitness": float(stats.get("best_fitness", 0.0)),
        "mean_fitness": float(stats.get("mean_fitness", 0.0)),
        "n_packs": int(stats.get("n_packs", 0)),
        "top_scores": [[float(s), int(g)] for s, g in stats.get("top_scores", [])],
    }
