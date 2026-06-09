"""Serialization between the Python simulation objects and the JSON the browser
front-end consumes over the WebSocket.

All coordinates are in **metres** (the simulation's native units); the front-end
handles scaling/centering for display. Each car is a ``CarFrame`` (a NamedTuple
defined in environment.py) — see its field list there.
"""
from __future__ import annotations

from f1_rl.simulation.environment import REWARD_PARTS, CarFrame

# Track-edge bands measured OUTWARD from the racing surface edge, in metres.
# Kept deliberately THIN: large run-off buffers (15 m) merge across straights and
# fill the infield on compact circuits, which looks like blobs rather than a
# track. A slim red kerb + a narrow asphalt verge reads as a clean racetrack edge.
_TRACK_ZONES = [
    ("runoff", 4.0),
]


def _poly_rings(poly) -> dict:
    return {
        "exterior": [[float(x), float(y)] for x, y in poly.exterior.coords],
        "interiors": [[[float(x), float(y)] for x, y in r.coords] for r in poly.interiors],
    }


def _geom_polys(geom) -> list:
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def track_to_dict(track) -> dict:
    """Geometry the browser needs to draw the circuit once: the racing surface,
    plus the official run-off zones (kerb / asphalt run-off / gravel / barrier)
    as outward buffers, so the edge looks like the real FIA cross-section."""
    corr = track.corridor
    poly = max(corr.geoms, key=lambda g: g.area) if corr.geom_type == "MultiPolygon" else corr
    minx, miny, maxx, maxy = corr.bounds

    # Concentric run-off bands, outermost first (the front-end stacks them so each
    # inner band sits on top, revealing the one beneath as a ring).
    zones = []
    for name, dist in _TRACK_ZONES:
        band = corr.buffer(dist, join_style=1).simplify(0.5, preserve_topology=True)
        zones.append({"name": name, "polygons": [_poly_rings(p) for p in _geom_polys(band)]})
    zones.reverse()  # send outermost (barrier) first

    return {
        "name": track.name,
        "total_length_m": float(track.total_length_m),
        "half_width_m": float(track.half_width_m),
        "centerline": [[float(x), float(y)] for x, y in track.centerline_m.coords],
        "corridor_exterior": [[float(x), float(y)] for x, y in poly.exterior.coords],
        "corridor_interiors": [
            [[float(x), float(y)] for x, y in ring.coords] for ring in poly.interiors
        ],
        "zones": zones,
        "bounds": {"minx": float(minx), "miny": float(miny),
                   "maxx": float(maxx), "maxy": float(maxy)},
    }


def car_to_dict(c: CarFrame) -> dict:
    """One car's per-frame state, as JSON-ready primitives."""
    return {
        "x": float(c.x), "y": float(c.y), "heading": float(c.heading),
        "speed": float(c.speed), "throttle": float(c.throttle),
        "checkpoint": int(c.checkpoint), "progress": float(c.progress),
        "lap": int(c.lap),
        "rays": [float(r) for r in c.rays],
        "pack": int(c.pack),
        "score": float(c.score),
        "reward_parts": {k: float(v) for k, v in zip(REWARD_PARTS, c.reward_parts)},
        "generation": int(c.generation),
        "a_long": float(c.a_long),
        "a_lat": float(c.a_lat),
    }


def racing_line_to_dict(points) -> dict:
    """The best lap's path for the browser to draw, coloured by speed.

    ``points`` is an iterable of (x, y, speed, throttle): metres / m·s⁻¹ / −1..1.
    vmin/vmax (speed) are sent alongside so the front-end can map speed → colour
    without a second pass; throttle drives the brake-vs-accel chart.
    """
    pts = [[float(x), float(y), float(v), float(thr)] for x, y, v, thr in points]
    speeds = [p[2] for p in pts] or [0.0]
    return {"points": pts, "vmin": min(speeds), "vmax": max(speeds)}


def inspect_to_dict(index: int, obs, q, hidden, action: int) -> dict:
    """One car's network state for the live net/Q-value visualisation.

    obs = 14 inputs, q = 20 Q-values (the DQN "table" for this state), hidden =
    per-layer post-ReLU activations, action = the greedy (chosen) action index.
    """
    return {
        "index": int(index),
        "obs": [float(v) for v in obs],
        "q": [float(v) for v in q],
        "hidden": [[float(v) for v in layer] for layer in hidden],
        "action": int(action),
    }


def qtable_to_dict(index: int, states, values, feature_idx, n_bins: int) -> dict:
    """The full learned Q-table of the inspected car, for the heatmap view.

    ``states`` is an iterable of discretised state keys (tuples of bin indices),
    ``values`` the aligned Q-rows (one ``N_ACTIONS`` vector per state). ``feature_idx``
    / ``n_bins`` describe the discretisation so the front-end can label the axes.
    Pure (takes primitives only) — built by the q-table display thread, which holds
    FEATURE_IDX / N_BINS — so there is no import cycle with learning.qtable.
    """
    return {
        "index": int(index),
        "states": [[int(b) for b in s] for s in states],
        "values": [[float(v) for v in row] for row in values],
        "feature_idx": [int(i) for i in feature_idx],
        "n_bins": int(n_bins),
    }


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
