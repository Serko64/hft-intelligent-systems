import math
import time
from typing import TypedDict

import numpy as np
from shapely.geometry import Point

from f1_rl.config import CURRICULUM_AFTER_LAP, LAP_BONUS, LAPS_PER_EPISODE
from f1_rl.simulation.track_loader import Track

MAX_SPEED_MS = 80.0      # ~288 km/h
STEER_GAIN = 0.15        # rad / (action * speed * dt)
THROTTLE_GAIN = 8.0      # m/s² pro Einheit Gas
FRICTION = 0.98          # Tempo-Abfall pro Schritt beim Rollen ohne Gas
DT = 1.0 / 60.0

# Grip-Kreis: Quer- und Längsbeschleunigung teilen sich EIN Reibungsbudget.
GRIP_ENABLED = True
GRIP_MAX = 45.0          # maximale Gesamtbeschleunigung, m/s² (~4.6 g)

# Auto-Abmessungen (Meter), für Kollisionen ist das Auto ein Rechteck.
CAR_LENGTH_M = 5.0
CAR_WIDTH_M = 2.0

DISCRETE_ACTIONS = [
    (steering, throttle)
    for steering in (-1.0, -0.5, 0.0, 0.5, 1.0)
    for throttle in (-1.0, 0.0, 0.5, 1.0)
]
N_ACTIONS = len(DISCRETE_ACTIONS)          # 20
DEFAULT_SPEED_MS = 40.0 / 3.6              # Startgeschwindigkeit
MIN_DRIVE_SPEED_MS = 5.0 / 3.6             # harte Untergrenze
MIN_SPEED_MS = 20.0 / 3.6                  # Schwelle für die Langsam-Strafe
SLOW_PENALTY = -15.0
N_CHECKPOINTS = 40
STEER_CHANGE_PENALTY = 2                   # Strafe pro Einheit Lenk-Änderung

# Reward-Gewichte
DIST_REWARD_FWD = 25.0   # Belohnung pro Meter Vorwärts-Fortschritt
DIST_REWARD_BACK = 75.0  # Strafe pro rückwärts gefahrenem Meter
SPEED_REWARD = 0.3
CHECKPOINT_BONUS = 10.0
CHECKPOINT_SPEED_BONUS = 5.0
STEP_COST = -0.15        # konstante Kosten pro Schritt = Zeitdruck
WRONG_WAY_PENALTY = 8.0

# Reward-Komponenten werden einzeln mitgezählt, damit das UI sie zeigen kann.
REWARD_PARTS = ("distance", "checkpoint", "speed", "align", "wrongway", "slow",
                "steer", "step", "lap", "crash")

# Raycast-Winkel relativ zur Fahrtrichtung (negativ = rechts, positiv = links).
RAY_ANGLES = tuple(math.radians(a) for a in (-75, -45, -20, 0, 20, 45, 75))
_RAY_ANGLES_ARR = np.asarray(RAY_ANGLES, dtype=np.float64)
MAX_RAY_M = 80.0

# Ohne Rays sieht die Policy konstante Werte, die Beobachtung bleibt aber 14 breit.
_NO_RAYS = (1.0,) * len(RAY_ANGLES)


class CarFrame(TypedDict):
    x: float                 # Position (Meter)
    y: float
    heading: float           # Fahrtrichtung (Radiant)
    speed: float             # m/s
    throttle: float          # letztes Gas-Kommando, -1..1
    checkpoint: int
    progress: float          # Meter entlang der Centerline
    lap: int
    rays: tuple              # Lidar-Abstandswerte, 0..1
    score: float             # kumulativer Episoden-Reward
    reward_parts: tuple      # Reihenfolge = REWARD_PARTS
    generation: int
    a_long: float            # Längsbeschleunigung, m/s²
    a_lat: float             # Querbeschleunigung, m/s²


class CarEnv(TypedDict):
    # konstant nach create_car_env()
    track: Track
    use_rays: bool
    wall_start: np.ndarray        # (K, 2) Anfangspunkte der Wand-Segmente
    wall_end: np.ndarray          # (K, 2) Endpunkte
    wall_mid: np.ndarray          # (K, 2) Mittelpunkte
    wall_cull_radius: np.ndarray  # (K,) max. Treffer-Distanz pro Segment
    # veränderlich pro Schritt
    x_m: float
    y_m: float
    heading: float
    speed_ms: float
    progress: float               # Meter entlang der Centerline
    prev_progress: float
    progress_delta: float
    lap_forward: bool
    lap_count: int
    step_count: int
    lap_start_time: float
    last_throttle: float
    last_steering: float
    understeer: bool
    a_long: float
    a_lat: float
    lateral_offset_m: float       # Abstand zur Centerline (vorzeichenbehaftet)
    track_heading: float          # lokale Streckenrichtung (Radiant)
    checkpoint_idx: int
    episode_reward: float
    reward_parts: dict[str, float]
    last_reward_parts: dict[str, float]


def _corridor_wall_segments(track: Track) -> tuple[np.ndarray, np.ndarray]:
    corridor = track["corridor"]
    polygons = corridor.geoms if corridor.geom_type == "MultiPolygon" else [
        corridor]
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            points = np.asarray(ring.coords, dtype=np.float64)
            if len(points) >= 2:
                start_parts.append(points[:-1])
                end_parts.append(points[1:])
    if not start_parts:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    return np.concatenate(start_parts, axis=0), np.concatenate(end_parts, axis=0)


def create_car_env(track: Track, use_rays: bool = True) -> CarEnv:
    wall_start, wall_end = _corridor_wall_segments(track)
    wall_mid = (wall_start + wall_end) * 0.5
    segment_vec = wall_end - wall_start
    if len(segment_vec):
        segment_half_length = 0.5 * \
            np.hypot(segment_vec[:, 0], segment_vec[:, 1])
    else:
        segment_half_length = np.empty(0)
    wall_cull_radius = MAX_RAY_M + segment_half_length

    return {
        "track": track,
        "use_rays": use_rays,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "wall_mid": wall_mid,
        "wall_cull_radius": wall_cull_radius,
        "x_m": 0.0,
        "y_m": 0.0,
        "heading": 0.0,
        "speed_ms": 0.0,
        "progress": 0.0,
        "prev_progress": 0.0,
        "progress_delta": 0.0,
        "lap_forward": False,
        "lap_count": 0,
        "step_count": 0,
        "lap_start_time": 0.0,
        "last_throttle": 0.0,
        "last_steering": 0.0,
        "understeer": False,
        "a_long": 0.0,
        "a_lat": 0.0,
        "lateral_offset_m": 0.0,
        "track_heading": 0.0,
        "checkpoint_idx": 0,
        "episode_reward": 0.0,
        "reward_parts": {part: 0.0 for part in REWARD_PARTS},
        "last_reward_parts": {part: 0.0 for part in REWARD_PARTS},
    }


def reset_env(env: CarEnv, start_progress: float = 0.0) -> np.ndarray:
    centerline = env["track"]["centerline_m"]
    total = env["track"]["total_length_m"]
    # start_progress = 0 -> klassischer Start an der Start/Ziel-Linie. Ein positiver
    # Wert setzt das Auto an die entsprechende Stelle der Centerline (für Multi-Start-Eval).
    start_progress = start_progress % total if total > 0 else 0.0
    if start_progress <= 0.0:
        coords = list(centerline.coords)
        env["x_m"], env["y_m"] = coords[0]
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        env["heading"] = math.atan2(dy, dx)
    else:
        here = centerline.interpolate(start_progress)
        ahead = centerline.interpolate(min(total, start_progress + 0.5))
        env["x_m"], env["y_m"] = here.x, here.y
        env["heading"] = math.atan2(ahead.y - here.y, ahead.x - here.x)
    env["speed_ms"] = DEFAULT_SPEED_MS
    env["progress"] = start_progress
    env["prev_progress"] = start_progress
    env["progress_delta"] = 0.0
    env["lap_forward"] = False
    env["lap_count"] = 0
    env["step_count"] = 0
    env["lap_start_time"] = time.time()
    env["last_throttle"] = 0.0
    env["last_steering"] = 0.0
    env["understeer"] = False
    env["a_long"] = 0.0
    env["a_lat"] = 0.0
    env["lateral_offset_m"] = 0.0
    env["track_heading"] = 0.0
    # Checkpoint-Index zum Startpunkt passend setzen, sonst gäbe es beim ersten Schritt
    # Bonus-Gutschriften für Tore, die das Auto gar nicht durchfahren hat.
    env["checkpoint_idx"] = (
        int(start_progress / total * N_CHECKPOINTS) if total > 0 else 0)
    env["episode_reward"] = 0.0
    env["reward_parts"] = {part: 0.0 for part in REWARD_PARTS}
    env["last_reward_parts"] = dict(env["reward_parts"])
    return _get_obs(env)


def _update_track_position(env: CarEnv) -> None:
    centerline = env["track"]["centerline_m"]
    projected = centerline.project(Point(env["x_m"], env["y_m"]))
    total = env["track"]["total_length_m"]

    # Naht-Korrektur an Start/Ziel: kürzester vorzeichenbehafteter Weg um die
    # Schleife, sonst würde ein Projektionssprung als Riesenbewegung zählen.
    progress_move = projected - env["progress"]
    if progress_move > total / 2:
        progress_move -= total
    elif progress_move < -total / 2:
        progress_move += total
    max_move = MAX_SPEED_MS * DT * 2
    progress_move = max(-max_move, min(max_move, progress_move))

    raw_new_progress = env["progress"] + progress_move
    env["progress_delta"] = progress_move
    env["lap_forward"] = raw_new_progress >= total
    env["progress"] = raw_new_progress % total

    # Lokale Streckenrichtung aus Punkten kurz vor/hinter der Position.
    lookahead_m = 0.5
    point_behind = centerline.interpolate(
        max(0.0, env["progress"] - lookahead_m))
    point_ahead = centerline.interpolate(
        min(total, env["progress"] + lookahead_m))
    env["track_heading"] = math.atan2(point_ahead.y - point_behind.y,
                                      point_ahead.x - point_behind.x)

    nearest = centerline.interpolate(env["progress"])
    normal_angle = env["track_heading"] + math.pi / 2
    env["lateral_offset_m"] = (
        (env["x_m"] - nearest.x) * math.cos(normal_angle)
        + (env["y_m"] - nearest.y) * math.sin(normal_angle)
    )


def _accumulate_reward(env: CarEnv, parts: dict) -> float:
    total = float(sum(parts.values()))
    for part_name, value in parts.items():
        env["reward_parts"][part_name] += value
    env["episode_reward"] += total
    env["last_reward_parts"] = parts
    return total


def step_env(env: CarEnv, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
    steering, throttle = DISCRETE_ACTIONS[int(action)]
    steer_penalty = -STEER_CHANGE_PENALTY * \
        abs(steering - env["last_steering"])
    env["last_steering"] = steering
    env["last_throttle"] = throttle

    a_long = throttle * THROTTLE_GAIN

    # Grip-Kreis: was für Gas/Bremse draufgeht, fehlt fürs Kurvenfahren.
    if GRIP_ENABLED:
        a_long = max(-GRIP_MAX, min(GRIP_MAX, a_long))
        a_lat_available = math.sqrt(max(0.0, GRIP_MAX ** 2 - a_long ** 2))
    else:
        a_lat_available = float("inf")

    if throttle > 0:
        env["speed_ms"] += a_long * DT
    else:
        env["speed_ms"] = max(0.0, env["speed_ms"] + a_long * DT)
        env["speed_ms"] *= FRICTION
    env["speed_ms"] = max(MIN_DRIVE_SPEED_MS, min(
        env["speed_ms"], MAX_SPEED_MS))

    # Reicht der Grip nicht für die gewünschte Drehrate, untersteuert das Auto.
    yaw_rate = steering * STEER_GAIN * env["speed_ms"]
    a_lat_requested = abs(yaw_rate) * env["speed_ms"]
    if a_lat_requested > a_lat_available and env["speed_ms"] > 0.1:
        yaw_rate_max = a_lat_available / env["speed_ms"]
        yaw_rate = max(-yaw_rate_max, min(yaw_rate_max, yaw_rate))
        env["understeer"] = True
    else:
        env["understeer"] = False
    env["a_long"] = a_long
    env["a_lat"] = yaw_rate * env["speed_ms"]

    env["heading"] += yaw_rate * DT
    env["x_m"] += math.cos(env["heading"]) * env["speed_ms"] * DT
    env["y_m"] += math.sin(env["heading"]) * env["speed_ms"] * DT
    env["step_count"] += 1

    _update_track_position(env)

    # Crash-Prüfung: Auto-Silhouette (Rechteck) auf die Quer-Achse projiziert.
    yaw_vs_track = (env["heading"] - env["track_heading"] +
                    math.pi) % (2 * math.pi) - math.pi
    half_extent = ((CAR_WIDTH_M / 2) * abs(math.cos(yaw_vs_track))
                   + (CAR_LENGTH_M / 2) * abs(math.sin(yaw_vs_track)))
    if abs(env["lateral_offset_m"]) + half_extent > env["track"]["half_width_m"]:
        _accumulate_reward(env, {"crash": -500.0})
        return _get_obs(env), -500.0, True, False, {}

    total = env["track"]["total_length_m"]

    # Rundenzählung: Episode endet erst nach LAPS_PER_EPISODE Runden.
    lap_bonus_this_step = 0.0
    if env["lap_forward"]:
        env["lap_count"] += 1
        env["prev_progress"] = env["progress"]
        env["checkpoint_idx"] = 0
        if env["lap_count"] >= LAPS_PER_EPISODE:
            _accumulate_reward(env, {"lap": LAP_BONUS})
            return _get_obs(env), LAP_BONUS, True, False, {"lap_complete": True}
        lap_bonus_this_step = LAP_BONUS

    # Checkpoint-Tore: fester Bonus + Tempo-Bonus pro überschrittenem Tor.
    checkpoint_reward = 0.0
    next_gate_at_m = (env["checkpoint_idx"] + 1) / N_CHECKPOINTS * total
    while env["checkpoint_idx"] < N_CHECKPOINTS - 1 and env["progress"] >= next_gate_at_m:
        checkpoint_reward += (CHECKPOINT_BONUS
                              + (env["speed_ms"] / MAX_SPEED_MS) * CHECKPOINT_SPEED_BONUS)
        env["checkpoint_idx"] += 1
        next_gate_at_m = (env["checkpoint_idx"] + 1) / N_CHECKPOINTS * total

    progress_gain = env["progress_delta"]
    speed_norm = env["speed_ms"] / MAX_SPEED_MS
    if env["speed_ms"] < MIN_SPEED_MS:
        slow_penalty = SLOW_PENALTY * (1.0 - (env["speed_ms"] - MIN_DRIVE_SPEED_MS)
                                       / (MIN_SPEED_MS - MIN_DRIVE_SPEED_MS))
    else:
        slow_penalty = 0.0
    heading_error = (env["heading"] - env["track_heading"] +
                     math.pi) % (2 * math.pi) - math.pi
    cos_heading = math.cos(heading_error)
    alignment_bonus = max(0.0, cos_heading - 0.5) * 0.3
    wrong_way_penalty = WRONG_WAY_PENALTY * min(0.0, cos_heading)
    if progress_gain >= 0:
        dist_reward = progress_gain * DIST_REWARD_FWD
    else:
        dist_reward = progress_gain * DIST_REWARD_BACK
    # Curriculum: Optimierungs-Rewards zählen erst nach der ersten vollen Runde.
    refine = 1.0 if env["lap_count"] >= CURRICULUM_AFTER_LAP else 0.0
    reward = _accumulate_reward(env, {
        "distance":   dist_reward,
        "checkpoint": checkpoint_reward,
        "speed":      speed_norm * SPEED_REWARD * refine,
        "align":      alignment_bonus,
        "wrongway":   wrong_way_penalty,
        "slow":       slow_penalty * refine,
        "steer":      steer_penalty * refine,
        "step":       STEP_COST * refine,
        "lap":        lap_bonus_this_step,
    })
    env["prev_progress"] = env["progress"]

    return _get_obs(env), reward, False, False, {}


def cast_rays(env: CarEnv) -> list[float]:
    n_rays = len(RAY_ANGLES)
    segment_start, segment_end = env["wall_start"], env["wall_end"]
    if len(segment_start) == 0:
        return [1.0] * n_rays

    origin_x, origin_y = env["x_m"], env["y_m"]

    # Vorab-Aussieben: Segmente außerhalb der Ray-Reichweite verwerfen.
    is_near = (np.hypot(env["wall_mid"][:, 0] - origin_x,
                        env["wall_mid"][:, 1] - origin_y) <= env["wall_cull_radius"])
    segment_start = segment_start[is_near]
    segment_end = segment_end[is_near]
    if len(segment_start) == 0:
        return [1.0] * n_rays

    segment_dx = segment_end[:, 0] - segment_start[:, 0]
    segment_dy = segment_end[:, 1] - segment_start[:, 1]
    to_segment_x = segment_start[:, 0] - origin_x
    to_segment_y = segment_start[:, 1] - origin_y

    ray_angles = env["heading"] + _RAY_ANGLES_ARR
    ray_dx = np.cos(ray_angles)[:, None]
    ray_dy = np.sin(ray_angles)[:, None]

    # Linien-Schnitt über 2D-Kreuzprodukte für alle Paare gleichzeitig.
    denominator = segment_dx[None, :] * ray_dy - \
        segment_dy[None, :] * ray_dx
    cross_segment_offset = segment_dx * \
        to_segment_y - segment_dy * to_segment_x
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_along_ray = cross_segment_offset[None, :] / denominator
        frac_along_segment = (ray_dx * to_segment_y[None, :]
                              - ray_dy * to_segment_x[None, :]) / denominator
    hit = ((np.abs(denominator) > 1e-12)
           & (frac_along_segment >= 0.0) & (frac_along_segment <= 1.0)
           & (dist_along_ray >= 0.0))

    dist_along_ray = np.where(hit, dist_along_ray, np.inf)
    nearest_hit = np.clip(np.min(dist_along_ray, axis=1),
                          0.0, MAX_RAY_M)
    return (nearest_hit / MAX_RAY_M).tolist()


def _get_obs(env: CarEnv) -> np.ndarray:
    center_x, center_y = env["track"]["bounds_center"]
    half_diag = env["track"]["half_diag_m"]
    total = env["track"]["total_length_m"]
    half_width = env["track"]["half_width_m"]
    heading_error = (env["heading"] - env["track_heading"] +
                     math.pi) % (2 * math.pi) - math.pi
    # Abstände werden von der Autokante gemessen, passend zum Kollisionsmodell.
    half_extent = ((CAR_WIDTH_M / 2) * abs(math.cos(heading_error))
                   + (CAR_LENGTH_M / 2) * abs(math.sin(heading_error)))
    dist_left = max(0.0, min(
        1.0, (half_width + env["lateral_offset_m"] - half_extent) / half_width))
    dist_right = max(0.0, min(
        1.0, (half_width - env["lateral_offset_m"] - half_extent) / half_width))
    rays = cast_rays(env) if env["use_rays"] else _NO_RAYS
    return np.array(
        [
            max(-1.0, min(1.0, (env["x_m"] - center_x) / half_diag)),
            max(-1.0, min(1.0, (env["y_m"] - center_y) / half_diag)),
            heading_error / math.pi,
            env["speed_ms"] / MAX_SPEED_MS,
            env["progress"] / total if total > 0 else 0.0,
            dist_left,
            dist_right,
            *rays,
        ],
        dtype=np.float32,
    )


def make_car_frame(env: CarEnv, *, generation: int = 0) -> CarFrame:
    return {
        "x": env["x_m"],
        "y": env["y_m"],
        "heading": env["heading"],
        "speed": env["speed_ms"],
        "throttle": env["last_throttle"],
        "checkpoint": env["checkpoint_idx"],
        "progress": env["progress"],
        "lap": env["lap_count"],
        "rays": tuple(cast_rays(env)),
        "score": env["episode_reward"],
        "reward_parts": tuple(env["reward_parts"][part] for part in REWARD_PARTS),
        "generation": generation,
        "a_long": env["a_long"],
        "a_lat": env["a_lat"],
    }
