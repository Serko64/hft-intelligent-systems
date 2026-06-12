"""F1-Fahrsimulation mit kinematischer Fahrzeugphysik — als reine Funktionen.

Es gibt keine Klasse: der komplette Zustand eines Autos steckt in einem
normalen Dict (Typ-Beschreibung: ``CarEnv``), und drei Funktionen arbeiten damit:

  create_car_env(track)  -> env             neuen Auto-Zustand anlegen
  reset_env(env)         -> obs             Auto an den Start setzen
  step_env(env, action)  -> (obs, reward, terminated, truncated, info)

Beobachtung: 14 Zahlen in [-1, 1] (Position, Richtungsabweichung, Tempo,
Fortschritt, Abstand links/rechts und 7 Lidar-Strahlen). Aktion: eine von 20
diskreten (Lenkung, Gas)-Kombinationen. Das Index-Layout steht in config.py.
"""
from __future__ import annotations

import math
import time
from typing import TypedDict

import numpy as np
from shapely.geometry import Point

from f1_rl.config import CURRICULUM_AFTER_LAP, LAP_BONUS, LAPS_PER_EPISODE
from f1_rl.simulation.track_loader import Track

MAX_SPEED_MS = 80.0      # ~288 km/h
STEER_GAIN = 0.15      # rad / (action * speed * dt)
THROTTLE_GAIN = 8.0      # m/s² pro Einheit Gas
FRICTION = 0.98          # Tempo-Abfall pro Schritt beim Rollen ohne Gas
DT = 1.0 / 60.0

# ── Fahrphysik: Grip-Kreis ─────────────────────────────────────────
# Quer- (Kurven-) und Längsbeschleunigung (Gas/Bremse) teilen sich EIN
# Reibungsbudget GRIP_MAX. Wer härter in die Kurve will, als der Grip hergibt,
# untersteuert (das Auto schiebt geradeaus) statt magisch zu drehen — das
# erzwingt Bremsen vor Kurven und erzeugt eine echte Ideallinie.
GRIP_ENABLED = True      # False = altes Arcade-Modell (unbegrenzte Kurvenkraft)
GRIP_MAX     = 45.0      # maximale Gesamtbeschleunigung, m/s²  (~4.6 g, F1-typisch)

# Physische Auto-Abmessungen (Meter). Für Kollisionen ist das Auto ein
# Rechteck — nicht nur sein Mittelpunkt muss auf die Strecke passen.
CAR_LENGTH_M = 5.0
CAR_WIDTH_M = 2.0

DISCRETE_ACTIONS = [
    (steering, throttle)
    for steering in (-1.0, -0.5, 0.0, 0.5, 1.0)
    for throttle in (-1.0, 0.0, 0.5, 1.0)
]
N_ACTIONS = len(DISCRETE_ACTIONS)          # 20
DEFAULT_SPEED_MS  = 40.0 / 3.6            # 40 km/h Startgeschwindigkeit
MIN_DRIVE_SPEED_MS = 5.0 / 3.6           # harte Untergrenze — nie ganz stehenbleiben
MIN_SPEED_MS = 20.0 / 3.6               # Schwelle für die Langsam-Strafe
SLOW_PENALTY = -15.0                      # Strafhöhe bei MIN_DRIVE_SPEED
N_CHECKPOINTS = 40                        # Belohnungs-Tore gleichmäßig über die Runde
# Strafe pro Einheit LenkÄNDERUNG zwischen zwei Schritten. Bestraft hektisches
# Links-Rechts-Zappeln, erlaubt aber gehaltenes Kurvenfahren (Änderung = 0).
STEER_CHANGE_PENALTY = 2

# Reward-Gewichte — hier den Schwerpunkt verschieben. Distanz dominiert Tempo
# um ~250× (DIST_REWARD_FWD vs SPEED_REWARD).
DIST_REWARD_FWD        = 25.0   # Belohnung pro Meter Vorwärts-Fortschritt
DIST_REWARD_BACK       = 75.0   # Strafe pro rückwärts gefahrenem Meter (> vorwärts)
SPEED_REWARD           = 0.3    # Bonus für Tempo (gegen "langsam aber weit"-Schleichen)
CHECKPOINT_BONUS       = 10.0   # fester Bonus pro durchfahrenem Checkpoint-Tor
CHECKPOINT_SPEED_BONUS = 5.0    # zusätzlich pro Tor, skaliert mit Tempo
STEP_COST              = -0.15  # konstante Kosten pro Schritt = Zeitdruck
WRONG_WAY_PENALTY      = 8.0    # Strafe pro Schritt bei Fahrt in die falsche Richtung

# Der Reward jedes Schritts ist die Summe dieser Komponenten. Sie werden einzeln
# (kumulativ pro Episode) mitgezählt, damit das UI die Zusammensetzung zeigt.
REWARD_PARTS = ("distance", "checkpoint", "speed", "align", "wrongway", "slow",
                "steer", "step", "lap", "crash")

# Raycast-Winkel relativ zur Fahrtrichtung (negativ = rechts, positiv = links).
# 7 Strahlen: weite Peripherie (±75°) für frühe Wand-Erkennung, enge (±20°) für Präzision.
RAY_ANGLES = tuple(math.radians(a) for a in (-75, -45, -20, 0, 20, 45, 75))
_RAY_ANGLES_ARR = np.asarray(RAY_ANGLES, dtype=np.float64)
MAX_RAY_M = 80.0         # Normierungs-Reichweite der Strahlen

# Ohne Rays sieht die Policy in den 7 Ray-Slots diese Konstante — ein fester Wert
# trägt keine Information, der Agent fährt also "blind" (keine Wandsensoren).
# Die Beobachtung bleibt 14 Werte breit, damit Modelle kompatibel bleiben.
_NO_RAYS = (1.0,) * len(RAY_ANGLES)


class CarFrame(TypedDict):
    """Zustand EINES Autos für EIN gerendertes Bild (reines Dict, Schlüssel =
    Feldnamen, die das Web-Frontend erwartet — siehe protocol.car_to_dict)."""
    x: float                 # Position (Meter)
    y: float
    heading: float           # Fahrtrichtung (Radiant)
    speed: float             # m/s
    throttle: float          # letztes Gas-Kommando, -1..1
    checkpoint: int          # Index des zuletzt passierten Belohnungs-Tors
    progress: float          # Meter entlang der Centerline
    lap: int                 # abgeschlossene Runden
    rays: tuple              # Lidar-Abstandswerte, 0..1
    score: float             # laufender kumulativer Episoden-Reward
    reward_parts: tuple      # Reward-Aufschlüsselung (Reihenfolge = REWARD_PARTS)
    generation: int          # aus welcher Evolutions-Generation die Policy stammt
    a_long: float            # Längsbeschleunigung dieses Schritts, m/s² (+Gas / −Bremse)
    a_lat: float             # Querbeschleunigung (Kurve) dieses Schritts, m/s²


class CarEnv(TypedDict):
    """Kompletter Zustand eines simulierten Autos — ein normales Dict.

    Die ersten Schlüssel sind nach create_car_env() konstant (Streckendaten),
    der Rest wird von reset_env()/step_env() bei jedem Schritt verändert.
    """
    # — konstant nach create_car_env() —
    track: Track
    use_rays: bool
    wall_start: np.ndarray        # (K, 2) Anfangspunkte der Wand-Segmente (Meter)
    wall_end: np.ndarray          # (K, 2) Endpunkte der Wand-Segmente
    wall_mid: np.ndarray          # (K, 2) Segment-Mittelpunkte (für das Vorab-Aussieben)
    wall_cull_radius: np.ndarray  # (K,) max. Distanz, in der ein Ray das Segment treffen kann
    # — veränderlich pro Schritt —
    x_m: float
    y_m: float
    heading: float
    speed_ms: float
    progress: float               # Meter entlang der Centerline, 0..total_length
    prev_progress: float
    progress_delta: float         # vorzeichenbehaftete Bewegung im letzten Schritt (Meter)
    lap_forward: bool             # Ziellinie im letzten Schritt vorwärts überquert?
    lap_count: int
    step_count: int
    lap_start_time: float
    last_throttle: float
    last_steering: float
    understeer: bool
    a_long: float                 # zuletzt angewendete Längsbeschleunigung (m/s²)
    a_lat: float                  # zuletzt angewendete Querbeschleunigung (m/s²)
    lateral_offset_m: float       # Abstand zur Centerline (Meter, vorzeichenbehaftet)
    track_heading: float          # lokale Streckenrichtung (Radiant)
    checkpoint_idx: int           # nächstes freizuschaltendes Checkpoint-Tor
    episode_reward: float
    reward_parts: dict[str, float]
    last_reward_parts: dict[str, float]


def _corridor_wall_segments(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Zerlegt den Rand des Streckenkorridors in gerade Liniensegmente (Meter).

    Gibt (start, end) zurück, beide mit Shape (K, 2): Segment k läuft von
    start[k] nach end[k]. Enthält den Außenrand UND alle Innenränder (Infield),
    damit Rays von beiden Streckenkanten blockiert werden.
    """
    corridor = track["corridor"]
    polygons = corridor.geoms if corridor.geom_type == "MultiPolygon" else [corridor]
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            points = np.asarray(ring.coords, dtype=np.float64)
            if len(points) >= 2:
                start_parts.append(points[:-1])   # alle Punkte außer dem letzten
                end_parts.append(points[1:])      # alle Punkte außer dem ersten
    if not start_parts:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    return np.concatenate(start_parts, axis=0), np.concatenate(end_parts, axis=0)


def create_car_env(track: Track, use_rays: bool = True) -> CarEnv:
    """Legt einen neuen Auto-Zustand an. Danach immer zuerst reset_env() aufrufen."""
    wall_start, wall_end = _corridor_wall_segments(track)
    wall_mid = (wall_start + wall_end) * 0.5
    segment_vec = wall_end - wall_start
    if len(segment_vec):
        segment_half_length = 0.5 * np.hypot(segment_vec[:, 0], segment_vec[:, 1])
    else:
        segment_half_length = np.empty(0)
    # Ein Ray (max. MAX_RAY_M lang) kann ein Segment nur treffen, wenn dessen
    # Mittelpunkt höchstens MAX_RAY_M + halbe Segmentlänge entfernt ist.
    wall_cull_radius = MAX_RAY_M + segment_half_length

    return {
        "track": track,
        "use_rays": use_rays,   # False -> Policy bekommt konstante (blinde) Ray-Werte
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


def reset_env(env: CarEnv) -> np.ndarray:
    """Setzt das Auto an den Start (Anfang der Centerline) und gibt die Beobachtung zurück."""
    coords = list(env["track"]["centerline_m"].coords)
    env["x_m"], env["y_m"] = coords[0]
    # Anfangs-Fahrtrichtung = Richtung vom ersten zum zweiten Centerline-Punkt
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    env["heading"] = math.atan2(dy, dx)
    env["speed_ms"] = DEFAULT_SPEED_MS
    env["progress"] = 0.0
    env["prev_progress"] = 0.0
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
    env["checkpoint_idx"] = 0
    env["episode_reward"] = 0.0
    env["reward_parts"] = {part: 0.0 for part in REWARD_PARTS}
    env["last_reward_parts"] = dict(env["reward_parts"])
    return _get_obs(env)


def _update_track_position(env: CarEnv) -> None:
    """Projiziert das Auto einmal pro Schritt auf die Centerline und merkt sich
    Fortschritt, Seitenversatz und lokale Streckenrichtung."""
    centerline = env["track"]["centerline_m"]
    projected = centerline.project(Point(env["x_m"], env["y_m"]))
    total = env["track"]["total_length_m"]

    # Vorzeichenbehaftete Bewegung seit dem letzten Schritt, korrigiert an der
    # Start/Ziel-Naht: shapelys Projektion springt zu einem WEIT entfernten Teil
    # der Strecke, wenn das Auto an Start/Ziel rückwärts schaut (der nächste
    # Centerline-Punkt liegt dann am Rundenende). Der kürzeste vorzeichenbehaftete
    # Weg um die Schleife macht aus diesem falschen +total-Sprung die kleine
    # RÜCKWÄRTS-Bewegung, die er wirklich ist.
    progress_move = projected - env["progress"]
    if progress_move > total / 2:
        progress_move -= total
    elif progress_move < -total / 2:
        progress_move += total
    # Auf eine physikalisch mögliche Bewegung pro Schritt begrenzen (2× Topspeed
    # × dt), damit ein Projektions-Glitch nie als Riesensprung gelesen wird.
    max_move = MAX_SPEED_MS * DT * 2
    progress_move = max(-max_move, min(max_move, progress_move))

    raw_new_progress = env["progress"] + progress_move
    env["progress_delta"] = progress_move              # echter Fortschritt dieses Schritts
    env["lap_forward"] = raw_new_progress >= total     # Ziellinie vorwärts überquert
    env["progress"] = raw_new_progress % total

    # Lokale Streckenrichtung: Richtung zwischen einem Punkt kurz hinter und
    # kurz vor der aktuellen Position auf der Centerline.
    lookahead_m = 0.5
    point_behind = centerline.interpolate(max(0.0, env["progress"] - lookahead_m))
    point_ahead = centerline.interpolate(min(total, env["progress"] + lookahead_m))
    env["track_heading"] = math.atan2(point_ahead.y - point_behind.y,
                                      point_ahead.x - point_behind.x)

    # Seitenversatz: Auto-Position auf die Achse quer zur Strecke projiziert.
    nearest = centerline.interpolate(env["progress"])
    normal_angle = env["track_heading"] + math.pi / 2
    env["lateral_offset_m"] = (
        (env["x_m"] - nearest.x) * math.cos(normal_angle)
        + (env["y_m"] - nearest.y) * math.sin(normal_angle)
    )


def _accumulate_reward(env: CarEnv, parts: dict) -> float:
    """Addiert die Reward-Komponenten dieses Schritts auf die laufenden Summen;
    gibt die Schritt-Summe zurück."""
    total = float(sum(parts.values()))
    for part_name, value in parts.items():
        env["reward_parts"][part_name] += value
    env["episode_reward"] += total
    env["last_reward_parts"] = parts
    return total


def step_env(env: CarEnv, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
    """Führt eine Aktion aus und simuliert einen Zeitschritt (1/60 s).

    Rückgabe wie bei gym üblich: (obs, reward, terminated, truncated, info).
    terminated = Crash oder alle Runden geschafft; truncated wird nie gesetzt.
    """
    steering, throttle = DISCRETE_ACTIONS[int(action)]
    steer_penalty = -STEER_CHANGE_PENALTY * abs(steering - env["last_steering"])
    env["last_steering"] = steering
    env["last_throttle"] = throttle

    # Längsbeschleunigung aus Gas/Bremse.
    a_long = throttle * THROTTLE_GAIN

    # Grip-Kreis: Längs- und Querkraft teilen sich GRIP_MAX. Was für
    # Gas/Bremse draufgeht, fehlt fürs Kurvenfahren.
    if GRIP_ENABLED:
        a_long = max(-GRIP_MAX, min(GRIP_MAX, a_long))
        a_lat_available = math.sqrt(max(0.0, GRIP_MAX ** 2 - a_long ** 2))
    else:
        a_lat_available = float("inf")

    # Tempo fortschreiben.
    if throttle > 0:
        env["speed_ms"] += a_long * DT
    else:
        env["speed_ms"] = max(0.0, env["speed_ms"] + a_long * DT)
        env["speed_ms"] *= FRICTION
    env["speed_ms"] = max(MIN_DRIVE_SPEED_MS, min(env["speed_ms"], MAX_SPEED_MS))

    # Lenken mit Grip-Limit: die gewünschte Drehrate ω = Lenkung·STEER_GAIN·v
    # braucht die Querbeschleunigung a_lat = ω·v. Reicht der Grip nicht, wird ω
    # gekappt → Untersteuern (das Auto schiebt geradeaus), bis es langsamer ist.
    yaw_rate = steering * STEER_GAIN * env["speed_ms"]
    a_lat_requested = abs(yaw_rate) * env["speed_ms"]
    if a_lat_requested > a_lat_available and env["speed_ms"] > 0.1:
        yaw_rate_max = a_lat_available / env["speed_ms"]
        yaw_rate = max(-yaw_rate_max, min(yaw_rate_max, yaw_rate))
        env["understeer"] = True
    else:
        env["understeer"] = False
    # Die tatsächlich angewendeten Kräfte merken, damit das UI sie zeigen kann.
    env["a_long"] = a_long
    env["a_lat"] = yaw_rate * env["speed_ms"]   # vorzeichenbehaftet: + = Linkskurve

    # Position fortschreiben.
    env["heading"] += yaw_rate * DT
    env["x_m"] += math.cos(env["heading"]) * env["speed_ms"] * DT
    env["y_m"] += math.sin(env["heading"]) * env["speed_ms"] * DT
    env["step_count"] += 1

    _update_track_position(env)

    # Streckenbegrenzung: das Auto ist ein Rechteck (CAR_LENGTH × CAR_WIDTH),
    # seine Ecken erreichen die Wand also vor dem Mittelpunkt — und zwar umso
    # früher, je querer es zur Strecke steht. half_extent ist die Auto-Silhouette
    # projiziert auf die Quer-Achse. Crash, sobald die die Wand berührt.
    yaw_vs_track = (env["heading"] - env["track_heading"] + math.pi) % (2 * math.pi) - math.pi
    half_extent = ((CAR_WIDTH_M / 2) * abs(math.cos(yaw_vs_track))
                   + (CAR_LENGTH_M / 2) * abs(math.sin(yaw_vs_track)))
    if abs(env["lateral_offset_m"]) + half_extent > env["track"]["half_width_m"]:
        _accumulate_reward(env, {"crash": -500.0})
        return _get_obs(env), -500.0, True, False, {}

    total = env["track"]["total_length_m"]

    # Rundenzählung. Mehrrunden-Episode (Curriculum Variante B): jede VORWÄRTS-
    # Überquerung der Ziellinie schreibt LAP_BONUS gut. Die Episode endet erst
    # nach LAPS_PER_EPISODE Runden (oder bei einem Crash). lap_forward wird nur
    # bei echter Vorwärts-Überquerung gesetzt — Rückwärtsrollen zählt nie.
    lap_bonus_this_step = 0.0
    if env["lap_forward"]:
        env["lap_count"] += 1
        env["prev_progress"] = env["progress"]
        env["checkpoint_idx"] = 0
        if env["lap_count"] >= LAPS_PER_EPISODE:
            _accumulate_reward(env, {"lap": LAP_BONUS})
            return _get_obs(env), LAP_BONUS, True, False, {"lap_complete": True}
        # Noch nicht fertig: Bonus in diesen Schritt buchen und weiterfahren.
        lap_bonus_this_step = LAP_BONUS

    # Checkpoint-Tore: fester Bonus + Tempo-Bonus für jedes in diesem Schritt
    # überschrittene Tor.
    checkpoint_reward = 0.0
    next_gate_at_m = (env["checkpoint_idx"] + 1) / N_CHECKPOINTS * total
    while env["checkpoint_idx"] < N_CHECKPOINTS - 1 and env["progress"] >= next_gate_at_m:
        checkpoint_reward += (CHECKPOINT_BONUS
                              + (env["speed_ms"] / MAX_SPEED_MS) * CHECKPOINT_SPEED_BONUS)
        env["checkpoint_idx"] += 1
        next_gate_at_m = (env["checkpoint_idx"] + 1) / N_CHECKPOINTS * total

    # Reward-Prioritäten: 1) Distanz  2) Zeit  3) Tempo.
    # Wir nutzen das naht-korrigierte progress_delta (nicht progress−prev_progress),
    # damit die Start/Ziel-Naht nie als Riesensprung nach vorn zählt.
    progress_gain = env["progress_delta"]
    speed_norm = env["speed_ms"] / MAX_SPEED_MS
    if env["speed_ms"] < MIN_SPEED_MS:
        slow_penalty = SLOW_PENALTY * (1.0 - (env["speed_ms"] - MIN_DRIVE_SPEED_MS)
                                       / (MIN_SPEED_MS - MIN_DRIVE_SPEED_MS))
    else:
        slow_penalty = 0.0
    heading_error = (env["heading"] - env["track_heading"] + math.pi) % (2 * math.pi) - math.pi
    cos_heading = math.cos(heading_error)
    alignment_bonus = max(0.0, cos_heading - 0.5) * 0.3
    # Falschfahrer: zeigt das Auto mehr als 90° von der Streckenrichtung weg
    # (cos < 0), fährt es rückwärts — Strafe wächst, je weiter es falsch zeigt.
    wrong_way_penalty = WRONG_WAY_PENALTY * min(0.0, cos_heading)
    # Vorwärts-Fortschritt wird mit 25× belohnt, Rückwärtsfahren mit 75× bestraft.
    if progress_gain >= 0:
        dist_reward = progress_gain * DIST_REWARD_FWD
    else:
        dist_reward = progress_gain * DIST_REWARD_BACK
    # Curriculum-Schalter: die Optimierungs-Rewards (Tempo / Sanftheit / Zeit)
    # zählen erst, wenn das Auto wirklich eine Runde geschafft hat. Bis dahin
    # wird es rein nach "rumkommen" benotet (Distanz, Checkpoints, Runde, auf
    # der Strecke bleiben).
    refine = 1.0 if env["lap_count"] >= CURRICULUM_AFTER_LAP else 0.0
    reward = _accumulate_reward(env, {
        "distance":   dist_reward,
        "checkpoint": checkpoint_reward,
        "speed":      speed_norm * SPEED_REWARD * refine,
        "align":      alignment_bonus,
        "wrongway":   wrong_way_penalty,        # Strafe fürs Falschherum-Fahren
        "slow":       slow_penalty * refine,
        "steer":      steer_penalty * refine,   # Zappel-Strafe gegen Lenk-Flattern
        "step":       STEP_COST * refine,       # konstante Schrittkosten (Zeitdruck)
        "lap":        lap_bonus_this_step,      # gutgeschrieben bei Zwischenrunden
    })
    env["prev_progress"] = env["progress"]

    return _get_obs(env), reward, False, False, {}


def cast_rays(env: CarEnv) -> list[float]:
    """Echter Raycast: Abstand vom Auto zur nächsten Streckenwand, pro Strahl.

    Jeder Strahl wird mit den echten Wand-Segmenten (Außenrand + Infield)
    geschnitten. Ergebnis: ein Wert pro RAY_ANGLES-Eintrag in [0, 1]:
    0 = Wand direkt hier, 1 = keine Wand innerhalb von MAX_RAY_M.

    Die Rechnung läuft vektorisiert über alle (Strahl, Segment)-Paare auf
    einmal — eine Python-Schleife über tausende Segmente × 40 Autos × 60 fps
    wäre viel zu langsam. Die Shape-Kommentare zeigen, welche Form jedes Array
    hat: R = Anzahl Strahlen (7), K = Anzahl Wand-Segmente.
    """
    n_rays = len(RAY_ANGLES)
    segment_start, segment_end = env["wall_start"], env["wall_end"]
    if len(segment_start) == 0:
        return [1.0] * n_rays

    origin_x, origin_y = env["x_m"], env["y_m"]

    # Vorab-Aussieben (broad phase): Segmente verwerfen, deren Mittelpunkt
    # weiter weg ist, als irgendein Strahl reichen kann.
    is_near = (np.hypot(env["wall_mid"][:, 0] - origin_x,
                        env["wall_mid"][:, 1] - origin_y) <= env["wall_cull_radius"])
    segment_start = segment_start[is_near]
    segment_end = segment_end[is_near]
    if len(segment_start) == 0:
        return [1.0] * n_rays

    # Segment-Richtungsvektoren und Versatz vom Strahl-Ursprung zum Segment-Anfang.
    segment_dx = segment_end[:, 0] - segment_start[:, 0]      # (K,)
    segment_dy = segment_end[:, 1] - segment_start[:, 1]      # (K,)
    to_segment_x = segment_start[:, 0] - origin_x             # (K,)
    to_segment_y = segment_start[:, 1] - origin_y             # (K,)

    # Strahl-Richtungen (Einheitsvektoren) in Weltkoordinaten.
    ray_angles = env["heading"] + _RAY_ANGLES_ARR
    ray_dx = np.cos(ray_angles)[:, None]                      # (R, 1)
    ray_dy = np.sin(ray_angles)[:, None]                      # (R, 1)

    # Klassischer Linien-Schnitt: löse  Ursprung + t·Strahl = Segmentanfang + u·Segment
    # für jedes (Strahl, Segment)-Paar gleichzeitig über 2D-Kreuzprodukte.
    #   t = Distanz entlang des Strahls (gesucht: das Minimum über alle Treffer)
    #   u = Position entlang des Segments (Treffer nur, wenn 0 ≤ u ≤ 1)
    denominator = segment_dx[None, :] * ray_dy - segment_dy[None, :] * ray_dx   # (R, K)
    cross_segment_offset = segment_dx * to_segment_y - segment_dy * to_segment_x  # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        # denominator ≈ 0 heißt: Strahl und Segment sind parallel → t wird inf/nan,
        # solche Paare werden unten über die hit-Maske aussortiert.
        dist_along_ray = cross_segment_offset[None, :] / denominator             # (R, K)
        frac_along_segment = (ray_dx * to_segment_y[None, :]
                              - ray_dy * to_segment_x[None, :]) / denominator    # (R, K)
    hit = ((np.abs(denominator) > 1e-12)
           & (frac_along_segment >= 0.0) & (frac_along_segment <= 1.0)
           & (dist_along_ray >= 0.0))

    # Pro Strahl den nächstgelegenen Treffer nehmen; kein Treffer = volle Reichweite.
    dist_along_ray = np.where(hit, dist_along_ray, np.inf)
    nearest_hit = np.clip(np.min(dist_along_ray, axis=1), 0.0, MAX_RAY_M)        # (R,)
    return (nearest_hit / MAX_RAY_M).tolist()


def _get_obs(env: CarEnv) -> np.ndarray:
    """Baut den 14-Werte-Beobachtungsvektor, alles normiert auf [-1, 1] bzw. [0, 1]."""
    center_x, center_y = env["track"]["bounds_center"]
    half_diag = env["track"]["half_diag_m"]
    total = env["track"]["total_length_m"]
    half_width = env["track"]["half_width_m"]
    # Abweichung von der Streckenrichtung: 0 = parallel, ±1 = genau falsch herum.
    heading_error = (env["heading"] - env["track_heading"] + math.pi) % (2 * math.pi) - math.pi
    # Abstände werden von der AUTOKANTE gemessen (Silhouette auf die Quer-Achse
    # projiziert), damit die Sensoren zum rechteckigen Kollisionsmodell passen.
    half_extent = ((CAR_WIDTH_M / 2) * abs(math.cos(heading_error))
                   + (CAR_LENGTH_M / 2) * abs(math.sin(heading_error)))
    dist_left = max(0.0, min(1.0, (half_width + env["lateral_offset_m"] - half_extent) / half_width))
    dist_right = max(0.0, min(1.0, (half_width - env["lateral_offset_m"] - half_extent) / half_width))
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
    """Baut aus dem Auto-Zustand das Frame-Dict, das ans Web-UI gestreamt wird."""
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
