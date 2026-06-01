"""F1 gymnasium environment with kinematic vehicle physics."""
from __future__ import annotations

import math
import time

import gymnasium as gym
import numpy as np
from shapely.geometry import Point

from .config import LAP_BONUS
from .track import TrackData, draw_track, meters_to_pixels

MAX_SPEED_MS = 80.0      # ~288 km/h
STEER_GAIN = 0.15      # rad / (action * speed * dt)
THROTTLE_GAIN = 8.0      # m/s² per unit throttle
FRICTION = 0.98          # speed decay per step when coasting
DT = 1.0 / 60.0
HALF_WIDTH_M = 8.0

DISCRETE_ACTIONS = [
    (s, t)
    for s in (-1.0, -0.5, 0.0, 0.5, 1.0)
    for t in (-1.0, 0.0, 0.5, 1.0)
]
N_ACTIONS = len(DISCRETE_ACTIONS)          # 20
DEFAULT_SPEED_MS  = 40.0 / 3.6            # 40 km/h starting speed
MIN_DRIVE_SPEED_MS = 5.0 / 3.6           # hard floor — agents can never fully stop
MIN_SPEED_MS = 20.0 / 3.6               # slow-speed penalty threshold
SLOW_PENALTY = -2.0                      # penalty magnitude at MIN_DRIVE_SPEED
N_CHECKPOINTS = 40                        # reward gates evenly spaced around the lap
# Penalty per unit of steering CHANGE between consecutive steps. Punishes rapid
# left/right flip-flopping (the visible jitter) while still allowing sustained
# cornering, since a held steering angle has zero change.
STEER_CHANGE_PENALTY = 2

# Reward is the sum of these components each step. They are tracked separately
# (cumulative per episode) so the UI can show how a car's score is composed.
REWARD_PARTS = ("distance", "checkpoint", "speed", "align", "slow", "steer", "step",
                "lap", "crash")

# Raycast angles relative to car heading (negative = right, positive = left)
# 7 rays: wide peripherals (±75°) for early wall detection, narrow (±20°) for precision
RAY_ANGLES = tuple(math.radians(a) for a in (-75, -45, -20, 0, 20, 45, 75))
_RAY_ANGLES_ARR = np.asarray(RAY_ANGLES, dtype=np.float64)
MAX_RAY_M = 80.0         # normalisation range for raycast distances


def _corridor_wall_segments(track: TrackData) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the track corridor boundary into line segments (in metres).

    Returns (A, B) arrays of shape (K, 2): segment k runs from A[k] to B[k].
    Includes the exterior ring and every interior (infield) ring, so rays are
    blocked by both the outer wall and the inner edge of the track ribbon.
    """
    corr = track.corridor
    polys = corr.geoms if corr.geom_type == "MultiPolygon" else [corr]
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    for poly in polys:
        for ring in (poly.exterior, *poly.interiors):
            pts = np.asarray(ring.coords, dtype=np.float64)
            if len(pts) >= 2:
                a_parts.append(pts[:-1])
                b_parts.append(pts[1:])
    if not a_parts:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    return np.concatenate(a_parts, axis=0), np.concatenate(b_parts, axis=0)


class F1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, track: TrackData, render_mode: str | None = None):
        super().__init__()
        self.track = track
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=np.float32([-1.0] * 14),
            high=np.float32([1.0] * 14),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self.x_m = 0.0
        self.y_m = 0.0
        self.heading = 0.0
        self.speed_ms = 0.0
        self.progress = 0.0
        self.prev_progress = 0.0
        self.lap_count = 0
        self.step_count = 0
        self.lap_start_time = 0.0
        self._last_throttle = 0.0
        self._last_steering = 0.0

        self._lateral = 0.0        # cached lateral offset from centerline (m)
        self._track_hdg = 0.0      # cached local track heading (rad)
        self._checkpoint_idx = 0   # next checkpoint gate to unlock (0..N_CHECKPOINTS-1)

        self._surface = None
        self._screen = None
        self._clock = None

        # Precompute corridor wall segments + broad-phase culling data for raycasts
        self._wall_a, self._wall_b = _corridor_wall_segments(track)
        self._wall_mid = (self._wall_a + self._wall_b) * 0.5
        seg = self._wall_b - self._wall_a
        seg_half_len = 0.5 * np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.empty(0)
        self._wall_cull_r = MAX_RAY_M + seg_half_len   # max midpoint distance a ray can reach

    # ------------------------------------------------------------------
    def _update_track_position(self) -> None:
        """Project car onto centerline once per step; cache progress + lateral."""
        proj = self.track.centerline_m.project(Point(self.x_m, self.y_m))
        total = self.track.total_length_m

        # shapely's nearest-point projection can flip to the incoming side of a
        # tight hairpin (the entry is geometrically closer than the exit in 2D).
        # Clamp backward jumps to twice the max single-step movement so progress
        # never regresses by more than physics allows.  Exception: a large drop
        # near total→0 is a genuine lap crossing and must not be clamped.
        is_lap_wrap = self.progress > total * 0.9 and proj < total * 0.1
        if not is_lap_wrap and proj < self.progress - MAX_SPEED_MS * DT * 2:
            proj = self.progress - MAX_SPEED_MS * DT * 2

        self.progress = proj
        eps = 0.5
        p1 = self.track.centerline_m.interpolate(max(0.0, proj - eps))
        p2 = self.track.centerline_m.interpolate(min(total, proj + eps))
        self._track_hdg = math.atan2(p2.y - p1.y, p2.x - p1.x)
        nearest = self.track.centerline_m.interpolate(proj)
        perp = self._track_hdg + math.pi / 2
        self._lateral = (
            (self.x_m - nearest.x) * math.cos(perp)
            + (self.y_m - nearest.y) * math.sin(perp)
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        coords = list(self.track.centerline_m.coords)
        self.x_m, self.y_m = coords[0]
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        self.heading = math.atan2(dy, dx)
        self.speed_ms = DEFAULT_SPEED_MS
        self.progress = 0.0
        self.prev_progress = 0.0
        self.lap_count = 0
        self.step_count = 0
        self.lap_start_time = time.time()
        self._last_throttle = 0.0
        self._last_steering = 0.0
        self._lateral = 0.0
        self._track_hdg = 0.0
        self._checkpoint_idx = 0
        self.episode_reward = 0.0
        self.reward_parts = {k: 0.0 for k in REWARD_PARTS}
        self._last_reward_parts = dict(self.reward_parts)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _accumulate_reward(self, parts: dict) -> float:
        """Add this step's reward components to the running totals; return the step sum."""
        total = float(sum(parts.values()))
        for k, v in parts.items():
            self.reward_parts[k] += v
        self.episode_reward += total
        self._last_reward_parts = parts
        return total

    # ------------------------------------------------------------------
    def step(self, action):
        action_idx = int(action)
        steering, throttle = DISCRETE_ACTIONS[action_idx]
        steer_penalty = -STEER_CHANGE_PENALTY * abs(steering - self._last_steering)
        self._last_steering = steering
        self._last_throttle = throttle

        if throttle > 0:
            self.speed_ms += throttle * THROTTLE_GAIN * DT
        else:
            self.speed_ms = max(0.0, self.speed_ms + throttle * THROTTLE_GAIN * DT)
            self.speed_ms *= FRICTION
        self.speed_ms = max(MIN_DRIVE_SPEED_MS, min(self.speed_ms, MAX_SPEED_MS))

        self.heading += steering * STEER_GAIN * self.speed_ms * DT
        self.x_m += math.cos(self.heading) * self.speed_ms * DT
        self.y_m += math.sin(self.heading) * self.speed_ms * DT
        self.step_count += 1

        # One project+interpolate pass — replaces contains() + project() + 3×interpolate()
        self._update_track_position()

        # OOB: lateral distance exceeds track half-width
        if abs(self._lateral) > self.track.half_width_m:
            self._accumulate_reward({"crash": -500.0})
            if self.render_mode is not None:
                self.render()
            return self._get_obs(), -500.0, True, False, {}

        total = self.track.total_length_m

        # Lap completion — single-lap objective: a full lap (back at the start) is the
        # goal, so it pays out a big bonus and TERMINATES the episode (car is reset).
        if self.prev_progress > total * 0.95 and self.progress < total * 0.05:
            self.lap_count += 1
            self.prev_progress = self.progress
            self._checkpoint_idx = 0
            self._accumulate_reward({"lap": LAP_BONUS})
            if self.render_mode is not None:
                self.render()
            return self._get_obs(), LAP_BONUS, True, False, {"lap_complete": True}

        # Checkpoint gates: flat bonus + speed bonus for each threshold crossed this step
        checkpoint_reward = 0.0
        cp_thresh = (self._checkpoint_idx + 1) / N_CHECKPOINTS * total
        while self._checkpoint_idx < N_CHECKPOINTS - 1 and self.progress >= cp_thresh:
            checkpoint_reward += 10.0 + (self.speed_ms / MAX_SPEED_MS) * 5.0
            self._checkpoint_idx += 1
            cp_thresh = (self._checkpoint_idx + 1) / N_CHECKPOINTS * total

        # Reward priorities: 1) distance  2) time  3) speed
        progress_gain = self.progress - self.prev_progress
        speed_norm = self.speed_ms / MAX_SPEED_MS
        if self.speed_ms < MIN_SPEED_MS:
            slow_penalty = SLOW_PENALTY * (1.0 - (self.speed_ms - MIN_DRIVE_SPEED_MS)
                                           / (MIN_SPEED_MS - MIN_DRIVE_SPEED_MS))
        else:
            slow_penalty = 0.0
        hdg_diff = (self.heading - self._track_hdg + math.pi) % (2 * math.pi) - math.pi
        alignment_bonus = max(0.0, math.cos(hdg_diff) - 0.5) * 0.3
        # Forward progress rewarded at 25×; driving backwards punished at 75×
        dist_reward = progress_gain * 25.0 if progress_gain >= 0 else progress_gain * 75.0
        reward = self._accumulate_reward({
            "distance":   dist_reward,
            "checkpoint": checkpoint_reward,
            "speed":      speed_norm * 0.1,   # speed bonus
            "align":      alignment_bonus,
            "slow":       slow_penalty,
            "steer":      steer_penalty,      # jerk penalty — discourage flip-flop steering
            "step":       -0.05,              # constant step cost (time pressure)
        })
        self.prev_progress = self.progress

        if self.render_mode is not None:
            self.render()

        return self._get_obs(), reward, False, False, {}

    # ------------------------------------------------------------------
    def _cast_rays(self) -> list[float]:
        """True raycast: distance from the car to the nearest corridor wall per ray.

        Each ray is intersected against the actual track boundary segments
        (exterior + infield rings). Returns one value per RAY_ANGLES entry in
        [0, 1]: 0 = wall right here, 1 = no wall within MAX_RAY_M.

        Vectorised over rays × segments, with a broad-phase distance cull so only
        segments a ray could plausibly reach take part in the intersection test.
        """
        n_rays = len(RAY_ANGLES)
        a, b = self._wall_a, self._wall_b
        if len(a) == 0:
            return [1.0] * n_rays

        ox, oy = self.x_m, self.y_m

        # Broad phase: drop segments whose midpoint is farther than any ray can reach.
        near = (np.hypot(self._wall_mid[:, 0] - ox,
                         self._wall_mid[:, 1] - oy) <= self._wall_cull_r)
        a = a[near]
        b = b[near]
        if len(a) == 0:
            return [1.0] * n_rays

        # Segment vectors e = b - a, and offset from ray origin w = a - O.
        ex = b[:, 0] - a[:, 0]
        ey = b[:, 1] - a[:, 1]
        wx = a[:, 0] - ox
        wy = a[:, 1] - oy

        # Ray directions (unit vectors) in world space.
        angles = self.heading + _RAY_ANGLES_ARR
        dx = np.cos(angles)[:, None]          # (R, 1)
        dy = np.sin(angles)[:, None]

        # Solve  O + t·d = a + u·e  for each (ray, segment) pair via 2D cross products.
        det = ex[None, :] * dy - ey[None, :] * dx          # (R, K)
        tew = ex * wy - ey * wx                             # (K,) cross(e, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = tew[None, :] / det                          # distance along ray
            u = (dx * wy[None, :] - dy * wx[None, :]) / det  # position along segment
        hit = (np.abs(det) > 1e-12) & (u >= 0.0) & (u <= 1.0) & (t >= 0.0)

        t = np.where(hit, t, np.inf)
        dist = np.clip(np.min(t, axis=1), 0.0, MAX_RAY_M)
        return (dist / MAX_RAY_M).tolist()

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        cx, cy = self.track.bounds_center
        hd = self.track.half_diag_m
        total = self.track.total_length_m
        hw = self.track.half_width_m
        dist_left  = max(0.0, min(1.0, (hw + self._lateral) / hw))
        dist_right = max(0.0, min(1.0, (hw - self._lateral) / hw))
        # Track-relative heading deviation: 0 = aligned with track, ±1 = pointing backwards
        hdg_diff = (self.heading - self._track_hdg + math.pi) % (2 * math.pi) - math.pi
        return np.array(
            [
                max(-1.0, min(1.0, (self.x_m - cx) / hd)),
                max(-1.0, min(1.0, (self.y_m - cy) / hd)),
                hdg_diff / math.pi,
                self.speed_ms / MAX_SPEED_MS,
                self.progress / total if total > 0 else 0.0,
                dist_left,
                dist_right,
                *self._cast_rays(),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def get_hud_state(self) -> dict:
        return {
            "speed_kmh": self.speed_ms * 3.6,
            "throttle": self._last_throttle,
            "lap": self.lap_count + 1,
            "lap_start_time": self.lap_start_time,
        }

    # ------------------------------------------------------------------
    def render(self):
        import pygame

        if self._surface is None:
            if not pygame.get_init():
                pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode(
                    (self.track.canvas_w, self.track.canvas_h)
                )
                pygame.display.set_caption("F1 RL Simulator")
            self._surface = pygame.Surface(
                (self.track.canvas_w, self.track.canvas_h)
            )
            self._clock = pygame.time.Clock()

        from .track import GRASS_COLOR
        self._surface.fill(GRASS_COLOR)
        draw_track(self._surface, self.track)

        px, py = meters_to_pixels(self.track, self.x_m, self.y_m)
        ipx, ipy = int(px), int(py)
        pygame.draw.circle(self._surface, (255, 50, 50), (ipx, ipy), 8)
        hdx = int(px + math.cos(self.heading) * 20)
        hdy = int(py - math.sin(self.heading) * 20)
        pygame.draw.line(self._surface, (255, 220, 0), (ipx, ipy), (hdx, hdy), 3)

        if self.render_mode == "human" and self._screen is not None:
            self._screen.blit(self._surface, (0, 0))
            # do NOT flip here — main.py owns the flip
        elif self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self._surface), (1, 0, 2)
            )

    # ------------------------------------------------------------------
    def close(self):
        if self._surface is not None:
            import pygame
            pygame.quit()
            self._surface = None
            self._screen = None
            self._clock = None
