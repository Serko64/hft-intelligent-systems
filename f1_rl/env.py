"""F1 gymnasium environment with kinematic vehicle physics."""
from __future__ import annotations

import math
import time

import gymnasium as gym
import numpy as np
from shapely.geometry import Point

from .track import TrackData, draw_track, meters_to_pixels

MAX_SPEED_MS = 80.0      # ~288 km/h
STEER_GAIN = 0.015       # rad / (action * speed * dt)
THROTTLE_GAIN = 8.0      # m/s² per unit throttle
FRICTION = 0.98          # speed decay per step when coasting
DT = 1.0 / 60.0
HALF_WIDTH_M = 8.0

# Raycast angles relative to car heading (negative = right, positive = left)
RAY_ANGLES = tuple(math.radians(a) for a in (-60, -30, 0, 30, 60))
MAX_RAY_M = 80.0         # normalisation range for raycast distances


class F1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, track: TrackData, render_mode: str | None = None):
        super().__init__()
        self.track = track
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=np.float32([-1.0] * 12),
            high=np.float32([1.0] * 12),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.float32([-1.0, -1.0]),
            high=np.float32([1.0, 1.0]),
            dtype=np.float32,
        )

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

        self._lateral = 0.0        # cached lateral offset from centerline (m)
        self._track_hdg = 0.0      # cached local track heading (rad)

        self._surface = None
        self._screen = None
        self._clock = None

    # ------------------------------------------------------------------
    def _update_track_position(self) -> None:
        """Project car onto centerline once per step; cache progress + lateral."""
        proj = self.track.centerline_m.project(Point(self.x_m, self.y_m))
        self.progress = proj
        total = self.track.total_length_m
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
        self.speed_ms = 0.0
        self.progress = 0.0
        self.prev_progress = 0.0
        self.lap_count = 0
        self.step_count = 0
        self.lap_start_time = time.time()
        self._last_throttle = 0.0
        self._lateral = 0.0
        self._track_hdg = 0.0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        steering = float(action[0])
        throttle = float(action[1])
        self._last_throttle = throttle

        if throttle > 0:
            self.speed_ms += throttle * THROTTLE_GAIN * DT
        else:
            self.speed_ms = max(0.0, self.speed_ms + throttle * THROTTLE_GAIN * DT)
            self.speed_ms *= FRICTION
        self.speed_ms = min(self.speed_ms, MAX_SPEED_MS)

        self.heading += steering * STEER_GAIN * self.speed_ms * DT
        self.x_m += math.cos(self.heading) * self.speed_ms * DT
        self.y_m += math.sin(self.heading) * self.speed_ms * DT
        self.step_count += 1

        # One project+interpolate pass — replaces contains() + project() + 3×interpolate()
        self._update_track_position()

        # OOB: lateral distance exceeds track half-width
        if abs(self._lateral) > self.track.half_width_m:
            if self.render_mode is not None:
                self.render()
            return self._get_obs(), -100.0, True, False, {}

        total = self.track.total_length_m

        # Lap completion
        if self.prev_progress > total * 0.95 and self.progress < total * 0.05:
            self.lap_count += 1
            self.prev_progress = self.progress
            if self.render_mode is not None:
                self.render()
            return self._get_obs(), 200.0, False, False, {"lap_complete": True}

        progress_gain = self.progress - self.prev_progress
        speed_norm = self.speed_ms / MAX_SPEED_MS          # [0, 1]
        time_penalty = -0.5 + speed_norm * 0.3             # -0.5 at rest, -0.2 at top speed
        reward = progress_gain * 10.0 + time_penalty
        self.prev_progress = self.progress

        if self.render_mode is not None:
            self.render()

        return self._get_obs(), reward, False, False, {}

    # ------------------------------------------------------------------
    def _cast_rays(self) -> list[float]:
        """Return 5 normalised wall-distance readings using local track geometry.

        Uses the cached _lateral and _track_hdg so no extra shapely calls are needed.
        Each value is in [0, 1]: 0 = wall right here, 1 = wall ≥ MAX_RAY_M away.
        Sign convention: _lateral > 0 → car is to the left of the centreline.
        """
        hw = self.track.half_width_m
        rel_hdg = self.heading - self._track_hdg
        out = []
        for theta in RAY_ANGLES:
            # Component of the ray in the perpendicular-to-track direction
            lat_rate = math.sin(rel_hdg + theta)
            if lat_rate > 1e-3:
                # Ray sweeps toward the left wall (lateral = +hw)
                d = (hw - self._lateral) / lat_rate
            elif lat_rate < -1e-3:
                # Ray sweeps toward the right wall (lateral = -hw)
                d = (hw + self._lateral) / (-lat_rate)
            else:
                # Ray is (nearly) parallel to both walls
                d = MAX_RAY_M
            out.append(max(0.0, min(1.0, d / MAX_RAY_M)))
        return out

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        cx, cy = self.track.bounds_center
        hd = self.track.half_diag_m
        total = self.track.total_length_m
        hw = self.track.half_width_m
        dist_left  = max(0.0, min(1.0, (hw + self._lateral) / hw))
        dist_right = max(0.0, min(1.0, (hw - self._lateral) / hw))
        return np.array(
            [
                max(-1.0, min(1.0, (self.x_m - cx) / hd)),
                max(-1.0, min(1.0, (self.y_m - cy) / hd)),
                max(-1.0, min(1.0, self.heading / math.pi)),
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
