"""F1 gymnasium environment with kinematic vehicle physics.

This is a standard ``gymnasium.Env``:
  - reset()  -> (observation, info)
  - step(a)  -> (observation, reward, terminated, truncated, info)

Observation: 14 floats in [-1, 1] (position, heading error, speed, progress,
left/right distance, and 7 lidar-style rays). Action: one of 20 discrete
(steering, throttle) combinations. See config.py for the index layout.
"""
from __future__ import annotations

import math
import time
from typing import NamedTuple

import gymnasium as gym
import numpy as np
from shapely.geometry import Point

from f1_rl.config import CURRICULUM_AFTER_LAP, LAP_BONUS, LAPS_PER_EPISODE
from f1_rl.simulation.track_loader import TrackData

MAX_SPEED_MS = 80.0      # ~288 km/h
STEER_GAIN = 0.15      # rad / (action * speed * dt)
THROTTLE_GAIN = 8.0      # m/s² per unit throttle
FRICTION = 0.98          # speed decay per step when coasting
DT = 1.0 / 60.0

# ── Vehicle dynamics: grip circle ─────────────────────────────────
# When enabled, lateral (cornering) and longitudinal (accel/brake)
# acceleration share one friction budget GRIP_MAX. Cornering harder than
# grip allows causes understeer (the car pushes wide) instead of magically
# rotating — which forces braking before corners and creates a racing line.
GRIP_ENABLED = True      # False = old arcade model (unlimited cornering)
GRIP_MAX     = 45.0      # max combined acceleration, m/s²  (~4.6 g, F1-typical)

# Physical car footprint (metres). The car is treated as a rectangle for
# collisions, so its body — not just its centre point — must fit within the
# track. Roughly F1-sized.
CAR_LENGTH_M = 5.0
CAR_WIDTH_M = 2.0

DISCRETE_ACTIONS = [
    (s, t)
    for s in (-1.0, -0.5, 0.0, 0.5, 1.0)
    for t in (-1.0, 0.0, 0.5, 1.0)
]
N_ACTIONS = len(DISCRETE_ACTIONS)          # 20
DEFAULT_SPEED_MS  = 40.0 / 3.6            # 40 km/h starting speed
MIN_DRIVE_SPEED_MS = 5.0 / 3.6           # hard floor — agents can never fully stop
MIN_SPEED_MS = 20.0 / 3.6               # slow-speed penalty threshold
SLOW_PENALTY = -15.0                      # penalty magnitude at MIN_DRIVE_SPEED
N_CHECKPOINTS = 40                        # reward gates evenly spaced around the lap
# Penalty per unit of steering CHANGE between consecutive steps. Punishes rapid
# left/right flip-flopping (the visible jitter) while still allowing sustained
# cornering, since a held steering angle has zero change.
STEER_CHANGE_PENALTY = 2

# Reward weights — tune these to shift the balance. Distance already dominates
# speed by ~250× (DIST_REWARD_FWD vs SPEED_REWARD); raise the gap further to make
# the agent prioritise completing the lap over raw pace, or vice versa.
DIST_REWARD_FWD        = 25.0   # reward per metre of forward progress
DIST_REWARD_BACK       = 75.0   # penalty per metre driven backwards (> forward)
SPEED_REWARD           = 0.3    # bonus for raw speed (fights "slow but far" crawling)
CHECKPOINT_BONUS       = 10.0   # flat bonus per checkpoint gate crossed
CHECKPOINT_SPEED_BONUS = 5.0    # extra per gate, scaled by speed
STEP_COST              = -0.15  # constant per-step cost = time pressure (punishes dawdling)
WRONG_WAY_PENALTY      = 8.0    # per-step penalty when the car points the wrong way

# Reward is the sum of these components each step. They are tracked separately
# (cumulative per episode) so the UI can show how a car's score is composed.
REWARD_PARTS = ("distance", "checkpoint", "speed", "align", "wrongway", "slow",
                "steer", "step", "lap", "crash")


class CarFrame(NamedTuple):
    """One car's state for a single rendered frame.

    A NamedTuple behaves exactly like a normal tuple (so it still travels through
    the render queue and can be indexed), but every field has a *name*. That
    means the rest of the code can write ``frame.score`` instead of the cryptic
    ``frame[10]``. The field order is the wire format the front-end expects.
    """
    x: float                 # position (metres)
    y: float
    heading: float           # radians
    speed: float             # m/s
    throttle: float          # last throttle input, -1..1
    checkpoint: int          # index of the last reward gate passed
    progress: float          # fraction of the lap completed, 0..1
    lap: int                 # completed laps
    rays: tuple              # lidar-style distance readings
    pack: int                # pack/swarm id (for colouring), 0 in classic mode
    score: float             # live cumulative episode reward
    reward_parts: tuple      # per-component reward breakdown (order = REWARD_PARTS)
    generation: int          # which evolution generation this car's policy is from
    a_long: float            # longitudinal acceleration this step, m/s² (+accel / −brake)
    a_lat: float             # lateral (cornering) acceleration this step, m/s²

# Raycast angles relative to car heading (negative = right, positive = left)
# 7 rays: wide peripherals (±75°) for early wall detection, narrow (±20°) for precision
RAY_ANGLES = tuple(math.radians(a) for a in (-75, -45, -20, 0, 20, 45, 75))
_RAY_ANGLES_ARR = np.asarray(RAY_ANGLES, dtype=np.float64)
MAX_RAY_M = 80.0         # normalisation range for raycast distances

# When rays are disabled, the policy sees this constant in the 7 ray slots — a
# fixed value carries no information, so the agent effectively trains "blind"
# (no wall sensors). The observation stays 14-wide so models remain compatible.
_NO_RAYS = (1.0,) * len(RAY_ANGLES)


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

    def __init__(self, track: TrackData, render_mode: str | None = None,
                 use_rays: bool = True):
        super().__init__()
        self.track = track
        self.render_mode = render_mode
        self.use_rays = use_rays   # False -> policy gets constant (blind) ray inputs

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
        self._progress_delta = 0.0   # signed along-track movement last step
        self._lap_forward = False    # crossed start/finish forward last step
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

        # Signed along-track movement since last step, wrap-corrected at the seam.
        # shapely's nearest-point projection flips to a FAR part of the track when
        # the car faces backwards at the start/finish (the nearest centerline point
        # jumps to the end of the lap). Taking the shortest signed move around the
        # loop turns that spurious +total jump into the small BACKWARD move it
        # really is, so progress reflects true direction of travel.
        delta = proj - self.progress
        if delta > total / 2:
            delta -= total
        elif delta < -total / 2:
            delta += total
        # Clamp to a physically possible per-step move (twice max speed × dt) so a
        # projection glitch can never be read as a giant jump.
        max_step = MAX_SPEED_MS * DT * 2
        delta = max(-max_step, min(max_step, delta))

        raw_new = self.progress + delta
        self._progress_delta = delta            # true signed progress this step
        self._lap_forward = raw_new >= total     # crossed the finish going forward
        self.progress = raw_new % total

        eps = 0.5
        p1 = self.track.centerline_m.interpolate(max(0.0, self.progress - eps))
        p2 = self.track.centerline_m.interpolate(min(total, self.progress + eps))
        self._track_hdg = math.atan2(p2.y - p1.y, p2.x - p1.x)
        nearest = self.track.centerline_m.interpolate(self.progress)
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
        self._progress_delta = 0.0   # signed along-track movement last step
        self._lap_forward = False    # crossed start/finish forward last step
        self.lap_count = 0
        self.step_count = 0
        self.lap_start_time = time.time()
        self._last_throttle = 0.0
        self._last_steering = 0.0
        self._understeer = False
        self._a_long = 0.0          # last applied longitudinal accel (m/s²)
        self._a_lat = 0.0           # last applied lateral/cornering accel (m/s²)
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

        # Longitudinal acceleration from throttle/brake.
        a_long = throttle * THROTTLE_GAIN

        # Grip circle: longitudinal and lateral forces share GRIP_MAX. What is
        # spent on accel/braking is no longer available for cornering.
        if GRIP_ENABLED:
            a_long      = max(-GRIP_MAX, min(GRIP_MAX, a_long))
            a_lat_avail = math.sqrt(max(0.0, GRIP_MAX ** 2 - a_long ** 2))
        else:
            a_lat_avail = float("inf")

        # Advance speed (unchanged dynamics, now via clamped a_long).
        if throttle > 0:
            self.speed_ms += a_long * DT
        else:
            self.speed_ms = max(0.0, self.speed_ms + a_long * DT)
            self.speed_ms *= FRICTION
        self.speed_ms = max(MIN_DRIVE_SPEED_MS, min(self.speed_ms, MAX_SPEED_MS))

        # Steering with grip limit: the requested yaw rate ω = steering·STEER_GAIN·v
        # implies a lateral acceleration a_lat = ω·v. If grip can't supply it, ω is
        # capped → understeer (the car pushes wide) until it slows enough.
        yaw_rate  = steering * STEER_GAIN * self.speed_ms
        a_lat_req = abs(yaw_rate) * self.speed_ms
        if a_lat_req > a_lat_avail and self.speed_ms > 0.1:
            yaw_rate_max     = a_lat_avail / self.speed_ms
            yaw_rate         = max(-yaw_rate_max, min(yaw_rate_max, yaw_rate))
            self._understeer = True
        else:
            self._understeer = False
        # Cache the actually-applied forces so the UI can visualise them.
        self._a_long = a_long
        self._a_lat = yaw_rate * self.speed_ms   # signed: + = turning left
        self.heading += yaw_rate * DT
        self.x_m += math.cos(self.heading) * self.speed_ms * DT
        self.y_m += math.sin(self.heading) * self.speed_ms * DT
        self.step_count += 1

        # One project+interpolate pass — replaces contains() + project() + 3×interpolate()
        self._update_track_position()

        # OOB: the car is a rectangle (CAR_LENGTH × CAR_WIDTH), so its corners
        # reach the wall before the centre does — and reach further the more it
        # points across the track. half_extent is the car footprint projected
        # onto the track-normal (lateral) axis. Crash when that touches the wall.
        yaw = (self.heading - self._track_hdg + math.pi) % (2 * math.pi) - math.pi
        half_extent = ((CAR_WIDTH_M / 2) * abs(math.cos(yaw))
                       + (CAR_LENGTH_M / 2) * abs(math.sin(yaw)))
        if abs(self._lateral) + half_extent > self.track.half_width_m:
            self._accumulate_reward({"crash": -500.0})
            if self.render_mode is not None:
                self.render()
            return self._get_obs(), -500.0, True, False, {}

        total = self.track.total_length_m

        # Lap completion. Multi-lap episode (curriculum variant B): each FORWARD
        # crossing of the line banks LAP_BONUS. The episode only ends once
        # LAPS_PER_EPISODE laps are done (or on a crash). _lap_forward is only set
        # for a genuine forward crossing, so reversing across the line never counts.
        lap_bonus_this_step = 0.0
        if self._lap_forward:
            self.lap_count += 1
            self.prev_progress = self.progress
            self._checkpoint_idx = 0
            if self.lap_count >= LAPS_PER_EPISODE:
                self._accumulate_reward({"lap": LAP_BONUS})
                if self.render_mode is not None:
                    self.render()
                return self._get_obs(), LAP_BONUS, True, False, {"lap_complete": True}
            # Not finished: bank the bonus into this step's reward and keep driving.
            lap_bonus_this_step = LAP_BONUS

        # Checkpoint gates: flat bonus + speed bonus for each threshold crossed this step
        checkpoint_reward = 0.0
        cp_thresh = (self._checkpoint_idx + 1) / N_CHECKPOINTS * total
        while self._checkpoint_idx < N_CHECKPOINTS - 1 and self.progress >= cp_thresh:
            checkpoint_reward += CHECKPOINT_BONUS + (self.speed_ms / MAX_SPEED_MS) * CHECKPOINT_SPEED_BONUS
            self._checkpoint_idx += 1
            cp_thresh = (self._checkpoint_idx + 1) / N_CHECKPOINTS * total

        # Reward priorities: 1) distance  2) time  3) speed
        # Use the wrap-corrected signed delta (not progress−prev_progress) so the
        # start/finish seam can never be read as a giant forward jump.
        progress_gain = self._progress_delta
        speed_norm = self.speed_ms / MAX_SPEED_MS
        if self.speed_ms < MIN_SPEED_MS:
            slow_penalty = SLOW_PENALTY * (1.0 - (self.speed_ms - MIN_DRIVE_SPEED_MS)
                                           / (MIN_SPEED_MS - MIN_DRIVE_SPEED_MS))
        else:
            slow_penalty = 0.0
        hdg_diff = (self.heading - self._track_hdg + math.pi) % (2 * math.pi) - math.pi
        cos_hdg = math.cos(hdg_diff)
        alignment_bonus = max(0.0, cos_hdg - 0.5) * 0.3
        # Wrong-way: when the car points more than 90° off the track direction
        # (cos < 0) it is driving backwards — penalise it harder the more it faces
        # the wrong way, on top of the negative-progress penalty below.
        wrong_way_penalty = WRONG_WAY_PENALTY * min(0.0, cos_hdg)
        # Forward progress rewarded at 25×; driving backwards punished at 75×
        dist_reward = (progress_gain * DIST_REWARD_FWD if progress_gain >= 0
                       else progress_gain * DIST_REWARD_BACK)
        # Curriculum gate: the optimisation rewards (pace / smoothness / time) only
        # count once the car has actually completed a lap. Until then it is graded
        # purely on getting round (distance, checkpoints, lap, staying on track).
        refine = 1.0 if self.lap_count >= CURRICULUM_AFTER_LAP else 0.0
        reward = self._accumulate_reward({
            "distance":   dist_reward,
            "checkpoint": checkpoint_reward,
            "speed":      speed_norm * SPEED_REWARD * refine,   # speed bonus
            "align":      alignment_bonus,
            "wrongway":   wrong_way_penalty,  # punish facing/driving backwards
            "slow":       slow_penalty * refine,
            "steer":      steer_penalty * refine,  # jerk penalty — discourage flip-flop
            "step":       STEP_COST * refine,      # constant step cost (time pressure)
            "lap":        lap_bonus_this_step,     # banked when a non-final lap completes
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
        # Track-relative heading deviation: 0 = aligned with track, ±1 = pointing backwards
        hdg_diff = (self.heading - self._track_hdg + math.pi) % (2 * math.pi) - math.pi
        # Clearance reported from the car's edge (its footprint projected onto the
        # lateral axis), so the sensors match the rectangular collision model.
        half_extent = ((CAR_WIDTH_M / 2) * abs(math.cos(hdg_diff))
                       + (CAR_LENGTH_M / 2) * abs(math.sin(hdg_diff)))
        dist_left  = max(0.0, min(1.0, (hw + self._lateral - half_extent) / hw))
        dist_right = max(0.0, min(1.0, (hw - self._lateral - half_extent) / hw))
        rays = self._cast_rays() if self.use_rays else _NO_RAYS
        return np.array(
            [
                max(-1.0, min(1.0, (self.x_m - cx) / hd)),
                max(-1.0, min(1.0, (self.y_m - cy) / hd)),
                hdg_diff / math.pi,
                self.speed_ms / MAX_SPEED_MS,
                self.progress / total if total > 0 else 0.0,
                dist_left,
                dist_right,
                *rays,
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

        from f1_rl.simulation.track_loader import meters_to_pixels
        from f1_rl.simulation.track_render import GRASS_COLOR, draw_track

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
