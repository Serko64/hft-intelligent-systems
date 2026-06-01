"""Shared rendering helpers: track surface caching, world→screen projection,
and the common drawing bits (car dot, checkpoints, sensor rays) reused by the
training, replay, and drive views.

The cached surfaces (track baked at native + zoomed scale, the car glow) live
here as module-level singletons so every view shares them.
"""
from __future__ import annotations

import math

import pygame

from f1_rl.config import BG, CANVAS_H, CANVAS_W, GOLD, GREEN
from f1_rl.simulation.environment import MAX_RAY_M, N_CHECKPOINTS, RAY_ANGLES
from f1_rl.simulation.track_loader import meters_to_pixels
from f1_rl.simulation.track_render import draw_track


# ── Surface caches ────────────────────────────────────────────────────────────

_track_surf_cache:    dict                  = {}
_checkpoint_px_cache: dict                  = {}
_scaled_track_cache:  dict                  = {"zoom": None, "track_id": None, "surf": None}
_glow_surf:           pygame.Surface | None = None


def _get_checkpoint_positions_px(track) -> list[tuple[float, float]]:
    tid = id(track)
    if tid not in _checkpoint_px_cache:
        total = track.total_length_m
        _checkpoint_px_cache[tid] = [
            meters_to_pixels(track, *track.centerline_m.interpolate(
                (i + 1) / N_CHECKPOINTS * total
            ).coords[0])
            for i in range(N_CHECKPOINTS)
        ]
    return _checkpoint_px_cache[tid]


def _get_track_surface(track) -> pygame.Surface:
    tid = id(track)
    if tid not in _track_surf_cache:
        surf = pygame.Surface((track.canvas_w, track.canvas_h))
        surf.fill(BG)
        draw_track(surf, track)
        _track_surf_cache[tid] = surf
    return _track_surf_cache[tid]


def _get_glow_surf() -> pygame.Surface:
    global _glow_surf
    if _glow_surf is None:
        _glow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(_glow_surf, (255, 80, 80, 60), (15, 15), 15)
    return _glow_surf


def _get_zoomed_track_surface(track, zoom: float) -> pygame.Surface:
    key_zoom = round(zoom, 2)
    key_tid  = id(track)
    if _scaled_track_cache["zoom"] != key_zoom or _scaled_track_cache["track_id"] != key_tid:
        base = _get_track_surface(track)
        surf = base if abs(key_zoom - 1.0) < 0.01 else pygame.transform.scale(
            base, (int(base.get_width() * key_zoom), int(base.get_height() * key_zoom))
        )
        _scaled_track_cache.update(zoom=key_zoom, track_id=key_tid, surf=surf)
    return _scaled_track_cache["surf"]


# ── Projection + shared primitives ─────────────────────────────────────────────

def _world_to_screen(px: float, py: float, car_px: float, car_py: float,
                     zoom: float, pan_x: int = 0, pan_y: int = 0) -> tuple[int, int]:
    """Convert world-pixel coords to screen coords.

    At zoom=1 the entire track is visible so there is no centering — pan is ignored.
    At zoom>1 the camera is centred on the car, then shifted by (pan_x, pan_y) screen pixels.
    """
    if abs(zoom - 1.0) < 0.01:
        return int(px), int(py)
    return (
        int((px - car_px) * zoom + CANVAS_W / 2 + pan_x),
        int((py - car_py) * zoom + CANVAS_H / 2 + pan_y),
    )


def _draw_checkpoints_overlay(screen, track, car_px: float, car_py: float,
                               zoom: float, checkpoint_idx: int,
                               pan_x: int = 0, pan_y: int = 0) -> None:
    for i, (px, py) in enumerate(_get_checkpoint_positions_px(track)):
        sx, sy = _world_to_screen(px, py, car_px, car_py, zoom, pan_x, pan_y)
        if -20 <= sx <= CANVAS_W + 20 and -20 <= sy <= CANVAS_H + 20:
            if i < checkpoint_idx:
                color, r = GREEN, 5
            elif i == checkpoint_idx:
                color, r = GOLD, 7
            else:
                color, r = (80, 80, 100), 4
            pygame.draw.circle(screen, color, (sx, sy), r)
            if i == checkpoint_idx:
                pygame.draw.circle(screen, (255, 255, 255), (sx, sy), r, 1)


def _draw_rays(screen, track, x_m: float, y_m: float, heading: float,
               ray_norms, car_px: float, car_py: float,
               zoom: float, pan_x: int = 0, pan_y: int = 0) -> None:
    """Draw the car's sensor rays (lidar) from its position out to each wall hit.

    ray_norms: per-ray distances in [0, 1] (0 = wall here, 1 = ≥ MAX_RAY_M away),
    in the same order as RAY_ANGLES.  Rays fade red (near a wall) → green (clear).
    """
    sx, sy = _world_to_screen(car_px, car_py, car_px, car_py, zoom, pan_x, pan_y)
    for theta, norm in zip(RAY_ANGLES, ray_norms):
        d_m = norm * MAX_RAY_M
        ex_m = x_m + math.cos(heading + theta) * d_m
        ey_m = y_m + math.sin(heading + theta) * d_m
        epx, epy = meters_to_pixels(track, ex_m, ey_m)
        ex, ey = _world_to_screen(epx, epy, car_px, car_py, zoom, pan_x, pan_y)
        color = (int(255 * (1.0 - norm)), int(220 * norm), 60)
        pygame.draw.line(screen, color, (sx, sy), (ex, ey), 1)
        pygame.draw.circle(screen, color, (ex, ey), 3)


def _draw_scene(screen, track, car_px: float, car_py: float,
                zoom: float, car_heading: float,
                pan_x: int = 0, pan_y: int = 0,
                dot_color: tuple = (255, 50, 50)) -> None:
    """Blit track + draw the primary (best/selected) car dot."""
    if abs(zoom - 1.0) < 0.01:
        screen.blit(_get_track_surface(track), (0, 0))
        sx, sy = int(car_px), int(car_py)
    else:
        scaled = _get_zoomed_track_surface(track, zoom)
        screen.blit(scaled, (
            -(int(car_px * zoom - CANVAS_W / 2)) + pan_x,
            -(int(car_py * zoom - CANVAS_H / 2)) + pan_y,
        ))
        sx = int(CANVAS_W // 2 + pan_x)
        sy = int(CANVAS_H // 2 + pan_y)

    screen.blit(_get_glow_surf(), (sx - 15, sy - 15))
    pygame.draw.circle(screen, dot_color, (sx, sy), 9)
    pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 9, 2)
    pygame.draw.line(screen, (255, 220, 0),
                     (sx, sy),
                     (int(sx + math.cos(car_heading) * 14),
                      int(sy - math.sin(car_heading) * 14)), 2)
