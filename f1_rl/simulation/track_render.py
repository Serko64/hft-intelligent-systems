"""Track rendering: bake a Track dict into a pygame surface (asphalt, kerbs,
boundary lines, dashed centerline). The baked surface is cached per track so
it is only drawn once.
"""
from __future__ import annotations

import math

import pygame

from f1_rl.simulation.track_loader import Track

GRASS_COLOR = (18, 38, 18)

_BAKED_TRACK_CACHE: dict[int, object] = {}   # id(track) → pygame.Surface


def _draw_kerbs(surface, pts: list, kerb_px: float, thickness: int) -> None:
    colors = [(215, 40, 40), (235, 235, 235)]
    acc = 0.0
    ci = 0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1:
            continue
        traveled = 0.0
        while traveled < seg:
            step = min(kerb_px - acc, seg - traveled)
            t0 = traveled / seg
            t1 = (traveled + step) / seg
            pygame.draw.line(
                surface, colors[ci % 2],
                (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0)),
                (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1)),
                thickness,
            )
            acc += step
            traveled += step
            if acc >= kerb_px:
                acc = 0.0
                ci += 1


def _bake_track(track: Track) -> object:
    """Render track once at 3× resolution, smoothscale to native — gives free AA."""
    S = 3  # supersampling factor

    big = pygame.Surface((track["canvas_w"] * S, track["canvas_h"] * S))
    big.fill(GRASS_COLOR)

    def sc(pts):
        return [(x * S, y * S) for x, y in pts]

    pts_out = sc([(float(p[0]), float(p[1])) for p in track["corridor_px"]])
    pts_ctr = sc([(float(p[0]), float(p[1])) for p in track["centerline_px"]])
    pts_inn = (
        sc([(float(p[0]), float(p[1])) for p in track["corridor_interior_px"]])
        if track["corridor_interior_px"] is not None else None
    )

    # Shadow
    shadow = [(x + 6 * S, y + 6 * S) for x, y in pts_out]
    pygame.draw.polygon(big, (8, 8, 8), shadow)

    # Asphalt
    pygame.draw.polygon(big, (52, 52, 58), pts_out)

    # Inner grass
    if pts_inn:
        pygame.draw.polygon(big, GRASS_COLOR, pts_inn)

    # Kerbs (outer then inner)
    _draw_kerbs(big, pts_out, kerb_px=18 * S, thickness=6 * S)
    if pts_inn:
        _draw_kerbs(big, pts_inn, kerb_px=18 * S, thickness=6 * S)

    # White boundary lines
    pygame.draw.lines(big, (225, 225, 225), True, pts_out, 3 * S)
    if pts_inn:
        pygame.draw.lines(big, (225, 225, 225), True, pts_inn, 3 * S)

    # Yellow dashed centerline
    DASH = 14 * S
    for i in range(len(pts_ctr) - 1):
        x0, y0 = pts_ctr[i]
        x1, y1 = pts_ctr[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1:
            continue
        steps = max(1, int(seg_len / DASH))
        for s in range(steps):
            if s % 2 == 0:
                t0 = s / steps
                t1 = (s + 1) / steps
                pygame.draw.line(
                    big, (255, 210, 0),
                    (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0)),
                    (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1)),
                    2 * S,
                )

    return pygame.transform.smoothscale(big, (track["canvas_w"], track["canvas_h"]))


def draw_track(surface, track: Track) -> None:
    key = id(track)
    if key not in _BAKED_TRACK_CACHE:
        _BAKED_TRACK_CACHE[key] = _bake_track(track)
    surface.blit(_BAKED_TRACK_CACHE[key], (0, 0))
