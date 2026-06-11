"""Fonts and colour helpers shared across all views."""
from __future__ import annotations

import math

import pygame

from f1_rl.config import GREEN, GOLD


# ── Fonts ───────────────────────────────────────────────────────────────────

def make_fonts() -> tuple:
    """Return the four shared fonts: (title, large, medium, small)."""
    def _f(name, size, bold=False):
        return pygame.font.SysFont(name, size, bold=bold)
    return (
        _f("segoeui", 48, bold=True),  # title
        _f("segoeui", 28, bold=True),  # large
        _f("segoeui", 18),             # medium
        _f("segoeui", 14),             # small
    )


# ── Shapes ──────────────────────────────────────────────────────────────────

def draw_star(screen, cx: int, cy: int, r: int, color: tuple,
              border: tuple = (255, 255, 255)) -> None:
    """5-pointed star centred at (cx, cy) with outer radius r."""
    pts = []
    inner_r = r * 0.42
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        ri = r if i % 2 == 0 else inner_r
        pts.append((cx + ri * math.cos(angle), cy - ri * math.sin(angle)))
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, border, pts, 1)


# ── Rank colour/size for the multi-car training view ──────────────────────────

def rank_color_size(frac: float) -> tuple[tuple[int, int, int], int]:
    """frac=0 → best (gold), frac=1 → worst (dark red)."""
    if frac < 0.08:  return (255, 215, 0),   8   # gold  — top ~3
    if frac < 0.20:  return (50, 220, 80),   6   # green — next ~5
    if frac < 0.50:  return (200, 130, 50),  5   # orange — middle
    return (110, 40, 40), 3                       # dark red — bottom half


# ── Pack colours for swarm mode ───────────────────────────────────────────────

_PACK_PALETTE = [
    (0, 170, 255), (255, 120, 0), (180, 90, 255), (0, 210, 120),
    (255, 60, 140), (230, 210, 0), (120, 200, 255), (255, 160, 90),
]


def pack_color(pack_id: int) -> tuple[int, int, int]:
    return _PACK_PALETTE[int(pack_id) % len(_PACK_PALETTE)]
