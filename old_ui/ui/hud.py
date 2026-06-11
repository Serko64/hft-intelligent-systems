"""HUD overlay rendering (speed panel + throttle/brake graph)."""
from __future__ import annotations

import time
from collections import deque

_FONT_CACHE: dict = {}


def _font(name: str, size: int, bold: bool = False):
    import pygame
    key = (name, size, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = pygame.font.SysFont(name, size, bold=bold)
    return _FONT_CACHE[key]


def draw_hud(surface, hud_state: dict, best_lap: float | None = None) -> None:
    """Draw HUD overlay onto surface (top-left corner).

    hud_state keys: speed_kmh, throttle, lap, lap_start_time
    """
    import pygame

    speed_kmh = hud_state.get("speed_kmh", 0.0)
    throttle = hud_state.get("throttle", 0.0)
    lap = hud_state.get("lap", 1)
    lap_start = hud_state.get("lap_start_time", time.time())

    elapsed = time.time() - lap_start
    lap_str = _fmt_time(elapsed)

    font_lg = _font("segoeui", 38, bold=True)
    font_md = _font("segoeui", 24)
    font_sm = _font("segoeui", 20)

    x, y = 14, 14
    panel_h = 190 if best_lap is not None else 158

    panel = pygame.Surface((340, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    surface.blit(panel, (x - 8, y - 8))

    # Speed
    surface.blit(font_lg.render(f"{speed_kmh:6.1f} km/h", True, (255, 255, 255)), (x, y))
    y += 48

    # Status
    if throttle > 0.05:
        label, color = "ACCELERATING", (50, 220, 80)
    elif throttle < -0.05:
        label, color = "BRAKING", (220, 50, 50)
    else:
        label, color = "COASTING", (160, 160, 160)
    surface.blit(font_md.render(label, True, color), (x, y))
    y += 34

    # Lap counter + current time
    surface.blit(font_sm.render(f"Lap {lap}   {lap_str}", True, (200, 200, 200)), (x, y))
    y += 28

    # Best lap
    if best_lap is not None:
        best_str = _fmt_time(best_lap)
        surface.blit(font_sm.render(f"Best  {best_str}", True, (255, 215, 0)), (x, y))


def draw_throttle_graph(surface, history: deque[float], x: int, y: int) -> None:
    """Draw a scrolling throttle/brake bar graph.

    Green bars above centre = acceleration, red bars below = braking.
    Everything is drawn onto a panel Surface first so bars never overflow.
    """
    import pygame

    W, H = 360, 130
    BAR_W = 3
    N = W // BAR_W
    LABEL_H = 26
    PAD = 6

    panel_w = W + PAD * 2
    panel_h = H + LABEL_H + PAD
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))

    # Label
    label = _font("segoeui", 18).render("Gas / Bremse", True, (160, 160, 160))
    panel.blit(label, (PAD, 2))

    # Graph area in panel-local coords
    gx = PAD
    gy = LABEL_H
    half = H // 2
    mid = gy + half

    # Horizontal axis
    pygame.draw.line(panel, (80, 80, 80), (gx, mid), (gx + W, mid), 1)

    vals = list(history)[-N:]
    for i, v in enumerate(vals):
        bx = gx + i * BAR_W
        bar_h = min(int(abs(v) * half), half)   # clamp to half-height
        if bar_h == 0:
            continue
        if v >= 0:
            pygame.draw.rect(panel, (40, 200, 70),  (bx, mid - bar_h, BAR_W - 1, bar_h))
        else:
            pygame.draw.rect(panel, (200, 50, 50), (bx, mid,          BAR_W - 1, bar_h))

    # Border around graph area
    pygame.draw.rect(panel, (60, 60, 60), (gx, gy, W, H), 1)

    surface.blit(panel, (x, y))


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    ms = int((seconds % 1) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"
