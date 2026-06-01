"""Pygame rendering: track surfaces, all screen draw functions, and visualization loop."""
from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np
import pygame

from .config import (
    ACCENT, BG, CANVAS_H, CANVAS_W, DIM, FPS, GENS_PRESETS, GOLD, GREEN,
    HOV_BG, QTABLE_PATH, SEL_BG, STEPS_PRESETS, TEXT, CIRCUITS,
)
from f1_rl.env import MAX_RAY_M, N_CHECKPOINTS, RAY_ANGLES, REWARD_PARTS

# Human-readable labels for each reward component (order matches env.REWARD_PARTS)
REWARD_LABELS = {
    "distance":   "Distanz",
    "checkpoint": "Checkpoints",
    "speed":      "Tempo",
    "align":      "Ausrichtung",
    "slow":       "Langsam-Strafe",
    "steer":      "Lenk-Strafe",
    "step":       "Zeitkosten",
    "lap":        "Runden-Bonus",
    "crash":      "Crash",
}


# ── Font helper ───────────────────────────────────────────────────────────────

def make_fonts() -> tuple:
    def _f(name, size, bold=False):
        return pygame.font.SysFont(name, size, bold=bold)
    return (
        _f("segoeui", 48, bold=True),  # title
        _f("segoeui", 28, bold=True),  # large
        _f("segoeui", 18),             # medium
        _f("segoeui", 14),             # small
    )


# ── Surface caches ────────────────────────────────────────────────────────────

_track_surf_cache:    dict                    = {}
_checkpoint_px_cache: dict                    = {}
_scaled_track_cache:  dict                    = {"zoom": None, "track_id": None, "surf": None}
_glow_surf:           pygame.Surface | None   = None
_stats_panel_surf:    pygame.Surface | None   = None


def _get_checkpoint_positions_px(track) -> list[tuple[float, float]]:
    from f1_rl.track import meters_to_pixels
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
        from f1_rl.track import draw_track
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


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_star(screen, cx: int, cy: int, r: int, color: tuple,
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


# ── Rank color/size for multi-car display ─────────────────────────────────────

def _rank_color_size(frac: float) -> tuple[tuple[int, int, int], int]:
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


def _pack_color(pack_id: int) -> tuple[int, int, int]:
    return _PACK_PALETTE[int(pack_id) % len(_PACK_PALETTE)]


# ── Track preview (menu) ──────────────────────────────────────────────────────

_preview_surf_cache: dict = {}   # (name, w, h) → baked pygame.Surface


def _get_preview_surface(track, w: int, h: int) -> pygame.Surface:
    """Bake the track centerline into a (w, h) preview surface (cached per track)."""
    key = (track.name, w, h)
    if key in _preview_surf_cache:
        return _preview_surf_cache[key]

    S = 2  # light supersampling for smoother lines
    coords = np.asarray(track.centerline_m.coords, dtype=np.float64)
    minx, miny = coords.min(axis=0)
    maxx, maxy = coords.max(axis=0)
    span_x = (maxx - minx) or 1.0
    span_y = (maxy - miny) or 1.0
    pad = 16 * S
    bw, bh = w * S, h * S
    scale = min((bw - 2 * pad) / span_x, (bh - 2 * pad) / span_y)
    ox = pad + (bw - 2 * pad - span_x * scale) / 2 - minx * scale
    oy = pad + (bh - 2 * pad - span_y * scale) / 2 + maxy * scale   # Y-flip
    pts = [(ox + x * scale, oy - y * scale) for x, y in coords]

    big = pygame.Surface((bw, bh), pygame.SRCALPHA)
    if len(pts) >= 2:
        pygame.draw.lines(big, (70, 70, 80), False,
                          [(x + 3 * S, y + 3 * S) for x, y in pts], 7 * S)  # shadow
        pygame.draw.lines(big, (190, 190, 205), False, pts, 6 * S)         # track ribbon
        pygame.draw.lines(big, (255, 210, 0), False, pts, 1 * S)           # centerline
        pygame.draw.circle(big, GREEN, (int(pts[0][0]), int(pts[0][1])), 7 * S)  # start
    surf = pygame.transform.smoothscale(big, (w, h))
    _preview_surf_cache[key] = surf
    return surf


def draw_track_preview(screen, fonts, preview, rect: pygame.Rect) -> None:
    """Render a track preview into *rect*.

    *preview* is a TrackData, the string "loading", or None.
    """
    f_title, f_lg, f_md, f_sm = fonts
    pygame.draw.rect(screen, (18, 18, 26), rect, border_radius=8)
    pygame.draw.rect(screen, DIM, rect, 1, border_radius=8)
    screen.blit(f_sm.render("PREVIEW", True, GOLD), (rect.x + 12, rect.y + 8))

    if preview is None or isinstance(preview, str):
        msg = "Lade Vorschau…" if preview == "loading" else "—"
        m = f_md.render(msg, True, DIM)
        screen.blit(m, (rect.centerx - m.get_width() // 2,
                        rect.centery - m.get_height() // 2))
        return

    surf = _get_preview_surface(preview, rect.width - 24, rect.height - 56)
    screen.blit(surf, (rect.x + 12, rect.y + 30))
    name = f_sm.render(f"{preview.name}  ·  {preview.total_length_m:.0f} m", True, TEXT)
    screen.blit(name, (rect.x + 12, rect.bottom - 22))


# ── Shared drawing primitives ─────────────────────────────────────────────────

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
    from f1_rl.track import meters_to_pixels
    import math as _math
    sx, sy = _world_to_screen(car_px, car_py, car_px, car_py, zoom, pan_x, pan_y)
    for theta, norm in zip(RAY_ANGLES, ray_norms):
        d_m = norm * MAX_RAY_M
        ex_m = x_m + _math.cos(heading + theta) * d_m
        ey_m = y_m + _math.sin(heading + theta) * d_m
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


# ── Menu ──────────────────────────────────────────────────────────────────────

def draw_param_row(screen, fonts, rect, label: str, value: int,
                   preset_idx: int, n_presets: int,
                   typing: str | None, active: bool) -> None:
    f_title, f_lg, f_md, f_sm = fonts
    border_color = GOLD if active else DIM
    if typing is not None:
        pygame.draw.rect(screen, (30, 30, 50), rect, border_radius=6)
        pygame.draw.rect(screen, GOLD, rect, 2, border_radius=6)
        cursor = "_" if (int(time.time() * 2) % 2 == 0) else " "
        lbl = f_md.render(f"{label}  {typing}{cursor}", True, GOLD)
    else:
        pygame.draw.rect(screen, HOV_BG, rect, border_radius=6)
        pygame.draw.rect(screen, border_color, rect, 2 if active else 1, border_radius=6)
        can_l = preset_idx > 0
        can_r = preset_idx < n_presets - 1 or preset_idx == -1
        al = f_md.render("◄", True, GOLD if can_l else (50, 50, 60))
        ar = f_md.render("►", True, GOLD if can_r else (50, 50, 60))
        val_color = GOLD if preset_idx == -1 else (TEXT if active else DIM)
        lbl = f_md.render(f"{label}  {value:,}", True, val_color)
        cy  = rect.y + (rect.height - lbl.get_height()) // 2
        screen.blit(al, (rect.x + 10, cy))
        screen.blit(ar, (rect.right - ar.get_width() - 10, cy))
    cy = rect.y + (rect.height - lbl.get_height()) // 2
    screen.blit(lbl, (rect.x + rect.width // 2 - lbl.get_width() // 2, cy))


# Menu list geometry — exported so main.py can do hit-testing + scrolling
MENU_LIST_X    = CANVAS_W // 2 - 470
MENU_LIST_W    = 440
MENU_LIST_TOP  = 175
MENU_ITEM_H    = 40
MENU_VISIBLE   = 9
MENU_PREVIEW   = pygame.Rect(CANVAS_W // 2 + 30, 175, 440, 360)


def menu_param_rects() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    """(steps, generations, mode) row rects, just under the circuit list window."""
    py = MENU_LIST_TOP + MENU_VISIBLE * MENU_ITEM_H + 16
    w  = MENU_LIST_W
    return (
        pygame.Rect(MENU_LIST_X, py,                  w, MENU_ITEM_H - 4),
        pygame.Rect(MENU_LIST_X, py + MENU_ITEM_H,    w, MENU_ITEM_H - 4),
        pygame.Rect(MENU_LIST_X, py + 2 * MENU_ITEM_H, w, MENU_ITEM_H - 4),
    )


def draw_menu(screen, fonts, sel_idx: int, hover_idx: int,
              train_state: dict | None, n_replays: int,
              steps_preset_idx: int, steps_value: int, typing_steps: str | None,
              gens_preset_idx: int, gens_value: int, typing_gens: str | None,
              active_param: str, scroll_offset: int = 0,
              preview=None, evolution_mode: str = "classic") -> None:
    screen.fill(BG)
    f_title, f_lg, f_md, f_sm = fonts

    t = f_title.render("F1 RL SIMULATOR", True, ACCENT)
    screen.blit(t, (CANVAS_W // 2 - t.get_width() // 2, 55))
    sub = f_sm.render("Select circuit → ENTER to train", True, DIM)
    screen.blit(sub, (CANVAS_W // 2 - sub.get_width() // 2, 110))

    # ── Circuit list (scrollable window) ──────────────────────────────────
    n = len(CIRCUITS)
    end = min(n, scroll_offset + MENU_VISIBLE)
    for row, i in enumerate(range(scroll_offset, end)):
        name = CIRCUITS[i][0]
        y    = MENU_LIST_TOP + row * MENU_ITEM_H
        rect = pygame.Rect(MENU_LIST_X, y, MENU_LIST_W, MENU_ITEM_H - 4)
        if i == sel_idx:
            pygame.draw.rect(screen, SEL_BG, rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT, rect, 2, border_radius=6)
        elif i == hover_idx:
            pygame.draw.rect(screen, HOV_BG, rect, border_radius=6)
        label = f_md.render(name, True, TEXT if i == sel_idx else DIM)
        screen.blit(label, (rect.x + 16, rect.y + (rect.height - label.get_height()) // 2))

    # Scroll indicators
    if scroll_offset > 0:
        up = f_sm.render("▲ more", True, DIM)
        screen.blit(up, (MENU_LIST_X + MENU_LIST_W - up.get_width() - 6, MENU_LIST_TOP - 18))
    if end < n:
        dn = f_sm.render("▼ more", True, DIM)
        screen.blit(dn, (MENU_LIST_X + MENU_LIST_W - dn.get_width() - 6,
                         MENU_LIST_TOP + MENU_VISIBLE * MENU_ITEM_H - 4))
    cnt = f_sm.render(f"{sel_idx + 1}/{n}", True, DIM)
    screen.blit(cnt, (MENU_LIST_X, MENU_LIST_TOP - 18))

    # ── Track preview ─────────────────────────────────────────────────────
    draw_track_preview(screen, fonts, preview, MENU_PREVIEW)

    # ── Parameter + mode rows ─────────────────────────────────────────────
    steps_rect, gens_rect, mode_rect = menu_param_rects()
    draw_param_row(screen, fonts, steps_rect, "Steps/gen:", steps_value,
                   steps_preset_idx, len(STEPS_PRESETS), typing_steps,
                   active_param == "steps")
    draw_param_row(screen, fonts, gens_rect, "Generations:", gens_value,
                   gens_preset_idx, len(GENS_PRESETS), typing_gens,
                   active_param == "gens")

    # Mode row (toggle with M)
    is_pack = evolution_mode == "pack"
    pygame.draw.rect(screen, HOV_BG, mode_rect, border_radius=6)
    pygame.draw.rect(screen, GREEN if is_pack else DIM, mode_rect, 2 if is_pack else 1,
                     border_radius=6)
    mode_txt = "Rudel-Evolution (Pack)" if is_pack else "Klassisch (Survival of the fittest)"
    ml = f_md.render(f"Evolution [M]:  {mode_txt}", True, GREEN if is_pack else DIM)
    screen.blit(ml, (mode_rect.x + 16, mode_rect.y + (mode_rect.height - ml.get_height()) // 2))

    if typing_steps is not None or typing_gens is not None:
        hint_t = f_sm.render("ENTER confirm   ESC cancel", True, DIM)
        screen.blit(hint_t, (MENU_LIST_X, mode_rect.bottom + 6))
    elif train_state:
        gen = train_state.get("generation", 0)
        bf  = train_state.get("best_fitness", 0.0)
        info = f_sm.render(
            f"Last training: {train_state.get('circuit', '?')}  —  gen {gen}  best {bf:+.0f}",
            True, GOLD,
        )
        screen.blit(info, (MENU_LIST_X, mode_rect.bottom + 8))

    model_exists = os.path.exists(QTABLE_PATH)
    hints = [
        ("ENTER",  "Train (live view)"),
        ("R",      "Resume training" if train_state else "Resume (no checkpoint)"),
        ("L",      f"Load & drive  {'[model found]' if model_exists else '(no model yet)'}"),
        ("P",      f"Replay browser  ({n_replays} saved)"),
        ("UP/DOWN", "Select circuit"),
        ("M",      "Toggle evolution mode (classic / pack)"),
        ("Tab",    "Switch Steps / Generations row"),
        ("LEFT/RIGHT", "Cycle presets for active row"),
        ("0-9",    "Type custom value for active row"),
        ("ESC",    "Quit"),
    ]
    hx, hy = 60, CANVAS_H - 250
    for key, desc in hints:
        k = f_sm.render(f"[{key}]", True, GOLD)
        d = f_sm.render(f"  {desc}", True, DIM)
        screen.blit(k, (hx, hy))
        screen.blit(d, (hx + k.get_width(), hy))
        hy += 24


# ── Training view ─────────────────────────────────────────────────────────────

# Top-scores panel geometry — exported so main.py can do hit-testing
TOP_PANEL_W         = 230
TOP_PANEL_X0        = CANVAS_W - TOP_PANEL_W - 8
TOP_PANEL_Y0        = 8
TOP_ROW_FIRST_Y     = 26   # offset from panel top to first score row
TOP_ROW_H           = 17


def _draw_reward_breakdown(screen, fonts, car, car_idx: int, top_y: int) -> None:
    """Side panel showing how the followed car's live score is composed."""
    f_title, f_lg, f_md, f_sm = fonts
    score = car[10] if len(car) > 10 else 0.0
    parts = car[11] if len(car) > 11 else tuple(0.0 for _ in REWARD_PARTS)

    pad   = 10
    row_h = 22
    panel_w = TOP_PANEL_W
    panel_h = 56 + len(REWARD_PARTS) * row_h
    x0 = TOP_PANEL_X0
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 185))
    screen.blit(panel, (x0, top_y))

    screen.blit(f_md.render(f"CAR #{car_idx}", True, (120, 220, 255)), (x0 + pad, top_y + 6))
    screen.blit(f_sm.render(f"Score: {score:+.0f}", True, GOLD), (x0 + pad, top_y + 30))

    max_abs = max((abs(v) for v in parts), default=1.0) or 1.0
    bar_x   = x0 + 120
    bar_max = panel_w - (bar_x - x0) - pad
    y = top_y + 52
    for key, val in zip(REWARD_PARTS, parts):
        label = REWARD_LABELS.get(key, key)
        col   = GREEN if val >= 0 else (230, 80, 80)
        screen.blit(f_sm.render(label, True, (200, 200, 210)), (x0 + pad, y))
        # bar grows from a centre baseline at bar_x
        w = int(abs(val) / max_abs * bar_max)
        if val >= 0:
            pygame.draw.rect(screen, col, (bar_x, y + 3, max(1, w), row_h - 9))
        else:
            pygame.draw.rect(screen, col, (bar_x - w, y + 3, max(1, w), row_h - 9))
        screen.blit(f_sm.render(f"{val:+.0f}", True, col),
                    (x0 + panel_w - pad - 42, y))
        y += row_h
    return top_y + panel_h


def draw_training(screen, fonts, track, car_states: list | None, stats: dict,
                  elapsed: float, zoom: float = 1.0,
                  throttle_hist=None, car_trail=None,
                  hover_score_idx: int = -1,
                  best_marker_px: tuple[float, float] | None = None,
                  show_rays: bool = False,
                  swarm_mode: bool = False,
                  focus_idx: int = -1, show_scores: bool = False) -> None:
    """Render all population cars live.

    car_states: list of (x_m,y_m,heading,speed_ms,throttle,cp_idx,progress) tuples,
                sorted best-first.  None or empty = waiting for first generation.
    """
    from f1_rl.track import meters_to_pixels, GRASS_COLOR
    screen.fill(GRASS_COLOR)
    f_title, f_lg, f_md, f_sm = fonts

    best = car_states[0] if car_states else None
    car_px = car_py = 0.0
    speed_ms = checkpoint_idx = progress_m = 0.0
    lap_count = 0

    # Camera follows the selected car when one is clicked, else the best car.
    has_sel = car_states is not None and 0 <= focus_idx < len(car_states)
    focus = car_states[focus_idx] if has_sel else best

    if best is not None:
        x_m, y_m, heading, speed_ms, _, checkpoint_idx, progress_m, *rest = best
        lap_count = rest[0] if rest else 0

    if track is not None:
        if best is not None:
            car_px, car_py = meters_to_pixels(track, focus[0], focus[1])

            # Blit track (without car dots — drawn separately below)
            if abs(zoom - 1.0) < 0.01:
                screen.blit(_get_track_surface(track), (0, 0))
            else:
                scaled = _get_zoomed_track_surface(track, zoom)
                screen.blit(scaled, (
                    -(int(car_px * zoom - CANVAS_W / 2)),
                    -(int(car_py * zoom - CANVAS_H / 2)),
                ))

            # Fading trail for best car
            if car_trail:
                trail_list = list(car_trail)
                n = len(trail_list)
                for i, (tx_m, ty_m) in enumerate(trail_list[:-1]):
                    tsx, tsy = _world_to_screen(
                        *meters_to_pixels(track, tx_m, ty_m), car_px, car_py, zoom
                    )
                    if -6 <= tsx <= CANVAS_W + 6 and -6 <= tsy <= CANVAS_H + 6:
                        frac = i / max(n - 1, 1)
                        intensity = int(60 + 140 * frac)
                        r = max(1, int(1 + 3 * frac))
                        pygame.draw.circle(screen,
                                           (intensity, intensity // 4, intensity // 4),
                                           (tsx, tsy), r)

            # Draw permanent best-position marker (under cars so cars render on top)
            if best_marker_px is not None:
                bm_sx, bm_sy = _world_to_screen(
                    best_marker_px[0], best_marker_px[1], car_px, car_py, zoom
                )
                if -30 <= bm_sx <= CANVAS_W + 30 and -30 <= bm_sy <= CANVAS_H + 30:
                    # White backing circle so the star reads against any track colour
                    pygame.draw.circle(screen, (255, 255, 255), (bm_sx, bm_sy), 14)
                    pygame.draw.circle(screen, (40, 40, 40),    (bm_sx, bm_sy), 14, 1)
                    _draw_star(screen, bm_sx, bm_sy, 11, GOLD, (180, 120, 0))

            # Sensor rays for the best car (drawn under the cars)
            if show_rays and len(rest) > 1 and rest[1]:
                _draw_rays(screen, track, x_m, y_m, heading, rest[1],
                           car_px, car_py, zoom)

            # Draw all cars worst-first so the best car renders on top
            n_cars = len(car_states)
            pulse_r = int(14 + 4 * math.sin(time.time() * 6))
            for rank in range(n_cars - 1, -1, -1):
                ci = car_states[rank]
                ci_px, ci_py = meters_to_pixels(track, ci[0], ci[1])
                ci_sx, ci_sy = _world_to_screen(ci_px, ci_py, car_px, car_py, zoom)
                if not (-20 <= ci_sx <= CANVAS_W + 20 and -20 <= ci_sy <= CANVAS_H + 20):
                    continue
                frac = rank / max(n_cars - 1, 1)
                if swarm_mode and len(ci) > 9:
                    color = _pack_color(ci[9])
                    r = 9 if rank == 0 else 6
                else:
                    color, r = _rank_color_size(frac)
                if rank == 0:
                    # Pulsing outer ring so the best car is always easy to spot
                    pygame.draw.circle(screen, GOLD, (ci_sx, ci_sy), pulse_r, 2)
                    # Full glow + direction arrow
                    screen.blit(_get_glow_surf(), (ci_sx - 15, ci_sy - 15))
                    pygame.draw.circle(screen, color, (ci_sx, ci_sy), 9)
                    pygame.draw.circle(screen, (255, 255, 255), (ci_sx, ci_sy), 9, 2)
                    pygame.draw.line(screen, (255, 220, 0),
                                     (ci_sx, ci_sy),
                                     (int(ci_sx + math.cos(ci[2]) * 14),
                                      int(ci_sy - math.sin(ci[2]) * 14)), 2)
                else:
                    pygame.draw.circle(screen, color, (ci_sx, ci_sy), r)

                # Selected car: bright cyan ring (the one we follow)
                if has_sel and rank == focus_idx:
                    pygame.draw.circle(screen, (80, 220, 255), (ci_sx, ci_sy),
                                       pulse_r + 4, 2)

                # Per-car live score label
                if show_scores and len(ci) > 10:
                    lbl = f_sm.render(f"{ci[10]:+.0f}", True, (235, 235, 245))
                    screen.blit(lbl, (ci_sx + 8, ci_sy - 8))

            _draw_checkpoints_overlay(screen, track, car_px, car_py, zoom,
                                      int(checkpoint_idx))
        else:
            screen.blit(_get_track_surface(track), (0, 0))
    else:
        f_title, f_lg, f_md, f_sm = fonts
        msg = f_lg.render("Loading track...", True, DIM)
        screen.blit(msg, (CANVAS_W // 2 - msg.get_width() // 2, CANVAS_H // 2 - 20))

    f_title, f_lg, f_md, f_sm = fonts

    global _stats_panel_surf
    if _stats_panel_surf is None:
        _stats_panel_surf = pygame.Surface((310, 310), pygame.SRCALPHA)
        _stats_panel_surf.fill((0, 0, 0, 170))
    screen.blit(_stats_panel_surf, (8, 8))

    dots = "." * (int(elapsed * 2) % 4)
    n_live = len(car_states) if car_states else 0
    screen.blit(f_md.render(f"Training{dots}  ({n_live} cars live)", True, TEXT), (14, 14))

    y = 42
    if stats:
        screen.blit(f_sm.render(f"Steps:       {stats.get('timesteps', 0):,}",      True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Generation:  {stats.get('generation', 0)}",        True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Epsilon:     {stats.get('epsilon', 1.0):.3f}",     True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Best score:  {stats.get('best_fitness', 0):+.1f}", True, GOLD), (14, y)); y += 18
        screen.blit(f_sm.render(f"Mean score:  {stats.get('mean_fitness', 0):+.1f}", True, DIM),  (14, y)); y += 18

    if best is not None and track is not None:
        total_dist_m = lap_count * track.total_length_m + progress_m
        remaining_m  = max(0.0, track.total_length_m - progress_m)
        screen.blit(f_sm.render(f"Lap:         {lap_count + 1}",                       True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Speed:       {speed_ms * 3.6:.1f} km/h",            True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Covered:     {total_dist_m:.0f} m",                 True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Remaining:   {remaining_m:.0f} m this lap",         True, DIM),  (14, y)); y += 18
        screen.blit(f_sm.render(f"Checkpoint:  {int(checkpoint_idx)}/{N_CHECKPOINTS}", True, DIM), (14, y)); y += 18

    screen.blit(f_sm.render(f"Zoom:        {zoom:.1f}x  [Scroll/+/-]", True, DIM), (14, y)); y += 18
    rays_state = "ON" if show_rays else "OFF"
    rays_color = GREEN if show_rays else DIM
    screen.blit(f_sm.render(f"Rays:        {rays_state}  [V]", True, rays_color), (14, y)); y += 18
    sc_state = "ON" if show_scores else "OFF"
    sc_color = GREEN if show_scores else DIM
    screen.blit(f_sm.render(f"Scores:      {sc_state}  [B]", True, sc_color), (14, y)); y += 18
    follow_txt = f"Follow: CAR #{focus_idx}  [C] clear" if has_sel else "Follow: click a car"
    follow_col = (120, 220, 255) if has_sel else DIM
    screen.blit(f_sm.render(follow_txt, True, follow_col), (14, y)); y += 18
    screen.blit(f_sm.render("ESC -> Menu", True, (70, 70, 90)), (14, y + 4))

    # Colour legend
    legend_y = y + 28
    if swarm_mode:
        n_packs = stats.get("n_packs", 0) if stats else 0
        screen.blit(f_sm.render(f"PACKS  ({n_packs} Rudel)", True, GOLD), (14, legend_y))
        legend_y += 18
        for pid in range(max(n_packs, 1)):
            pygame.draw.circle(screen, _pack_color(pid), (22, legend_y + 6), 5)
            screen.blit(f_sm.render(f"Rudel {pid + 1}", True, _pack_color(pid)), (32, legend_y))
            legend_y += 17
    else:
        for label, color in [("Best", GOLD), ("Top 20%", GREEN), ("Mid", (200, 130, 50)), ("Bottom", (110, 40, 40))]:
            pygame.draw.circle(screen, color, (22, legend_y + 6), 5)
            screen.blit(f_sm.render(label, True, color), (32, legend_y))
            legend_y += 17

    top_scores = stats.get("top_scores", []) if stats else []
    if top_scores:
        px0, py0 = TOP_PANEL_X0, TOP_PANEL_Y0
        panel_h = TOP_ROW_FIRST_Y + len(top_scores) * TOP_ROW_H + 20
        top_panel = pygame.Surface((TOP_PANEL_W, panel_h), pygame.SRCALPHA)
        top_panel.fill((0, 0, 0, 170))
        screen.blit(top_panel, (px0, py0))
        screen.blit(f_sm.render("TOP SCORES  (click or 1-9)", True, GOLD), (px0 + 10, py0 + 6))
        for rank, (score, sgen) in enumerate(top_scores):
            row_y = py0 + TOP_ROW_FIRST_Y + rank * TOP_ROW_H
            if rank == hover_score_idx:
                pygame.draw.rect(screen, (60, 60, 100),
                                 (px0 + 2, row_y, TOP_PANEL_W - 4, TOP_ROW_H - 1),
                                 border_radius=3)
            color = GOLD if rank == 0 else (GREEN if rank < 3 else DIM)
            if rank == hover_score_idx:
                color = (255, 255, 255)
            label = f"#{rank + 1:<2}  {score:>+8.0f}  gen {sgen}"
            if rank == hover_score_idx:
                label += "  [watch]"
            screen.blit(f_sm.render(label, True, color), (px0 + 10, row_y + 1))

    # Reward breakdown for the followed car — under the top-scores panel
    if has_sel:
        top_scores = stats.get("top_scores", []) if stats else []
        bd_y = TOP_PANEL_Y0 + (TOP_ROW_FIRST_Y + len(top_scores) * TOP_ROW_H + 28
                               if top_scores else 0)
        _draw_reward_breakdown(screen, fonts, car_states[focus_idx], focus_idx, bd_y)

    if throttle_hist is not None:
        from f1_rl.hud import draw_throttle_graph
        draw_throttle_graph(screen, throttle_hist, x=CANVAS_W - 380, y=CANVAS_H - 170)


# ── Replay browser ────────────────────────────────────────────────────────────

def draw_replay_browser(screen, fonts, replays: list, sel_idx: int) -> None:
    screen.fill(BG)
    f_title, f_lg, f_md, f_sm = fonts

    t = f_title.render("REPLAY BROWSER", True, ACCENT)
    screen.blit(t, (CANVAS_W // 2 - t.get_width() // 2, 55))

    if not replays:
        msg = f_md.render("No replays saved yet — train a model first.", True, DIM)
        screen.blit(msg, (CANVAS_W // 2 - msg.get_width() // 2, CANVAS_H // 2))
    else:
        hdr = f_sm.render(
            f"{'#':<4}  {'Score':>10}   {'Gen':>5}   {'Circuit':<26}  {'Duration':>8}",
            True, DIM,
        )
        screen.blit(hdr, (CANVAS_W // 2 - 420, 125))
        pygame.draw.line(screen, DIM, (CANVAS_W // 2 - 420, 143), (CANVAS_W // 2 + 420, 143), 1)

        item_h, list_top = 38, 150
        for i, r in enumerate(replays):
            y    = list_top + i * item_h
            rect = pygame.Rect(CANVAS_W // 2 - 420, y, 840, item_h - 4)
            if i == sel_idx:
                pygame.draw.rect(screen, SEL_BG, rect, border_radius=6)
                pygame.draw.rect(screen, ACCENT, rect, 2, border_radius=6)
            color = TEXT if i == sel_idx else DIM
            row = f_sm.render(
                f"{i+1:<4}  {r['fitness']:>+10.1f}   {r['generation']:>5}   "
                f"{r['circuit']:<26}  {r['n_frames'] / 60.0:>6.1f}s",
                True, color,
            )
            screen.blit(row, (rect.x + 10, rect.y + (rect.height - row.get_height()) // 2))
            if i == 0:
                crown = f_sm.render("*", True, GOLD)
                screen.blit(crown, (rect.x - 20, rect.y + (rect.height - crown.get_height()) // 2))

    hints = [("UP/DOWN", "Select replay"), ("ENTER", "Play"), ("ESC", "Back to menu")]
    hx, hy = 60, CANVAS_H - 90
    for key, desc in hints:
        k = f_sm.render(f"[{key}]", True, GOLD)
        d = f_sm.render(f"  {desc}", True, DIM)
        screen.blit(k, (hx, hy))
        screen.blit(d, (hx + k.get_width(), hy))
        hy += 24


# Progress bar geometry (shared between draw_replay and main.py scrubbing)
REPLAY_BAR_X, REPLAY_BAR_Y, REPLAY_BAR_W, REPLAY_BAR_H = 14, 162, 280, 8


def draw_replay(screen, fonts, track, frames: np.ndarray,
                frame_idx: int, meta: dict, paused: bool,
                speed: int, zoom: float,
                pan_x: int = 0, pan_y: int = 0) -> None:
    from f1_rl.track import meters_to_pixels, GRASS_COLOR
    screen.fill(GRASS_COLOR)

    x_m, y_m, heading, speed_ms, _throttle, progress_m = frames[frame_idx]
    px, py = meters_to_pixels(track, float(x_m), float(y_m))

    # Blit track
    if abs(zoom - 1.0) < 0.01:
        screen.blit(_get_track_surface(track), (0, 0))
        car_sx, car_sy = int(px), int(py)
    else:
        scaled = _get_zoomed_track_surface(track, zoom)
        screen.blit(scaled, (
            -(int(px * zoom - CANVAS_W / 2)) + pan_x,
            -(int(py * zoom - CANVAS_H / 2)) + pan_y,
        ))
        car_sx = int(CANVAS_W // 2 + pan_x)
        car_sy = int(CANVAS_H // 2 + pan_y)

    # Blue trail: last 150 frames
    trail_len = 150
    for fi in range(max(0, frame_idx - trail_len), frame_idx):
        tx, ty = float(frames[fi][0]), float(frames[fi][1])
        tpx, tpy = meters_to_pixels(track, tx, ty)
        tsx, tsy = _world_to_screen(tpx, tpy, px, py, zoom, pan_x, pan_y)
        if -4 <= tsx <= CANVAS_W + 4 and -4 <= tsy <= CANVAS_H + 4:
            frac = (fi - max(0, frame_idx - trail_len)) / trail_len
            intensity = int(40 + 180 * frac)
            r = max(1, int(1 + 3 * frac))
            pygame.draw.circle(screen,
                                (intensity // 4, intensity // 2, intensity),
                                (tsx, tsy), r)

    # Car dot
    screen.blit(_get_glow_surf(), (car_sx - 15, car_sy - 15))
    pygame.draw.circle(screen, (80, 180, 255), (car_sx, car_sy), 9)
    pygame.draw.circle(screen, (255, 255, 255), (car_sx, car_sy), 9, 2)
    pygame.draw.line(screen, (255, 220, 0),
                     (car_sx, car_sy),
                     (int(car_sx + math.cos(float(heading)) * 14),
                      int(car_sy - math.sin(float(heading)) * 14)), 2)

    cp_idx = min(int(float(progress_m) / track.total_length_m * N_CHECKPOINTS), N_CHECKPOINTS - 1)
    _draw_checkpoints_overlay(screen, track, px, py, zoom, cp_idx, pan_x, pan_y)

    f_title, f_lg, f_md, f_sm = fonts
    panel = pygame.Surface((310, 185), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    screen.blit(panel, (8, 8))

    pct = (frame_idx + 1) / len(frames) * 100
    screen.blit(f_md.render("REPLAY", True, (80, 180, 255)), (14, 14))
    y = 42
    screen.blit(f_sm.render(f"Score:     {meta['fitness']:+.1f}",               True, GOLD), (14, y)); y += 18
    screen.blit(f_sm.render(f"Gen:       {meta['generation']}",                  True, DIM),  (14, y)); y += 18
    screen.blit(f_sm.render(f"Circuit:   {meta['circuit']}",                     True, DIM),  (14, y)); y += 18
    screen.blit(f_sm.render(f"Speed:     {float(speed_ms) * 3.6:.1f} km/h",     True, DIM),  (14, y)); y += 18
    screen.blit(f_sm.render(f"Frame:     {frame_idx+1}/{len(frames)}  ({pct:.0f}%)", True, DIM), (14, y)); y += 18
    screen.blit(f_sm.render(f"Zoom:      {zoom:.1f}x  [Scroll/+/-]",            True, DIM),  (14, y)); y += 18
    if pan_x != 0 or pan_y != 0:
        screen.blit(f_sm.render(f"Pan:       ({pan_x:+d}, {pan_y:+d})  [F] reset", True, (180, 180, 80)), (14, y))

    # Progress / scrub bar — click to jump
    bar_x, bar_y = REPLAY_BAR_X, REPLAY_BAR_Y
    bar_w, bar_h  = REPLAY_BAR_W, REPLAY_BAR_H
    pygame.draw.rect(screen, (40, 40, 60),  (bar_x, bar_y, bar_w, bar_h), border_radius=4)
    pygame.draw.rect(screen, GOLD, (bar_x, bar_y, int(bar_w * pct / 100), bar_h), border_radius=4)
    pygame.draw.rect(screen, (100, 100, 120), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

    if paused:
        p = f_md.render("PAUSED  [SPACE]", True, (220, 220, 80))
        screen.blit(p, (CANVAS_W // 2 - p.get_width() // 2, 20))

    hint = f_sm.render(
        "[SPACE] Pause   [+/-] Speed " + str(speed) + "x"
        "   [Scroll] Zoom   [RightDrag] Pan   [F] Reset pan"
        "   [R] Restart   [ESC] Back",
        True, (60, 60, 80),
    )
    screen.blit(hint, (CANVAS_W // 2 - hint.get_width() // 2, CANVAS_H - 20))

    from f1_rl.hud import draw_throttle_graph
    start = max(0, frame_idx - 119)
    throttle_hist = deque((float(frames[fi][4]) for fi in range(start, frame_idx + 1)), maxlen=120)
    draw_throttle_graph(screen, throttle_hist, x=CANVAS_W - 380, y=CANVAS_H - 170)


# ── Live visualization loop ───────────────────────────────────────────────────

def run_visualization(screen, clock, fonts, model, track,
                      best_laps: dict, save_lap_fn) -> str:
    """Drive the loaded model interactively. Returns 'quit' or 'menu'."""
    from f1_rl.env import F1Env
    from f1_rl.hud import draw_hud, draw_throttle_graph
    from f1_rl.track import meters_to_pixels, GRASS_COLOR

    f_title, f_lg, f_md, f_sm = fonts
    env = F1Env(track=track, render_mode=None)
    obs, _ = env.reset()
    best_lap: float | None = best_laps.get(track.name)
    paused = False
    zoom = 1.0
    throttle_hist: deque[float] = deque(maxlen=120)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEWHEEL:
                zoom = max(0.25, min(10.0, zoom * (1.15 ** event.y)))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                if event.key == pygame.K_t:
                    return "menu"
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    zoom = min(10.0, zoom * 1.3)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom = max(0.25, zoom / 1.3)
                if event.key == pygame.K_0:
                    zoom = 1.0

        if not paused:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, info = env.step(action)
            throttle_hist.append(env._last_throttle)
            if info.get("lap_complete"):
                lap_time = time.time() - env.lap_start_time
                if best_lap is None or lap_time < best_lap:
                    best_lap = lap_time
                    save_lap_fn(track.name, best_lap)
                env.lap_start_time = time.time()
            if terminated:
                obs, _ = env.reset()

        screen.fill(GRASS_COLOR)
        car_px, car_py = meters_to_pixels(track, env.x_m, env.y_m)
        _draw_scene(screen, track, car_px, car_py, zoom, env.heading)
        draw_hud(screen, env.get_hud_state(), best_lap)
        draw_throttle_graph(screen, throttle_hist, x=CANVAS_W - 380, y=CANVAS_H - 170)

        if paused:
            p = f_md.render("PAUSED  [SPACE]", True, (220, 220, 80))
            screen.blit(p, (CANVAS_W // 2 - p.get_width() // 2, 20))

        hint = f_sm.render(
            f"[R] Reset   [T] Menu   [SPACE] Pause   [Scroll/+/-] Zoom {zoom:.1f}x   [ESC] Quit",
            True, (60, 60, 80),
        )
        screen.blit(hint, (CANVAS_W // 2 - hint.get_width() // 2, CANVAS_H - 20))
        pygame.display.flip()
        clock.tick(FPS)
