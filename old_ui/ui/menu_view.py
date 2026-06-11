"""The start menu: circuit list, track preview, and training parameters.

Geometry constants (MENU_LIST_*, MENU_VISIBLE, ...) are exported so app.py can
do mouse hit-testing and scrolling against the same layout drawn here.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pygame

from f1_rl.config import (
    ACCENT, BG, CANVAS_H, CANVAS_W, CIRCUITS, DIM, GENS_PRESETS, GOLD, GREEN,
    HOV_BG, MODEL_PATH, SEL_BG, STEPS_PRESETS, TEXT,
)
from old_ui.ui.widgets import draw_button, draw_checkbox


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


# ── Parameter row widget ──────────────────────────────────────────────────────

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


# ── Menu list geometry — exported so app.py can do hit-testing + scrolling ─────

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


def menu_action_rects() -> dict[str, pygame.Rect]:
    """Clickable action buttons under the parameter rows: Train / Resume / Load / Replays."""
    _, _, mode_rect = menu_param_rects()
    top = mode_rect.bottom + 14
    w   = MENU_LIST_W
    train = pygame.Rect(MENU_LIST_X, top, w, 50)
    ry = train.bottom + 10
    bw = (w - 16) // 3
    return {
        "train":   train,
        "resume":  pygame.Rect(MENU_LIST_X,                 ry, bw, 42),
        "load":    pygame.Rect(MENU_LIST_X + bw + 8,         ry, bw, 42),
        "replays": pygame.Rect(MENU_LIST_X + 2 * (bw + 8),   ry, bw, 42),
    }


def param_arrow_zones(rect: pygame.Rect) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    """Split a parameter row into (decrement ◄, type-here middle, increment ►) click zones."""
    z = 46
    left  = pygame.Rect(rect.x, rect.y, z, rect.height)
    right = pygame.Rect(rect.right - z, rect.y, z, rect.height)
    mid   = pygame.Rect(rect.x + z, rect.y, rect.width - 2 * z, rect.height)
    return left, mid, right


def draw_menu(screen, fonts, sel_idx: int, hover_idx: int,
              train_state: dict | None, n_replays: int,
              steps_preset_idx: int, steps_value: int, typing_steps: str | None,
              gens_preset_idx: int, gens_value: int, typing_gens: str | None,
              active_param: str, scroll_offset: int = 0,
              preview=None, evolution_mode: str = "classic",
              mouse_pos: tuple = (-1, -1)) -> None:
    screen.fill(BG)
    f_title, f_lg, f_md, f_sm = fonts
    mx, my = mouse_pos

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

    # Mode row — a clickable checkbox (M still toggles it too)
    is_pack  = evolution_mode == "pack"
    hov_mode = mode_rect.collidepoint(mx, my)
    pygame.draw.rect(screen, HOV_BG, mode_rect, border_radius=6)
    pygame.draw.rect(screen, GREEN if is_pack else (GOLD if hov_mode else DIM),
                     mode_rect, 2 if (is_pack or hov_mode) else 1, border_radius=6)
    draw_checkbox(
        screen, f_md,
        pygame.Rect(mode_rect.x + 12, mode_rect.y, mode_rect.width - 24, mode_rect.height),
        "Rudel-Evolution (Pack-Modus)", is_pack, hovered=hov_mode,
    )

    # ── Action buttons (clickable; keyboard shortcuts still work) ─────────
    model_exists = os.path.exists(MODEL_PATH)
    acts = menu_action_rects()
    draw_button(screen, f_md, acts["train"], "▶  Training starten",
                hovered=acts["train"].collidepoint(mx, my), primary=True)
    draw_button(screen, f_sm, acts["resume"], "Fortsetzen",
                hovered=acts["resume"].collidepoint(mx, my), enabled=bool(train_state))
    draw_button(screen, f_sm, acts["load"], "Laden & Fahren",
                hovered=acts["load"].collidepoint(mx, my), enabled=model_exists)
    draw_button(screen, f_sm, acts["replays"], f"Replays ({n_replays})",
                hovered=acts["replays"].collidepoint(mx, my))

    # ── Status line + one compact hint ────────────────────────────────────
    info_y = acts["resume"].bottom + 12
    if typing_steps is not None or typing_gens is not None:
        screen.blit(f_sm.render("Zahl tippen, ENTER bestätigen · ESC abbrechen", True, GOLD),
                    (MENU_LIST_X, info_y))
    elif train_state:
        gen = train_state.get("generation", 0)
        bf  = train_state.get("best_fitness", 0.0)
        screen.blit(f_sm.render(
            f"Letztes Training: {train_state.get('circuit', '?')}  ·  Gen {gen}  ·  Best {bf:+.0f}",
            True, GOLD), (MENU_LIST_X, info_y))
    screen.blit(f_sm.render(
        "Strecke klicken zum Wählen · ◄ ► an den Reglern klicken · Wert anklicken zum Tippen",
        True, DIM), (MENU_LIST_X, info_y + 22))
