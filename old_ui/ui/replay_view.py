"""Replay browser, replay playback, and the interactive "load & drive" loop."""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import pygame

from f1_rl.config import ACCENT, BG, CANVAS_H, CANVAS_W, DIM, FPS, GOLD, SEL_BG, TEXT
from f1_rl.learning.agent import act
from f1_rl.simulation.environment import F1Env, N_CHECKPOINTS
from f1_rl.simulation.track_loader import meters_to_pixels
from f1_rl.simulation.track_render import GRASS_COLOR
from old_ui.ui.camera import (
    _draw_checkpoints_overlay, _draw_scene, _get_glow_surf,
    _get_track_surface, _get_zoomed_track_surface, _world_to_screen,
)
from old_ui.ui.hud import draw_hud, draw_throttle_graph


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


# Progress bar geometry (shared between draw_replay and app.py scrubbing)
REPLAY_BAR_X, REPLAY_BAR_Y, REPLAY_BAR_W, REPLAY_BAR_H = 14, 162, 280, 8


def draw_replay(screen, fonts, track, frames: np.ndarray,
                frame_idx: int, meta: dict, paused: bool,
                speed: int, zoom: float,
                pan_x: int = 0, pan_y: int = 0) -> None:
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

    start = max(0, frame_idx - 119)
    throttle_hist = deque((float(frames[fi][4]) for fi in range(start, frame_idx + 1)), maxlen=120)
    draw_throttle_graph(screen, throttle_hist, x=CANVAS_W - 380, y=CANVAS_H - 170)


# ── Live visualization loop ───────────────────────────────────────────────────

def run_visualization(screen, clock, fonts, model, track,
                      best_laps: dict, save_lap_fn) -> str:
    """Drive the loaded model interactively. Returns 'quit' or 'menu'."""
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
            action = act(model, obs)
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
