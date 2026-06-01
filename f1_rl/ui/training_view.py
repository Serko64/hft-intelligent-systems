"""The live training view: draws the whole population of cars, the stats panel,
the colour legend, the top-scores leaderboard, and (when a car is followed) its
reward breakdown.
"""
from __future__ import annotations

import math
import time

import pygame

from f1_rl.config import CANVAS_H, CANVAS_W, DIM, GOLD, GREEN
from f1_rl.simulation.environment import N_CHECKPOINTS, REWARD_PARTS
from f1_rl.simulation.track_loader import meters_to_pixels
from f1_rl.simulation.track_render import GRASS_COLOR
from f1_rl.ui.camera import (
    _draw_checkpoints_overlay, _draw_rays, _get_glow_surf,
    _get_track_surface, _get_zoomed_track_surface, _world_to_screen,
)
from f1_rl.ui.hud import draw_throttle_graph
from f1_rl.ui.theme import draw_star, pack_color, rank_color_size
from f1_rl.ui.widgets import draw_button, draw_checkbox

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

# Top-scores panel geometry — exported so app.py can do hit-testing
TOP_PANEL_W         = 230
TOP_PANEL_X0        = CANVAS_W - TOP_PANEL_W - 8
TOP_PANEL_Y0        = 8
TOP_ROW_FIRST_Y     = 26   # offset from panel top to first score row
TOP_ROW_H           = 17

_stats_panel_surf: pygame.Surface | None = None


def training_widget_rects() -> dict[str, pygame.Rect]:
    """Clickable controls in the bottom-left corner (checkboxes + buttons).
    Fixed position so app.py can hit-test them without tracking the dynamic
    stats-panel layout."""
    x0, y0 = 8, CANVAS_H - 150
    return {
        "rays":   pygame.Rect(x0 + 12, y0 + 10, 210, 26),
        "scores": pygame.Rect(x0 + 12, y0 + 40, 210, 26),
        "menu":   pygame.Rect(x0 + 12, y0 + 76, 100, 32),
        "clear":  pygame.Rect(x0 + 120, y0 + 76, 112, 32),
    }


def _draw_training_widgets(screen, fonts, show_rays: bool, show_scores: bool,
                           has_sel: bool, mouse_pos: tuple) -> None:
    f_title, f_lg, f_md, f_sm = fonts
    mx, my = mouse_pos
    panel = pygame.Surface((244, 150), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 175))
    screen.blit(panel, (8, CANVAS_H - 156))
    wr = training_widget_rects()
    draw_checkbox(screen, f_sm, wr["rays"], "Sensor-Strahlen", show_rays,
                  hovered=wr["rays"].collidepoint(mx, my))
    draw_checkbox(screen, f_sm, wr["scores"], "Scores anzeigen", show_scores,
                  hovered=wr["scores"].collidepoint(mx, my))
    draw_button(screen, f_sm, wr["menu"], "Menü",
                hovered=wr["menu"].collidepoint(mx, my))
    draw_button(screen, f_sm, wr["clear"], "Auswahl lösen",
                hovered=wr["clear"].collidepoint(mx, my), enabled=has_sel)


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
                  focus_idx: int = -1, show_scores: bool = False,
                  mouse_pos: tuple = (-1, -1)) -> None:
    """Render all population cars live.

    car_states: list of (x_m,y_m,heading,speed_ms,throttle,cp_idx,progress) tuples,
                sorted best-first.  None or empty = waiting for first generation.
    """
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
                    draw_star(screen, bm_sx, bm_sy, 11, GOLD, (180, 120, 0))

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
                    color = pack_color(ci[9])
                    r = 9 if rank == 0 else 6
                else:
                    color, r = rank_color_size(frac)
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
        msg = f_lg.render("Loading track...", True, DIM)
        screen.blit(msg, (CANVAS_W // 2 - msg.get_width() // 2, CANVAS_H // 2 - 20))

    global _stats_panel_surf
    if _stats_panel_surf is None:
        _stats_panel_surf = pygame.Surface((310, 310), pygame.SRCALPHA)
        _stats_panel_surf.fill((0, 0, 0, 170))
    screen.blit(_stats_panel_surf, (8, 8))

    dots = "." * (int(elapsed * 2) % 4)
    n_live = len(car_states) if car_states else 0
    screen.blit(f_md.render(f"Training{dots}  ({n_live} cars live)", True, (220, 220, 220)), (14, 14))

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
    follow_txt = f"Verfolge: CAR #{focus_idx}" if has_sel else "Auto anklicken zum Verfolgen"
    follow_col = (120, 220, 255) if has_sel else DIM
    screen.blit(f_sm.render(follow_txt, True, follow_col), (14, y)); y += 18

    # Colour legend
    legend_y = y + 28
    if swarm_mode:
        n_packs = stats.get("n_packs", 0) if stats else 0
        screen.blit(f_sm.render(f"PACKS  ({n_packs} Rudel)", True, GOLD), (14, legend_y))
        legend_y += 18
        for pid in range(max(n_packs, 1)):
            pygame.draw.circle(screen, pack_color(pid), (22, legend_y + 6), 5)
            screen.blit(f_sm.render(f"Rudel {pid + 1}", True, pack_color(pid)), (32, legend_y))
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
        draw_throttle_graph(screen, throttle_hist, x=CANVAS_W - 380, y=CANVAS_H - 170)

    # Clickable controls (mouse-friendly; keys V/B/C/ESC still work)
    _draw_training_widgets(screen, fonts, show_rays, show_scores, has_sel, mouse_pos)
