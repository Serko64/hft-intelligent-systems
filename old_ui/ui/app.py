"""The application: pygame window, global state, and the main event loop.

This module wires the views together. It owns the high-level "screens" (menu,
training, replay browser, replay playback, drive) and routes keyboard/mouse
events to whichever screen is active.
"""
from __future__ import annotations

import glob
import os
import queue
import sys
import threading
import time
from collections import deque

import pygame

from f1_rl.config import (
    CANVAS_H, CANVAS_W, CIRCUITS,
    FPS, GENS_PRESETS, MODEL_PATH, REPLAY_DIR, STEPS_PRESETS,
)
from f1_rl.learning.trainer import load_training_state
from f1_rl.simulation.track_loader import load_track, meters_to_pixels
from old_ui.ui.camera import _world_to_screen
from old_ui.ui.menu_view import (
    MENU_ITEM_H, MENU_LIST_TOP, MENU_LIST_W, MENU_LIST_X, MENU_VISIBLE,
    draw_menu, menu_action_rects, menu_param_rects, param_arrow_zones,
)
from old_ui.ui.replay_view import (
    REPLAY_BAR_H, REPLAY_BAR_W, REPLAY_BAR_X, REPLAY_BAR_Y,
    draw_replay, draw_replay_browser, run_visualization,
)
from old_ui.ui.storage import (
    ReplayState, load_best_laps, load_replay_list,
    open_replay_file, save_best_lap, try_open_training_replay,
)
from old_ui.ui.theme import make_fonts
from old_ui.ui.training_view import (
    TOP_PANEL_W, TOP_PANEL_X0, TOP_PANEL_Y0, TOP_ROW_FIRST_Y, TOP_ROW_H,
    draw_training, training_widget_rects,
)


# ── Track preview (menu) — lazy background loading ────────────────────────────

_preview_cache: dict[str, object] = {}   # name -> TrackData | "loading" | "error"
_preview_lock = threading.Lock()


def _load_preview(name: str, fallback: str | None, hw: float) -> None:
    try:
        tr = load_track(name, geojson_fallback_path=fallback, half_width_m=hw)
        with _preview_lock:
            _preview_cache[name] = tr
    except Exception as e:
        print(f"[main] Preview load failed for {name}: {e}")
        with _preview_lock:
            _preview_cache[name] = "error"


def _ensure_preview(sel_idx: int):
    """Return the cached preview for the selected circuit, kicking off a load if needed."""
    name, fallback, hw = CIRCUITS[sel_idx]
    with _preview_lock:
        if name in _preview_cache:
            return _preview_cache[name]
        _preview_cache[name] = "loading"
    threading.Thread(target=_load_preview, args=(name, fallback, hw), daemon=True).start()
    return "loading"


# ── Training thread ───────────────────────────────────────────────────────────

class _TrainingThread(threading.Thread):
    def __init__(self, query: str, fallback: str, half_width: float,
                 render_queue: queue.Queue, stats_queue: queue.Queue,
                 resume: bool = False, steps_per_gen: int = 5_000,
                 total_gens: int = 200):
        super().__init__(daemon=True)
        self.query          = query
        self.fallback       = fallback
        self.half_width     = half_width
        self.render_queue   = render_queue
        self.stats_queue    = stats_queue
        self.resume         = resume
        self.steps_per_gen  = steps_per_gen
        self.total_gens     = total_gens
        self.model          = None
        self.track          = None
        self.error: Exception | None = None
        self.done = False

    def run(self):
        try:
            from f1_rl.learning.trainer import train
            self.track = load_track(
                self.query,
                geojson_fallback_path=self.fallback,
                half_width_m=self.half_width,
            )
            self.model, _ = train(
                track=self.track,
                render_queue=self.render_queue,
                stats_queue=self.stats_queue,
                resume=self.resume,
                steps_per_gen=self.steps_per_gen,
                total_gens=self.total_gens,
            )
        except Exception as e:
            self.error = e
        finally:
            self.done = True


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption("F1 RL Simulator")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    # ── App state ─────────────────────────────────────────────────────────
    state          = "menu"
    sel_idx        = 0
    hover_idx      = -1
    zoom           = 1.0
    active_param   = "steps"
    scroll_offset  = 0

    steps_preset_idx = STEPS_PRESETS.index(5_000)
    steps_value      = 5_000
    typing_steps: str | None = None

    gens_preset_idx = GENS_PRESETS.index(200)
    gens_value      = 200
    typing_gens: str | None = None

    training_thread: _TrainingThread | None = None
    training_start = 0.0
    loaded_model   = None
    loaded_track   = None
    train_state    = load_training_state()

    render_queue: queue.Queue = queue.Queue(maxsize=4)
    stats_queue:  queue.Queue = queue.Queue(maxsize=10)
    car_states: list | None   = None
    last_stats: dict          = {}
    training_throttle_hist: deque = deque(maxlen=120)
    car_trail:              deque = deque(maxlen=60)
    hover_score_idx: int          = -1
    show_rays: bool               = False
    show_scores: bool             = True
    selected_car: int             = -1
    best_car_marker_m: tuple | None = None
    best_car_total_dist: float      = 0.0

    replay     = ReplayState()
    replay_list: list[dict] = []
    replay_sel:  int        = 0

    # ── Actions (shared by keyboard shortcuts and on-screen buttons) ───────
    def _start_training(resume_flag: bool) -> None:
        nonlocal render_queue, stats_queue, car_states, last_stats
        nonlocal best_car_marker_m, best_car_total_dist, selected_car, zoom
        nonlocal training_thread, training_start, state
        query, fallback, hw = CIRCUITS[sel_idx]
        render_queue        = queue.Queue(maxsize=4)
        stats_queue         = queue.Queue(maxsize=10)
        car_states          = None
        last_stats          = {}
        best_car_marker_m   = None
        best_car_total_dist = 0.0
        selected_car        = -1
        zoom                = 1.0
        training_throttle_hist.clear()
        car_trail.clear()
        training_thread = _TrainingThread(
            query, fallback, hw, render_queue, stats_queue,
            resume=resume_flag,
            steps_per_gen=steps_value,
            total_gens=gens_value,
        )
        training_thread.start()
        training_start = time.time()
        state = "training"

    def _load_and_drive() -> None:
        nonlocal loaded_track, loaded_model, state
        if os.path.exists(MODEL_PATH):
            from f1_rl.learning.agent import neuronal_net_from_weight_file
            query, fallback, hw = CIRCUITS[sel_idx]
            loaded_track = load_track(query, geojson_fallback_path=fallback, half_width_m=hw)
            loaded_model = neuronal_net_from_weight_file(MODEL_PATH)
            state = "visualization"
        else:
            print("[main] No model found at", MODEL_PATH)

    def _open_replays() -> None:
        nonlocal replay_list, replay_sel, state
        replay_list = load_replay_list()
        replay_sel  = 0
        state = "replays"

    def _cycle_preset(which: str, delta: int) -> None:
        """Step the steps/generations preset up (+1) or down (-1) and clear typing."""
        nonlocal steps_preset_idx, steps_value, gens_preset_idx, gens_value
        nonlocal typing_steps, typing_gens
        if which == "steps":
            base = steps_preset_idx if steps_preset_idx >= 0 else (len(STEPS_PRESETS) if delta < 0 else -1)
            steps_preset_idx = max(0, min(len(STEPS_PRESETS) - 1, base + delta))
            steps_value = STEPS_PRESETS[steps_preset_idx]
            typing_steps = None
        else:
            base = gens_preset_idx if gens_preset_idx >= 0 else (len(GENS_PRESETS) if delta < 0 else -1)
            gens_preset_idx = max(0, min(len(GENS_PRESETS) - 1, base + delta))
            gens_value = GENS_PRESETS[gens_preset_idx]
            typing_gens = None

    # ── Event loop ────────────────────────────────────────────────────────
    while True:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # ── Menu ──────────────────────────────────────────────────────
            if state == "menu":
                if event.type == pygame.KEYDOWN:
                    if typing_steps is not None or typing_gens is not None:
                        is_steps = typing_steps is not None
                        typed    = typing_steps if is_steps else typing_gens
                        if event.key == pygame.K_ESCAPE:
                            if is_steps: typing_steps = None
                            else:        typing_gens  = None
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            if typed:
                                val = max(1, int(typed))
                                if is_steps:
                                    steps_value      = val
                                    steps_preset_idx = STEPS_PRESETS.index(val) if val in STEPS_PRESETS else -1
                                else:
                                    gens_value      = val
                                    gens_preset_idx = GENS_PRESETS.index(val) if val in GENS_PRESETS else -1
                            if is_steps: typing_steps = None
                            else:        typing_gens  = None
                        elif event.key == pygame.K_BACKSPACE:
                            if is_steps: typing_steps = typed[:-1]
                            else:        typing_gens  = typed[:-1]
                        elif event.unicode.isdigit() and len(typed) < 9:
                            if is_steps: typing_steps = typed + event.unicode
                            else:        typing_gens  = typed + event.unicode
                    else:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit(); sys.exit()
                        if event.key == pygame.K_DOWN:
                            sel_idx = (sel_idx + 1) % len(CIRCUITS)
                        if event.key == pygame.K_UP:
                            sel_idx = (sel_idx - 1) % len(CIRCUITS)
                        if event.key == pygame.K_TAB:
                            active_param = "gens" if active_param == "steps" else "steps"
                        if event.key == pygame.K_LEFT:
                            _cycle_preset(active_param, -1)
                        if event.key == pygame.K_RIGHT:
                            _cycle_preset(active_param, +1)
                        if event.unicode.isdigit():
                            if active_param == "steps": typing_steps = event.unicode
                            else:                       typing_gens  = event.unicode
                        if event.key in (pygame.K_RETURN, pygame.K_r):
                            resume = event.key == pygame.K_r and train_state is not None
                            query, fallback, hw = CIRCUITS[sel_idx]
                            render_queue        = queue.Queue(maxsize=4)
                            stats_queue         = queue.Queue(maxsize=10)
                            car_states          = None
                            last_stats          = {}
                            best_car_marker_m   = None
                            best_car_total_dist = 0.0
                            selected_car        = -1
                            zoom                = 1.0
                            training_throttle_hist.clear()
                            car_trail.clear()
                            training_thread = _TrainingThread(
                                query, fallback, hw, render_queue, stats_queue,
                                resume=resume,
                                steps_per_gen=steps_value,
                                total_gens=gens_value,
                            )
                            training_thread.start()
                            training_start = time.time()
                            state = "training"
                        if event.key == pygame.K_l:
                            if os.path.exists(MODEL_PATH):
                                from f1_rl.learning.agent import neuronal_net_from_weight_file
                                query, fallback, hw = CIRCUITS[sel_idx]
                                loaded_track = load_track(query, geojson_fallback_path=fallback,
                                                          half_width_m=hw)
                                loaded_model = neuronal_net_from_weight_file(MODEL_PATH)
                                state = "visualization"
                            else:
                                print("[main] No model found at", MODEL_PATH)
                        if event.key == pygame.K_p:
                            replay_list = load_replay_list()
                            replay_sel  = 0
                            state = "replays"

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for row in range(MENU_VISIBLE):
                        i = scroll_offset + row
                        if i >= len(CIRCUITS):
                            break
                        r = pygame.Rect(MENU_LIST_X, MENU_LIST_TOP + row * MENU_ITEM_H,
                                        MENU_LIST_W, MENU_ITEM_H - 4)
                        if r.collidepoint(mx, my):
                            sel_idx = i

            # ── Training ──────────────────────────────────────────────────
            elif state == "training":
                if event.type == pygame.MOUSEWHEEL:
                    zoom = max(0.25, min(10.0, zoom * (1.15 ** event.y)))
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state = "menu"
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        zoom = min(10.0, zoom * 1.3)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        zoom = max(0.25, zoom / 1.3)
                    if event.key == pygame.K_0:
                        zoom = 1.0
                    if event.key == pygame.K_v:
                        show_rays = not show_rays
                    if event.key == pygame.K_b:
                        show_scores = not show_scores
                    if event.key == pygame.K_c:
                        selected_car = -1
                    if event.unicode in "123456789":
                        rank = int(event.unicode) - 1
                        top_scores = last_stats.get("top_scores", []) if last_stats else []
                        if rank < len(top_scores):
                            _score, sgen = top_scores[rank]
                            if try_open_training_replay(sgen, replay):
                                state = "training_replay"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    top_scores = last_stats.get("top_scores", []) if last_stats else []
                    panel_hit = False
                    for rank in range(len(top_scores)):
                        row_rect = pygame.Rect(
                            TOP_PANEL_X0,
                            TOP_PANEL_Y0 + TOP_ROW_FIRST_Y + rank * TOP_ROW_H,
                            TOP_PANEL_W, TOP_ROW_H,
                        )
                        if row_rect.collidepoint(mx, my):
                            panel_hit = True
                            _score, sgen = top_scores[rank]
                            if try_open_training_replay(sgen, replay):
                                state = "training_replay"
                            break
                    # Click a car to follow it (and show its score breakdown)
                    if not panel_hit and car_states and training_thread and training_thread.track:
                        _tr = training_thread.track
                        focus = selected_car if 0 <= selected_car < len(car_states) else 0
                        fpx, fpy = meters_to_pixels(_tr, car_states[focus][0], car_states[focus][1])
                        picked, best_d = -1, 22
                        for i, ci in enumerate(car_states):
                            cpx, cpy = meters_to_pixels(_tr, ci[0], ci[1])
                            sx, sy = _world_to_screen(cpx, cpy, fpx, fpy, zoom)
                            d = ((sx - mx) ** 2 + (sy - my) ** 2) ** 0.5
                            if d < best_d:
                                best_d, picked = d, i
                        selected_car = picked   # -1 if click missed all cars → clears

            # ── Replay browser ────────────────────────────────────────────
            elif state == "replays":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state = "menu"
                    if event.key == pygame.K_DOWN and replay_list:
                        replay_sel = (replay_sel + 1) % len(replay_list)
                    if event.key == pygame.K_UP and replay_list:
                        replay_sel = (replay_sel - 1) % len(replay_list)
                    if event.key == pygame.K_RETURN and replay_list:
                        meta = replay_list[replay_sel]
                        if open_replay_file(meta["path"], meta, replay, return_state="replays"):
                            state = "replay"

            # ── Replay playback (shared by "replay" and "training_replay") ──
            elif state in ("replay", "training_replay"):
                if event.type == pygame.MOUSEWHEEL:
                    replay.zoom = max(0.25, min(10.0, replay.zoom * (1.15 ** event.y)))
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:
                        replay.dragging      = True
                        replay.drag_start    = (mx, my)
                        replay.drag_start_pan = (replay.pan_x, replay.pan_y)
                    elif event.button == 1 and replay.frames is not None:
                        bx, by, bw, bh = REPLAY_BAR_X, REPLAY_BAR_Y, REPLAY_BAR_W, REPLAY_BAR_H
                        if bx <= mx <= bx + bw and by - 6 <= my <= by + bh + 6:
                            frac = max(0.0, min(1.0, (mx - bx) / bw))
                            replay.idx = int(frac * (len(replay.frames) - 1))
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:
                        replay.dragging = False
                if event.type == pygame.MOUSEMOTION and replay.dragging:
                    replay.pan_x = replay.drag_start_pan[0] + (mx - replay.drag_start[0])
                    replay.pan_y = replay.drag_start_pan[1] + (my - replay.drag_start[1])
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        replay.dragging = False
                        state = replay.return_state
                    if event.key == pygame.K_SPACE:
                        replay.paused = not replay.paused
                    if event.key == pygame.K_r:
                        replay.idx = 0; replay.paused = False
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        replay.speed = min(8, replay.speed * 2)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        replay.speed = max(1, replay.speed // 2)
                    if event.key == pygame.K_0:
                        replay.reset_view()
                    if event.key == pygame.K_f:
                        replay.reset_pan()

        # ── Per-frame updates ─────────────────────────────────────────────
        if state == "menu":
            # Keep selection within the scroll window
            if sel_idx < scroll_offset:
                scroll_offset = sel_idx
            elif sel_idx >= scroll_offset + MENU_VISIBLE:
                scroll_offset = sel_idx - MENU_VISIBLE + 1
            scroll_offset = max(0, min(scroll_offset, max(0, len(CIRCUITS) - MENU_VISIBLE)))

            hover_idx = -1
            for row in range(MENU_VISIBLE):
                i = scroll_offset + row
                if i >= len(CIRCUITS):
                    break
                r = pygame.Rect(MENU_LIST_X, MENU_LIST_TOP + row * MENU_ITEM_H,
                                MENU_LIST_W, MENU_ITEM_H - 4)
                if r.collidepoint(mx, my):
                    hover_idx = i

        if state == "training":
            try:
                while True:
                    car_states = render_queue.get_nowait()
                    if car_states:
                        training_throttle_hist.append(car_states[0][4])
                        car_trail.append((car_states[0][0], car_states[0][1]))
                        t_len = (training_thread.track.total_length_m
                                 if training_thread and training_thread.track else 1.0)
                        # Check ALL cars — the furthest one may not be rank-0.
                        # Single-lap objective: cap the marker at one lap so the
                        # "star" stops at the finish and is not collected again.
                        for ci in car_states:
                            prog     = ci[6]
                            laps     = ci[7] if len(ci) > 7 else 0
                            eff_dist = min(laps * t_len + prog, t_len)
                            if eff_dist > best_car_total_dist:
                                best_car_total_dist = eff_dist
                                best_car_marker_m   = (ci[0], ci[1])
            except queue.Empty:
                pass
            try:
                while True:
                    last_stats = stats_queue.get_nowait()
            except queue.Empty:
                pass
            hover_score_idx = -1
            top_scores = last_stats.get("top_scores", []) if last_stats else []
            for rank in range(len(top_scores)):
                row_rect = pygame.Rect(
                    TOP_PANEL_X0,
                    TOP_PANEL_Y0 + TOP_ROW_FIRST_Y + rank * TOP_ROW_H,
                    TOP_PANEL_W, TOP_ROW_H,
                )
                if row_rect.collidepoint(mx, my):
                    hover_score_idx = rank
                    break

        if state == "training_replay":
            try:
                while True: render_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                while True: last_stats = stats_queue.get_nowait()
            except queue.Empty:
                pass

        if state in ("replay", "training_replay") and not replay.paused and replay.frames is not None:
            for _ in range(replay.speed):
                replay.idx = (replay.idx + 1) % len(replay.frames)

        if state in ("training", "training_replay") and training_thread and training_thread.done:
            if training_thread.error:
                print(f"[main] Training error: {training_thread.error}")
                state = "menu"
            else:
                loaded_model = training_thread.model
                loaded_track = training_thread.track
                car_states   = None
                state = "visualization"
            train_state     = load_training_state()
            training_thread = None

        # ── Render ────────────────────────────────────────────────────────
        n_replays = len(glob.glob(os.path.join(REPLAY_DIR, "replay_*.npz")))

        if state == "menu":
            preview = _ensure_preview(sel_idx)
            draw_menu(screen, fonts, sel_idx, hover_idx, train_state, n_replays,
                      steps_preset_idx, steps_value, typing_steps,
                      gens_preset_idx,  gens_value,  typing_gens,
                      active_param, scroll_offset=scroll_offset,
                      preview=preview)

        elif state == "training":
            live_track     = training_thread.track if training_thread else None
            best_marker_px = None
            if best_car_marker_m is not None and live_track is not None:
                best_marker_px = meters_to_pixels(live_track, best_car_marker_m[0], best_car_marker_m[1])
            draw_training(
                screen, fonts, track=live_track, car_states=car_states,
                stats=last_stats, elapsed=time.time() - training_start,
                zoom=zoom, throttle_hist=training_throttle_hist, car_trail=car_trail,
                hover_score_idx=hover_score_idx, best_marker_px=best_marker_px,
                show_rays=show_rays,
                focus_idx=selected_car, show_scores=show_scores,
            )

        elif state == "visualization":
            result = run_visualization(screen, clock, fonts, loaded_model, loaded_track,
                                       load_best_laps(), save_best_lap)
            state = "menu" if result != "quit" else None
            if state is None:
                pygame.quit(); sys.exit()
            continue

        elif state == "replays":
            draw_replay_browser(screen, fonts, replay_list, replay_sel)

        elif state in ("replay", "training_replay") and replay.frames is not None and replay.track is not None:
            draw_replay(screen, fonts, replay.track, replay.frames,
                        replay.idx, replay.meta, replay.paused, replay.speed, replay.zoom,
                        pan_x=replay.pan_x, pan_y=replay.pan_y)
            if state == "training_replay":
                f_title, f_lg, f_md, f_sm = fonts
                badge = f_sm.render(
                    "TRAINING RUNNING IN BACKGROUND  [ESC] back to live view",
                    True, (80, 200, 80),
                )
                screen.blit(badge, (CANVAS_W // 2 - badge.get_width() // 2, CANVAS_H - 40))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
