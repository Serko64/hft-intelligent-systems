"""F1 RL Simulator — entry point: main menu, live training view, visualization."""
from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time

import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CANVAS_W, CANVAS_H = 1600, 1000
FPS = 60

ROOT = os.path.join(os.path.dirname(__file__), "..")
CIRCUITS_DIR = os.path.join(ROOT, "circuits")
MODEL_PATH = os.path.join(ROOT, "model", "ppo_f1.zip")
BEST_LAPS_PATH = os.path.join(ROOT, "model", "best_laps.json")


def _load_best_laps() -> dict[str, float]:
    if os.path.exists(BEST_LAPS_PATH):
        with open(BEST_LAPS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_best_lap(circuit_name: str, seconds: float) -> None:
    laps = _load_best_laps()
    laps[circuit_name] = seconds
    os.makedirs(os.path.dirname(BEST_LAPS_PATH), exist_ok=True)
    with open(BEST_LAPS_PATH, "w", encoding="utf-8") as f:
        json.dump(laps, f, indent=2)

# (name, geojson_path, half_width_m)
CIRCUITS = [
    ("Circuit de Monaco",            os.path.join(CIRCUITS_DIR, "mc-1929.geojson"), 20.0),
    ("Silverstone Circuit",          os.path.join(CIRCUITS_DIR, "gb-1948.geojson"), 20.0),
    ("Autodromo Nazionale Monza",    os.path.join(CIRCUITS_DIR, "it-1922.geojson"), 20.0),
    ("Circuit de Spa-Francorchamps", os.path.join(CIRCUITS_DIR, "be-1925.geojson"), 20.0),
    ("Suzuka Circuit",               os.path.join(CIRCUITS_DIR, "jp-1962.geojson"), 20.0),
]

BG     = (12, 12, 18)
ACCENT = (200, 30, 30)
TEXT   = (220, 220, 220)
DIM    = (110, 110, 130)
SEL_BG = (55, 25, 75)
HOV_BG = (35, 35, 55)
GOLD   = (255, 215, 0)


# ─────────────────────────────────────────────────────────────────────────────
class _TrainingThread(threading.Thread):
    def __init__(self, query: str, fallback: str, half_width: float,
                 render_queue: queue.Queue, stats_queue: queue.Queue,
                 resume: bool = False):
        super().__init__(daemon=True)
        self.query = query
        self.fallback = fallback
        self.half_width = half_width
        self.render_queue = render_queue
        self.stats_queue = stats_queue
        self.resume = resume
        self.model = None
        self.track = None
        self.error: Exception | None = None
        self.done = False

    def run(self):
        try:
            from f1_rl.track import load_track
            from f1_rl.train import train
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
            )
        except Exception as e:
            self.error = e
        finally:
            self.done = True


# ─────────────────────────────────────────────────────────────────────────────
def _font(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont(name, size, bold=bold)


def _draw_menu(screen, fonts, sel_idx: int, hover_idx: int, train_state: dict | None) -> None:
    screen.fill(BG)
    f_title, f_lg, f_md, f_sm = fonts

    t = f_title.render("F1 RL SIMULATOR", True, ACCENT)
    screen.blit(t, (CANVAS_W // 2 - t.get_width() // 2, 60))

    sub = f_sm.render("Select circuit → ENTER to train, L to load model", True, DIM)
    screen.blit(sub, (CANVAS_W // 2 - sub.get_width() // 2, 115))

    item_h = 44
    list_top = 175
    for i, (name, _, _hw) in enumerate(CIRCUITS):
        y = list_top + i * item_h
        rect = pygame.Rect(CANVAS_W // 2 - 230, y, 460, item_h - 4)
        if i == sel_idx:
            pygame.draw.rect(screen, SEL_BG, rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT, rect, 2, border_radius=6)
        elif i == hover_idx:
            pygame.draw.rect(screen, HOV_BG, rect, border_radius=6)
        label = f_md.render(name, True, TEXT if i == sel_idx else DIM)
        screen.blit(label, (rect.x + 16, rect.y + (rect.height - label.get_height()) // 2))

    model_exists = os.path.exists(MODEL_PATH)

    # Resume info panel
    if train_state:
        ts = train_state.get("timesteps_done", 0)
        tot = train_state.get("total_timesteps", 1)
        circuit = train_state.get("circuit", "?")
        pct = min(100.0, ts / tot * 100)
        info = f_sm.render(
            f"Letztes Training: {circuit}  —  {ts:,} / {tot:,} steps  ({pct:.1f}%)",
            True, GOLD,
        )
        screen.blit(info, (CANVAS_W // 2 - info.get_width() // 2, list_top + len(CIRCUITS) * item_h + 10))

    hints = [
        ("ENTER", "Neu trainieren  (live view)"),
        ("R", "Training fortsetzen  (Resume)" if train_state else "Resume  (kein Checkpoint)"),
        ("L", f"Laden & Fahren  {'✓ Modell gefunden' if model_exists else '(noch kein Modell)'}"),
        ("↑ ↓", "Strecke wählen"),
        ("ESC", "Beenden"),
    ]
    hx, hy = 60, CANVAS_H - 135
    for key, desc in hints:
        k = f_sm.render(f"[{key}]", True, GOLD)
        d = f_sm.render(f"  {desc}", True, DIM)
        screen.blit(k, (hx, hy))
        screen.blit(d, (hx + k.get_width(), hy))
        hy += 24


_track_surf_cache: dict = {}
_glow_surf: pygame.Surface | None = None
_stats_panel: pygame.Surface | None = None


def _get_track_surface(track) -> pygame.Surface:
    tid = id(track)
    if tid not in _track_surf_cache:
        from f1_rl.track import draw_track
        surf = pygame.Surface((track.canvas_w, track.canvas_h))
        surf.fill((12, 12, 18))
        draw_track(surf, track)
        _track_surf_cache[tid] = surf
    return _track_surf_cache[tid]


def _get_glow_surf() -> pygame.Surface:
    global _glow_surf
    if _glow_surf is None:
        _glow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(_glow_surf, (255, 80, 80, 60), (15, 15), 15)
    return _glow_surf


_scaled_track_cache: dict = {"zoom": None, "track_id": None, "surf": None}


def _get_zoomed_track_surface(track, zoom: float) -> pygame.Surface:
    key_zoom = round(zoom, 2)
    key_tid = id(track)
    if _scaled_track_cache["zoom"] != key_zoom or _scaled_track_cache["track_id"] != key_tid:
        base = _get_track_surface(track)
        if abs(key_zoom - 1.0) < 0.01:
            surf = base
        else:
            tw, th = base.get_width(), base.get_height()
            surf = pygame.transform.scale(base, (int(tw * key_zoom), int(th * key_zoom)))
        _scaled_track_cache["zoom"] = key_zoom
        _scaled_track_cache["track_id"] = key_tid
        _scaled_track_cache["surf"] = surf
    return _scaled_track_cache["surf"]


def _draw_scene(
    screen, track, car_px: float, car_py: float, zoom: float, car_heading: float
) -> None:
    """Draw track surface + car dot with zoom. At zoom=1 shows full track; zoom>1 follows car."""
    if abs(zoom - 1.0) < 0.01:
        screen.blit(_get_track_surface(track), (0, 0))
        sx, sy = int(car_px), int(car_py)
    else:
        scaled = _get_zoomed_track_surface(track, zoom)
        off_x = int(car_px * zoom - CANVAS_W / 2)
        off_y = int(car_py * zoom - CANVAS_H / 2)
        screen.blit(scaled, (-off_x, -off_y))
        sx, sy = CANVAS_W // 2, CANVAS_H // 2

    screen.blit(_get_glow_surf(), (sx - 15, sy - 15))
    pygame.draw.circle(screen, (255, 50, 50), (sx, sy), 9)
    pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 9, 2)
    hdx = int(sx + math.cos(car_heading) * 14)
    hdy = int(sy - math.sin(car_heading) * 14)
    pygame.draw.line(screen, (255, 220, 0), (sx, sy), (hdx, hdy), 2)


def _draw_training(screen, fonts, track, car_state, stats, elapsed: float,
                   zoom: float = 1.0, throttle_hist=None) -> None:
    """Render the live training view: track + moving car dot + stats overlay."""
    from f1_rl.track import meters_to_pixels, GRASS_COLOR

    screen.fill(GRASS_COLOR)

    if track is not None:
        if car_state is not None:
            x_m, y_m, heading, speed_ms, _thr = car_state
            px, py = meters_to_pixels(track, x_m, y_m)
            _draw_scene(screen, track, px, py, zoom, heading)
        else:
            screen.blit(_get_track_surface(track), (0, 0))
    else:
        f_title, f_lg, f_md, f_sm = fonts
        msg = f_lg.render("Lädt Strecke ...", True, DIM)
        screen.blit(msg, (CANVAS_W // 2 - msg.get_width() // 2, CANVAS_H // 2 - 20))

    # ── Stats overlay (top-left) ─────────────────────────────────────────
    f_title, f_lg, f_md, f_sm = fonts

    global _stats_panel
    if _stats_panel is None:
        _stats_panel = pygame.Surface((260, 140), pygame.SRCALPHA)
        _stats_panel.fill((0, 0, 0, 170))
    screen.blit(_stats_panel, (8, 8))

    dots = "." * (int(elapsed * 2) % 4)
    screen.blit(f_md.render(f"Training{dots}", True, TEXT), (14, 14))

    if stats:
        ts = stats.get("timesteps", 0)
        screen.blit(f_sm.render(f"Timesteps: {ts:,}", True, DIM), (14, 40))

    if car_state is not None:
        x_m, y_m, heading, speed_ms, _thr = car_state
        screen.blit(f_sm.render(f"Speed:  {speed_ms * 3.6:.1f} km/h", True, DIM), (14, 58))

    screen.blit(f_sm.render(f"Zoom:   {zoom:.1f}×  [Scroll / +/-]", True, DIM), (14, 76))
    screen.blit(f_sm.render("ESC → Menü", True, (70, 70, 90)), (14, 110))

    # ── Throttle/brake graph (bottom-right) ──────────────────────────────
    if throttle_hist is not None:
        from f1_rl.hud import draw_throttle_graph
        draw_throttle_graph(screen, throttle_hist, x=CANVAS_W - 380, y=CANVAS_H - 170)


def _run_visualization(screen, clock, fonts, model, track) -> str:
    from collections import deque

    from f1_rl.env import F1Env
    from f1_rl.hud import draw_hud, draw_throttle_graph
    from f1_rl.track import meters_to_pixels, GRASS_COLOR

    f_title, f_lg, f_md, f_sm = fonts

    env = F1Env(track=track, render_mode=None)
    obs, _ = env.reset()
    # Load persisted best lap for this circuit
    best_laps = _load_best_laps()
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
                    _save_best_lap(track.name, best_lap)
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
            f"[R] Reset   [T] Menü   [SPACE] Pause   [Scroll/+/-] Zoom {zoom:.1f}×   [0] Reset   [ESC] Quit",
            True, (60, 60, 80),
        )
        screen.blit(hint, (CANVAS_W // 2 - hint.get_width() // 2, CANVAS_H - 20))

        pygame.display.flip()
        clock.tick(FPS)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption("F1 RL Simulator")
    clock = pygame.time.Clock()

    fonts = (
        _font("segoeui", 48, bold=True),
        _font("segoeui", 28, bold=True),
        _font("segoeui", 18),
        _font("segoeui", 14),
    )

    state = "menu"
    sel_idx = 0
    hover_idx = -1
    training_thread: _TrainingThread | None = None
    training_start = 0.0
    loaded_model = None
    loaded_track = None
    zoom = 1.0

    from f1_rl.train import load_training_state
    train_state = load_training_state()

    # Shared queues for live training feedback
    render_queue: queue.Queue = queue.Queue(maxsize=5)
    stats_queue: queue.Queue = queue.Queue(maxsize=10)
    car_state = None       # (x_m, y_m, heading, speed_ms, throttle)
    last_stats: dict = {}

    from collections import deque
    training_throttle_hist: deque = deque(maxlen=120)

    item_h = 44
    list_top = 175

    while True:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if state == "menu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_DOWN:
                        sel_idx = (sel_idx + 1) % len(CIRCUITS)
                    if event.key == pygame.K_UP:
                        sel_idx = (sel_idx - 1) % len(CIRCUITS)
                    if event.key in (pygame.K_RETURN, pygame.K_r):
                        resume = event.key == pygame.K_r and train_state is not None
                        query, fallback, half_width = CIRCUITS[sel_idx]
                        render_queue = queue.Queue(maxsize=5)
                        stats_queue = queue.Queue(maxsize=10)
                        car_state = None
                        last_stats = {}
                        zoom = 1.0
                        training_throttle_hist.clear()
                        training_thread = _TrainingThread(
                            query, fallback, half_width, render_queue, stats_queue,
                            resume=resume,
                        )
                        training_thread.start()
                        training_start = time.time()
                        state = "training"
                    if event.key == pygame.K_l:
                        if os.path.exists(MODEL_PATH):
                            from stable_baselines3 import PPO
                            from f1_rl.track import load_track
                            query, fallback, half_width = CIRCUITS[sel_idx]
                            print(f"[main] Loading track: {query}")
                            loaded_track = load_track(query, geojson_fallback_path=fallback, half_width_m=half_width)
                            print("[main] Loading model...")
                            loaded_model = PPO.load(MODEL_PATH)
                            state = "visualization"
                        else:
                            print("[main] No model found at", MODEL_PATH)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i in range(len(CIRCUITS)):
                        r = pygame.Rect(CANVAS_W // 2 - 230, list_top + i * item_h, 460, item_h - 4)
                        if r.collidepoint(mx, my):
                            sel_idx = i

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

        # Hover
        if state == "menu":
            hover_idx = -1
            for i in range(len(CIRCUITS)):
                r = pygame.Rect(CANVAS_W // 2 - 230, list_top + i * item_h, 460, item_h - 4)
                if r.collidepoint(mx, my):
                    hover_idx = i

        # Drain queues
        if state == "training":
            try:
                while True:
                    car_state = render_queue.get_nowait()
                    training_throttle_hist.append(car_state[4])
            except queue.Empty:
                pass
            try:
                while True:
                    last_stats = stats_queue.get_nowait()
            except queue.Empty:
                pass

        # State transitions
        if state == "training" and training_thread and training_thread.done:
            if training_thread.error:
                print(f"[main] Training error: {training_thread.error}")
                state = "menu"
            else:
                loaded_model = training_thread.model
                loaded_track = training_thread.track
                state = "visualization"
            train_state = load_training_state()
            training_thread = None

        # Rendering
        if state == "menu":
            _draw_menu(screen, fonts, sel_idx, hover_idx, train_state)

        elif state == "training":
            live_track = training_thread.track if training_thread else None
            _draw_training(
                screen, fonts,
                track=live_track,
                car_state=car_state,
                stats=last_stats,
                elapsed=time.time() - training_start,
                zoom=zoom,
                throttle_hist=training_throttle_hist,
            )

        elif state == "visualization":
            result = _run_visualization(screen, clock, fonts, loaded_model, loaded_track)
            if result == "quit":
                pygame.quit()
                sys.exit()
            else:
                state = "menu"
                continue

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
