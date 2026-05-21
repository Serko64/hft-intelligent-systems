import json
import os
import sys

import pygame

SCREEN_W, SCREEN_H = 1280, 720
PANEL_W = 260
MAP_W = SCREEN_W - PANEL_W
ITEM_H = 28

BG = (12, 12, 18)
PANEL_BG = (20, 20, 30)
SELECTED_BG = (55, 30, 80)
HOVER_BG = (35, 35, 55)
TRACK_COLOR = (220, 40, 40)
START_COLOR = (50, 220, 80)
TEXT = (210, 210, 210)
DIM = (100, 100, 120)
ACCENT = (180, 30, 30)

CIRCUITS_DIR = "circuits"


def load_circuit(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    feature = data["features"][0]
    geom = feature["geometry"]
    if geom["type"] == "LineString":
        coords = geom["coordinates"]
    elif geom["type"] == "MultiLineString":
        coords = [pt for part in geom["coordinates"] for pt in part]
    else:
        coords = []
    props = feature["properties"]
    name = props.get("Name", os.path.basename(path))
    location = props.get("Location", "")
    length = props.get("length", "?")
    first_gp = props.get("firstgp", "?")
    return name, location, length, first_gp, coords


def to_screen_points(coords, area_x, area_y, area_w, area_h, pad=50):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    span_lon = max(lons) - min(lons) or 1e-9
    span_lat = max(lats) - min(lats) or 1e-9
    scale = min((area_w - 2 * pad) / span_lon, (area_h - 2 * pad) / span_lat)
    offset_x = area_x + (area_w - span_lon * scale) / 2
    offset_y = area_y + (area_h - span_lat * scale) / 2
    return [
        (offset_x + (lon - min(lons)) * scale,
         offset_y + (max(lats) - lat) * scale)
        for lon, lat in coords
    ]


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("F1 Circuit Viewer")
    clock = pygame.time.Clock()

    font_sm = pygame.font.SysFont("segoeui", 14)
    font_md = pygame.font.SysFont("segoeui", 16)
    font_lg = pygame.font.SysFont("segoeui", 24, bold=True)
    font_hd = pygame.font.SysFont("segoeui", 13)

    files = sorted(f for f in os.listdir(CIRCUITS_DIR) if f.endswith(".geojson"))
    circuits = []
    for fname in files:
        try:
            circuits.append((fname[:-8], *load_circuit(os.path.join(CIRCUITS_DIR, fname))))
        except Exception:
            pass

    selected = 0
    scroll = 0
    hover_idx = -1
    visible = (SCREEN_H - 60) // ITEM_H

    while True:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    selected = min(selected + 1, len(circuits) - 1)
                    if selected >= scroll + visible:
                        scroll += 1
                elif event.key == pygame.K_UP:
                    selected = max(selected - 1, 0)
                    if selected < scroll:
                        scroll -= 1

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mx < PANEL_W and my >= 60 and event.button == 1:
                    idx = (my - 60) // ITEM_H + scroll
                    if 0 <= idx < len(circuits):
                        selected = idx
                elif event.button == 4:
                    scroll = max(0, scroll - 1)
                elif event.button == 5:
                    scroll = min(max(0, len(circuits) - visible), scroll + 1)

        hover_idx = (my - 60) // ITEM_H + scroll if mx < PANEL_W and my >= 60 else -1

        screen.fill(BG)

        # ── Left panel ──────────────────────────────────────────
        pygame.draw.rect(screen, PANEL_BG, (0, 0, PANEL_W, SCREEN_H))

        hd = font_lg.render("F1 Circuits", True, ACCENT)
        screen.blit(hd, (12, 12))
        pygame.draw.line(screen, (50, 50, 70), (0, 52), (PANEL_W, 52))

        for i in range(visible):
            ci = i + scroll
            if ci >= len(circuits):
                break
            slug, name, location, length, first_gp, coords = circuits[ci]
            y = 60 + i * ITEM_H

            if ci == selected:
                pygame.draw.rect(screen, SELECTED_BG, (0, y, PANEL_W, ITEM_H))
            elif ci == hover_idx:
                pygame.draw.rect(screen, HOVER_BG, (0, y, PANEL_W, ITEM_H))

            screen.blit(font_md.render(slug, True, TEXT), (10, y + 6))

        pygame.draw.line(screen, (60, 60, 90), (PANEL_W - 1, 0), (PANEL_W - 1, SCREEN_H))

        # ── Map area ─────────────────────────────────────────────
        if circuits:
            slug, name, location, length, first_gp, coords = circuits[selected]

            # Info bar
            info = f"{location}   |   {length} m   |   First GP: {first_gp}"
            screen.blit(font_lg.render(name, True, TEXT), (PANEL_W + 20, 12))
            screen.blit(font_hd.render(info, True, DIM), (PANEL_W + 22, 42))
            pygame.draw.line(screen, (40, 40, 60), (PANEL_W, 58), (SCREEN_W, 58))

            if len(coords) > 1:
                pts = to_screen_points(coords, PANEL_W, 65, MAP_W, SCREEN_H - 70)
                ipts = [(int(x), int(y)) for x, y in pts]
                pygame.draw.lines(screen, TRACK_COLOR, False, ipts, 3)
                pygame.draw.circle(screen, START_COLOR, ipts[0], 7)
                pygame.draw.circle(screen, (255, 255, 255), ipts[0], 7, 2)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
