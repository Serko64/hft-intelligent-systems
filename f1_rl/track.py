"""Track loading, shapely corridor, coordinate normalization, and rendering."""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass

import numpy as np
import shapely.ops
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Polygon

HALF_WIDTH_M = 8.0
CANVAS_W = 1600
CANVAS_H = 1000
PAD = 80

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "circuits", "_cache")


@dataclass
class TrackData:
    name: str
    centerline_m: LineString
    corridor: Polygon
    total_length_m: float
    centerline_px: np.ndarray   # (N, 2) float32
    corridor_px: np.ndarray     # (M, 2) float32, exterior ring
    canvas_w: int
    canvas_h: int
    px_scale: float
    px_origin_x: float
    px_origin_y: float          # Y-flipped
    bounds_center: tuple[float, float]   # (cx_m, cy_m)
    half_diag_m: float
    half_width_m: float              # corridor half-width used for OOB check
    corridor_interior_px: np.ndarray | None  # (M, 2) float32, inner ring or None


def _project_linestring(ls: LineString) -> tuple[LineString, Transformer]:
    centroid = ls.centroid
    proj = (
        f"+proj=tmerc +lat_0={centroid.y} +lon_0={centroid.x}"
        " +units=m +datum=WGS84 +no_defs"
    )
    tf = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    return shapely.ops.transform(tf.transform, ls), tf


def _build_track_data(
    name: str,
    centerline_wgs84: LineString,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    half_width_m: float = HALF_WIDTH_M,
    pad: int = PAD,
) -> TrackData:
    centerline_m, _ = _project_linestring(centerline_wgs84)
    corridor = centerline_m.buffer(half_width_m, cap_style=2, join_style=2)

    minx, miny, maxx, maxy = centerline_m.bounds
    span_x = maxx - minx or 1e-9
    span_y = maxy - miny or 1e-9
    usable_w = canvas_w - 2 * pad
    usable_h = canvas_h - 2 * pad
    scale = min(usable_w / span_x, usable_h / span_y)

    origin_x = pad + (usable_w - span_x * scale) / 2 - minx * scale
    origin_y = pad + (usable_h - span_y * scale) / 2 + maxy * scale

    def to_px(coords):
        return np.array(
            [(origin_x + x * scale, origin_y - y * scale) for x, y in coords],
            dtype=np.float32,
        )

    centerline_px = to_px(centerline_m.coords)
    corridor_px = to_px(corridor.exterior.coords)
    corridor_interior_px = to_px(corridor.interiors[0].coords) if corridor.interiors else None

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half_diag = math.hypot(span_x, span_y) / 2 + 1e-6

    return TrackData(
        name=name,
        centerline_m=centerline_m,
        corridor=corridor,
        total_length_m=centerline_m.length,
        centerline_px=centerline_px,
        corridor_px=corridor_px,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        px_scale=scale,
        px_origin_x=origin_x,
        px_origin_y=origin_y,
        bounds_center=(cx, cy),
        half_diag_m=half_diag,
        half_width_m=half_width_m,
        corridor_interior_px=corridor_interior_px,
    )


def load_track_osmnx(
    query: str,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    half_width_m: float = HALF_WIDTH_M,
    pad: int = PAD,
) -> TrackData:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in query)
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import osmnx as ox

    # features_from_address searches within `dist` metres of the geocoded point,
    # which is more reliable than features_from_place for named circuits.
    try:
        gdf = ox.features_from_address(query, tags={"highway": "raceway"}, dist=10_000)
    except Exception:
        gdf = ox.features_from_place(query, tags={"highway": "raceway"})

    geom_rows = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if geom_rows.empty:
        raise ValueError(f"No raceway geometry found for '{query}'")

    parts = []
    for geom in geom_rows.geometry:
        if geom.geom_type == "LineString":
            parts.append(geom)
        else:
            parts.extend(geom.geoms)

    merged = shapely.ops.linemerge(MultiLineString(parts))
    if merged.geom_type == "MultiLineString":
        merged = max(merged.geoms, key=lambda g: g.length)

    track = _build_track_data(query, merged, canvas_w, canvas_h, half_width_m, pad)

    with open(cache_path, "wb") as f:
        pickle.dump(track, f)

    return track


def load_track_geojson(
    path: str,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    half_width_m: float = HALF_WIDTH_M,
    pad: int = PAD,
) -> TrackData:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    feature = data["features"][0]
    geom = feature["geometry"]
    props = feature["properties"]
    name = props.get("Name", os.path.basename(path))

    if geom["type"] == "LineString":
        coords = geom["coordinates"]
    else:
        coords = [pt for part in geom["coordinates"] for pt in part]

    # GeoJSON coords are [lon, lat]
    centerline_wgs84 = LineString(coords)
    return _build_track_data(name, centerline_wgs84, canvas_w, canvas_h, half_width_m, pad)


def load_track(
    query: str,
    geojson_fallback_path: str | None = None,
    **kwargs,
) -> TrackData:
    # Prefer the curated GeoJSON over live OSM data when available.
    if geojson_fallback_path and os.path.exists(geojson_fallback_path):
        print(f"[track] Using GeoJSON: {geojson_fallback_path}")
        return load_track_geojson(geojson_fallback_path, **kwargs)
    return load_track_osmnx(query, **kwargs)


def meters_to_pixels(track: TrackData, x_m: float, y_m: float) -> tuple[float, float]:
    return (
        track.px_origin_x + x_m * track.px_scale,
        track.px_origin_y - y_m * track.px_scale,
    )


_BAKED_TRACK_CACHE: dict[int, object] = {}   # id(track) → pygame.Surface

GRASS_COLOR = (18, 38, 18)


def _draw_kerbs(surface, pts: list, kerb_px: float, thickness: int) -> None:
    import pygame
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


def _bake_track(track: TrackData) -> object:
    """Render track once at 3× resolution, smoothscale to native — gives free AA."""
    import pygame
    S = 3  # supersampling factor

    big = pygame.Surface((track.canvas_w * S, track.canvas_h * S))
    big.fill(GRASS_COLOR)

    def sc(pts):
        return [(x * S, y * S) for x, y in pts]

    pts_out = sc([(float(p[0]), float(p[1])) for p in track.corridor_px])
    pts_ctr = sc([(float(p[0]), float(p[1])) for p in track.centerline_px])
    pts_inn = (
        sc([(float(p[0]), float(p[1])) for p in track.corridor_interior_px])
        if track.corridor_interior_px is not None else None
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

    return pygame.transform.smoothscale(big, (track.canvas_w, track.canvas_h))


def draw_track(surface, track: TrackData) -> None:
    import pygame
    key = id(track)
    if key not in _BAKED_TRACK_CACHE:
        _BAKED_TRACK_CACHE[key] = _bake_track(track)
    surface.blit(_BAKED_TRACK_CACHE[key], (0, 0))
