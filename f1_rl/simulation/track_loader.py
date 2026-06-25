import json
import math
import os
import pickle
from typing import TypedDict

import numpy as np
import shapely.ops
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Polygon

HALF_WIDTH_M = 8.0
CANVAS_W = 1600
CANVAS_H = 1000
PAD = 80

CACHE_DIR = os.path.join(os.path.dirname(
    __file__), "..", "..", "circuits", "_cache")
# Cache-Format-Version: hochzählen, wenn sich der Track-Aufbau ändert.
TRACK_CACHE_VERSION = 2


class Track(TypedDict):
    name: str
    centerline_m: LineString    # Streckenmitte in Metern (shapely-Linie)
    corridor: Polygon           # befahrbare Fläche (Centerline ± half_width_m)
    total_length_m: float
    centerline_px: np.ndarray   # (N, 2) float32, Pixel-Koordinaten
    corridor_px: np.ndarray     # (M, 2) float32, äußerer Rand in Pixeln
    canvas_w: int
    canvas_h: int
    px_scale: float             # Umrechnung Meter in Pixel
    px_origin_x: float
    px_origin_y: float          # Y ist gespiegelt
    bounds_center: tuple[float, float]
    half_diag_m: float          # halbe Diagonale der Bounding-Box
    half_width_m: float         # halbe Streckenbreite, für die Crash-Prüfung
    corridor_interior_px: np.ndarray | None


def _project_linestring(line_wgs84: LineString) -> tuple[LineString, Transformer]:
    centroid = line_wgs84.centroid
    projection = (
        f"+proj=tmerc +lat_0={centroid.y} +lon_0={centroid.x}"
        " +units=m +datum=WGS84 +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    return shapely.ops.transform(transformer.transform, line_wgs84), transformer


def _build_track(
    name: str,
    centerline_wgs84: LineString,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    half_width_m: float = HALF_WIDTH_M,
    pad: int = PAD,
) -> Track:
    centerline_m, _ = _project_linestring(centerline_wgs84)
    corridor = centerline_m.buffer(half_width_m, cap_style=2, join_style=2)

    # Skalierung berechnen, damit die Strecke (mit Rand `pad`) auf die Canvas passt.
    min_x, min_y, max_x, max_y = centerline_m.bounds
    span_x = max_x - min_x or 1e-9
    span_y = max_y - min_y or 1e-9
    usable_w = canvas_w - 2 * pad
    usable_h = canvas_h - 2 * pad
    scale = min(usable_w / span_x, usable_h / span_y)

    origin_x = pad + (usable_w - span_x * scale) / 2 - min_x * scale
    origin_y = pad + (usable_h - span_y * scale) / 2 + max_y * scale

    def to_px(coords):
        return np.array(
            [(origin_x + x * scale, origin_y - y * scale) for x, y in coords],
            dtype=np.float32,
        )

    centerline_px = to_px(centerline_m.coords)
    corridor_px = to_px(corridor.exterior.coords)
    corridor_interior_px = to_px(
        corridor.interiors[0].coords) if corridor.interiors else None

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    half_diag = math.hypot(span_x, span_y) / 2 + 1e-6

    return {
        "name": name,
        "centerline_m": centerline_m,
        "corridor": corridor,
        "total_length_m": centerline_m.length,
        "centerline_px": centerline_px,
        "corridor_px": corridor_px,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "px_scale": scale,
        "px_origin_x": origin_x,
        "px_origin_y": origin_y,
        "bounds_center": (center_x, center_y),
        "half_diag_m": half_diag,
        "half_width_m": half_width_m,
        "corridor_interior_px": corridor_interior_px,
    }


def load_track_osmnx(
    query: str,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    half_width_m: float = HALF_WIDTH_M,
    pad: int = PAD,
) -> Track:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in query)
    cache_path = os.path.join(
        CACHE_DIR, f"{safe_name}_v{TRACK_CACHE_VERSION}.pkl")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # Ein kaputter Cache darf die App nie crashen, dann einfach neu bauen.
            print(f"[track] Ignoring unreadable cache {cache_path}: {e}")

    import osmnx as ox

    # features_from_address sucht im Umkreis von `dist` Metern um den geocodierten
    # Punkt, zuverlässiger als features_from_place für benannte Rennstrecken.
    try:
        gdf = ox.features_from_address(
            query, tags={"highway": "raceway"}, dist=10_000)
    except Exception:
        gdf = ox.features_from_place(query, tags={"highway": "raceway"})

    geom_rows = gdf[gdf.geometry.geom_type.isin(
        ["LineString", "MultiLineString"])]
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

    track = _build_track(query, merged, canvas_w, canvas_h, half_width_m, pad)

    with open(cache_path, "wb") as f:
        pickle.dump(track, f)

    return track


def load_track_geojson(
    path: str,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    half_width_m: float = HALF_WIDTH_M,
    pad: int = PAD,
) -> Track:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    feature = data["features"][0]
    geometry = feature["geometry"]
    properties = feature["properties"]
    name = properties.get("Name", os.path.basename(path))

    if geometry["type"] == "LineString":
        coords = geometry["coordinates"]
    else:
        coords = [pt for part in geometry["coordinates"] for pt in part]

    # GeoJSON-Koordinaten sind [lon, lat]
    centerline_wgs84 = LineString(coords)
    return _build_track(name, centerline_wgs84, canvas_w, canvas_h, half_width_m, pad)


def load_track(
    query: str,
    geojson_fallback_path: str | None = None,
    **kwargs,
) -> Track:
    # Die kuratierte GeoJSON-Datei hat Vorrang vor Live-OSM-Daten.
    if geojson_fallback_path and os.path.exists(geojson_fallback_path):
        print(f"[track] Using GeoJSON: {geojson_fallback_path}")
        return load_track_geojson(geojson_fallback_path, **kwargs)
    return load_track_osmnx(query, **kwargs)


def meters_to_pixels(track: Track, x_m: float, y_m: float) -> tuple[float, float]:
    return (
        track["px_origin_x"] + x_m * track["px_scale"],
        track["px_origin_y"] - y_m * track["px_scale"],
    )
