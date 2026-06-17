from f1_rl.simulation.environment import REWARD_PARTS, CarFrame

# Streckenrand-Bänder, von der Fahrbahnkante nach AUSSEN gemessen, in Metern.
# Bewusst SCHMAL: große Auslauf-Puffer (15 m) verschmelzen über Geraden und füllen
# auf kompakten Strecken das Infield — sieht nach Klecksen statt Strecke aus. Ein
# schmaler roter Randstein + ein schmaler Asphaltstreifen liest sich als saubere Kante.
_TRACK_ZONES = [
    ("runoff", 4.0),
]


def _poly_rings(poly) -> dict:
    return {
        "exterior": [[float(x), float(y)] for x, y in poly.exterior.coords],
        "interiors": [[[float(x), float(y)] for x, y in r.coords] for r in poly.interiors],
    }


def _geom_polys(geom) -> list:
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def track_to_dict(track) -> dict:
    corr = track["corridor"]
    poly = max(
        corr.geoms, key=lambda g: g.area) if corr.geom_type == "MultiPolygon" else corr
    minx, miny, maxx, maxy = corr.bounds

    # Konzentrische Auslauf-Bänder, äußerstes zuerst (das Frontend stapelt sie, jedes
    # innere liegt oben und lässt das darunter als Ring durchscheinen).
    zones = []
    for name, dist in _TRACK_ZONES:
        band = corr.buffer(dist, join_style=1).simplify(
            0.5, preserve_topology=True)
        zones.append({"name": name, "polygons": [
                     _poly_rings(p) for p in _geom_polys(band)]})
    zones.reverse()  # äußerstes (Bande) zuerst senden

    return {
        "name": track["name"],
        "total_length_m": float(track["total_length_m"]),
        "half_width_m": float(track["half_width_m"]),
        "centerline": [[float(x), float(y)] for x, y in track["centerline_m"].coords],
        "corridor_exterior": [[float(x), float(y)] for x, y in poly.exterior.coords],
        "corridor_interiors": [
            [[float(x), float(y)] for x, y in ring.coords] for ring in poly.interiors
        ],
        "zones": zones,
        "bounds": {"minx": float(minx), "miny": float(miny),
                   "maxx": float(maxx), "maxy": float(maxy)},
    }


def car_to_dict(car: CarFrame) -> dict:
    return {
        "x": float(car["x"]), "y": float(car["y"]), "heading": float(car["heading"]),
        "speed": float(car["speed"]), "throttle": float(car["throttle"]),
        "checkpoint": int(car["checkpoint"]), "progress": float(car["progress"]),
        "lap": int(car["lap"]),
        "rays": [float(r) for r in car["rays"]],
        "score": float(car["score"]),
        "reward_parts": {k: float(v) for k, v in zip(REWARD_PARTS, car["reward_parts"])},
        "generation": int(car["generation"]),
        "a_long": float(car["a_long"]),
        "a_lat": float(car["a_lat"]),
    }


def racing_line_to_dict(points) -> dict:
    pts = [[float(x), float(y), float(v), float(thr)]
           for x, y, v, thr in points]
    speeds = [p[2] for p in pts] or [0.0]
    return {"points": pts, "vmin": min(speeds), "vmax": max(speeds)}


def inspect_to_dict(index: int, obs, q, hidden, action: int) -> dict:
    return {
        "index": int(index),
        "obs": [float(v) for v in obs],
        "q": [float(v) for v in q],
        "hidden": [[float(v) for v in layer] for layer in hidden],
        "action": int(action),
    }


def qtable_to_dict(index: int, values, n_states: int) -> dict:
    return {
        "index": int(index),
        "n_states": int(n_states),
        "values": [[round(float(v), 2) for v in row] for row in values],
    }


def stats_to_dict(stats: dict) -> dict:
    return {
        "timesteps": int(stats.get("timesteps", 0)),
        "generation": int(stats.get("generation", 0)),
        "epsilon": float(stats.get("epsilon", 1.0)),
        "best_fitness": float(stats.get("best_fitness", 0.0)),
        "mean_fitness": float(stats.get("mean_fitness", 0.0)),
        # Scoreboard: [score, generation], best first — das Frontend-HUD erwartet das Feld.
        "top_scores": [[float(s), int(g)] for s, g in stats.get("top_scores", [])],
    }
