"""Die JS-zugewandte API, die pywebview an das React-Frontend exponiert.

Jede Methode entspricht 1:1 einem der alten WebSocket-Befehle und delegiert
einfach an das bestehende Session-Modul (f1_rl/server/session.py). Kein
Netzwerk, kein asyncio: JavaScript ruft diese Methoden direkt über
``window.pywebview.api.*`` auf. Die Methoden laufen auf einem pywebview-Thread,
die Simulation auf ihrem eigenen Thread — genau wie zuvor.

Datenfluss zum Frontend: statt eines Servers, der Frames pusht, *zieht* das UI
über :meth:`Api.poll` (~60×/s). Das passt exakt zur "nur das Neueste zählt"-
Semantik der Session-Queues (siehe utils/queues.py).
"""
from __future__ import annotations

from f1_rl.config import CIRCUITS
from f1_rl.server import session
from f1_rl.server.protocol import (
    car_to_dict, racing_line_to_dict, stats_to_dict, track_to_dict,
)
from f1_rl.server.session import SESSION, find_circuit
from f1_rl.simulation.track_loader import load_track
from f1_rl.utils.queues import get_latest


class Api:
    # ── Befehle (Client -> Simulation) ───────────────────────────────────────

    def list_circuits(self) -> list[str]:
        return [c[0] for c in CIRCUITS]

    def get_track(self, name: str) -> dict:
        """Streckengeometrie für die Vorschau (ersetzt GET /api/track)."""
        fallback, half_width = find_circuit(name)
        track = load_track(name, geojson_fallback_path=fallback, half_width_m=half_width)
        return track_to_dict(track)

    def start_training(self, cfg: dict) -> dict:
        session.start_training(
            cfg["circuit"],
            int(cfg.get("steps_per_gen", 5000)),
            int(cfg.get("total_gens", 200)),
            cfg.get("evolution_mode", "classic"),
            bool(cfg.get("resume", False)),
            use_rays=bool(cfg.get("use_rays", True)),
            auto_speed=bool(cfg.get("auto_speed", False)),
        )
        return track_to_dict(SESSION["track"])

    def load_and_drive(self, circuit: str, use_rays: bool = True,
                       backend: str = "dqn") -> dict:
        try:
            session.start_driving(circuit, use_rays=bool(use_rays), backend=backend)
            return track_to_dict(SESSION["track"])
        except Exception as e:               # noqa: BLE001
            # z.B. noch kein trainiertes Modell vorhanden — Fehler ans UI zurück.
            return {"error": str(e)}

    def set_speed(self, value: int) -> None:
        session.set_speed(int(value))

    def inspect_car(self, index) -> None:
        session.set_inspect(index)

    def stop(self) -> None:
        session.stop_session()

    # ── Polling (Simulation -> Client) ───────────────────────────────────────

    def poll(self) -> dict:
        """Liefert den neuesten Stand aus den Session-Queues.

        Das Frontend ruft dies ~60×/s (requestAnimationFrame) auf. Entspricht
        dem alten asyncio-``_pump``, nur pull-basiert: pro Queue wird nur der
        frischeste Eintrag zurückgegeben. ``status`` ist immer dabei; alle
        anderen Schlüssel nur, wenn neue Daten vorliegen.
        """
        out: dict = {"status": SESSION["mode"]}

        frame = get_latest(SESSION["render_q"])
        if frame is not None:
            out["frame"] = [car_to_dict(c) for c in frame]

        stats = get_latest(SESSION["stats_q"])
        if stats is not None:
            out["stats"] = stats_to_dict(stats)

        racing_line = get_latest(SESSION["line_q"])
        if racing_line is not None:
            out["racing_line"] = racing_line_to_dict(racing_line)

        # inspect_q / table_q enthalten bereits fertige Dicts (siehe session.py
        # und Trainer) — unverändert durchreichen.
        inspect = get_latest(SESSION["inspect_q"])
        if inspect is not None:
            out["inspect"] = inspect

        qtable = get_latest(SESSION["table_q"])
        if qtable is not None:
            out["qtable"] = qtable

        return out
