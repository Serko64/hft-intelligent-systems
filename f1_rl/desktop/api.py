
from f1_rl.config import CIRCUITS
from f1_rl.server import session
from f1_rl.server.protocol import (
    car_to_dict, racing_line_to_dict, stats_to_dict, track_to_dict,
)
from f1_rl.server.session import SESSION, find_circuit
from f1_rl.simulation.track_loader import load_track
from f1_rl.utils.queues import get_latest


class Api:
    def list_circuits(self) -> list[str]:
        return [c[0] for c in CIRCUITS]

    def get_track(self, name: str) -> dict:
        fallback, half_width = find_circuit(name)
        track = load_track(
            name, geojson_fallback_path=fallback, half_width_m=half_width)
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
            session.start_driving(
                circuit, use_rays=bool(use_rays), backend=backend)
            return track_to_dict(SESSION["track"])
        except Exception as e:
            return {"error": str(e)}

    def set_speed(self, value: int) -> None:
        session.set_speed(int(value))

    def inspect_car(self, index) -> None:
        session.set_inspect(index)

    def stop(self) -> None:
        session.stop_session()

    def poll(self) -> dict:
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

        inspect = get_latest(SESSION["inspect_q"])
        if inspect is not None:
            out["inspect"] = inspect

        qtable = get_latest(SESSION["table_q"])
        if qtable is not None:
            out["qtable"] = qtable

        return out
