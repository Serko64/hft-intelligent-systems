"""FastAPI-App: REST-Endpunkte + ein WebSocket, der die Live-Simulation streamt.

Endpunkte
---------
GET  /api/circuits        -> {"circuits": [name, ...]}
GET  /api/track?name=...  -> Streckengeometrie (Centerline + Wände), Meter
WS   /ws                  -> bidirektional:
       Client -> Server (Befehle):
         {"type": "start_training", "circuit", "steps_per_gen", "total_gens",
          "evolution_mode", "resume"}
         {"type": "load_and_drive", "circuit", "backend"}
         {"type": "stop"}
       Server -> Client (Nachrichten):
         {"type": "track", ...geometry}
         {"type": "frame", "cars": [...]}     (~60 fps)
         {"type": "stats", ...}               (pro Generation)
         {"type": "status", "state", "message"?}
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from f1_rl.config import CIRCUITS
from f1_rl.server.protocol import (
    car_to_dict, racing_line_to_dict, stats_to_dict, track_to_dict,
)  # inspect-Dicts werden in Session/Trainer gebaut und unverändert durchgereicht
from f1_rl.server import session
from f1_rl.server.session import SESSION, find_circuit
from f1_rl.utils.queues import get_latest

clients: set[WebSocket] = set()


# ── Broadcasting ───────────────────────────────────────────────────────────────

async def _broadcast(obj: dict) -> None:
    dead = []
    for ws in list(clients):
        try:
            await ws.send_json(obj)
        except Exception:            # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def _pump() -> None:
    """Leert die Session-Queues ~60×/s und schiebt das Neueste an die Clients."""
    last_mode = SESSION["mode"]
    while True:
        await asyncio.sleep(1.0 / 60.0)
        if not clients:
            last_mode = SESSION["mode"]
            continue
        # Vom Client nicht ausgelöste Moduswechsel spiegeln (z. B. Training fertig).
        if SESSION["mode"] != last_mode:
            last_mode = SESSION["mode"]
            await _broadcast({"type": "status", "state": SESSION["mode"]})

        frame = get_latest(SESSION["render_q"])
        if frame is not None:
            await _broadcast({"type": "frame", "cars": [car_to_dict(c) for c in frame]})

        stats = get_latest(SESSION["stats_q"])
        if stats is not None:
            await _broadcast({"type": "stats", **stats_to_dict(stats)})

        racing_line = get_latest(SESSION["line_q"])
        if racing_line is not None:
            await _broadcast({"type": "racing_line", **racing_line_to_dict(racing_line)})

        inspect_data = get_latest(SESSION["inspect_q"])
        if inspect_data is not None:
            await _broadcast({"type": "inspect", **inspect_data})

        table_data = get_latest(SESSION["table_q"])
        if table_data is not None:
            await _broadcast({"type": "qtable", **table_data})


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_pump())
    yield
    task.cancel()


app = FastAPI(title="F1 RL Simulator", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── REST ─────────────────────────────────────────────────────────────────────

@app.get("/api/circuits")
def list_circuits() -> dict:
    return {"circuits": [c[0] for c in CIRCUITS]}


@app.get("/api/track")
def get_track(name: str) -> dict:
    fallback, half_width = find_circuit(name)
    from f1_rl.simulation.track_loader import load_track
    track = load_track(name, geojson_fallback_path=fallback, half_width_m=half_width)
    return track_to_dict(track)


# ── WebSocket ──────────────────────────────────────────────────────────────────

async def _handle(cmd: dict, ws: WebSocket) -> None:
    kind = cmd.get("type")
    if kind == "start_training":
        session.start_training(
            cmd["circuit"],
            int(cmd.get("steps_per_gen", 5000)),
            int(cmd.get("total_gens", 200)),
            cmd.get("evolution_mode", "classic"),
            bool(cmd.get("resume", False)),
            use_rays=bool(cmd.get("use_rays", True)),
            auto_speed=bool(cmd.get("auto_speed", False)),
        )
        await _broadcast({"type": "track", **track_to_dict(SESSION["track"])})
        await _broadcast({"type": "status", "state": "training"})
    elif kind == "load_and_drive":
        try:
            session.start_driving(cmd["circuit"], use_rays=bool(cmd.get("use_rays", True)),
                                  backend=cmd.get("backend", "dqn"))
            await _broadcast({"type": "track", **track_to_dict(SESSION["track"])})
            await _broadcast({"type": "status", "state": "driving"})
        except Exception as e:       # noqa: BLE001
            await ws.send_json({"type": "status", "state": "idle", "message": str(e)})
    elif kind == "set_speed":
        session.set_speed(int(cmd.get("value", 1)))
    elif kind == "inspect_car":
        session.set_inspect(cmd.get("index"))
    elif kind == "stop":
        session.stop_session()
        await _broadcast({"type": "status", "state": "idle"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        # Frisch verbundenen Client auf den aktuellen Stand bringen.
        await ws.send_json({"type": "status", "state": SESSION["mode"]})
        if SESSION["track"] is not None:
            await ws.send_json({"type": "track", **track_to_dict(SESSION["track"])})
        while True:
            cmd = await ws.receive_json()
            await _handle(cmd, ws)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
