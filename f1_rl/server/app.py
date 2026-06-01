"""FastAPI app: REST endpoints + a WebSocket that streams the live simulation.

Endpoints
---------
GET  /api/circuits        -> {"circuits": [name, ...]}
GET  /api/track?name=...  -> track geometry (centerline + walls), metres
WS   /ws                  -> bidirectional:
       client -> server commands:
         {"type": "start_training", "circuit", "steps_per_gen", "total_gens",
          "evolution_mode", "resume"}
         {"type": "load_and_drive", "circuit"}
         {"type": "stop"}
       server -> client messages:
         {"type": "track", ...geometry}
         {"type": "frame", "cars": [...]}     (~60 fps)
         {"type": "stats", ...}               (per generation)
         {"type": "status", "state", "message"?}
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from f1_rl.config import CIRCUITS
from f1_rl.server.protocol import car_to_dict, stats_to_dict, track_to_dict
from f1_rl.server.session import Session, _find_circuit

session = Session()
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
    """Drain the session queues ~60×/s and push the newest frame/stats to clients."""
    last_mode = session.mode
    while True:
        await asyncio.sleep(1.0 / 60.0)
        if not clients:
            last_mode = session.mode
            continue
        # Reflect mode changes the client didn't trigger (e.g. training finished).
        if session.mode != last_mode:
            last_mode = session.mode
            await _broadcast({"type": "status", "state": session.mode})
        if session.render_q is not None:
            frame = None
            try:
                while True:
                    frame = session.render_q.get_nowait()
            except Exception:        # noqa: BLE001
                pass
            if frame is not None:
                await _broadcast({"type": "frame", "cars": [car_to_dict(c) for c in frame]})
        if session.stats_q is not None:
            stats = None
            try:
                while True:
                    stats = session.stats_q.get_nowait()
            except Exception:        # noqa: BLE001
                pass
            if stats is not None:
                await _broadcast({"type": "stats", **stats_to_dict(stats)})


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
    fallback, hw = _find_circuit(name)
    from f1_rl.simulation.track_loader import load_track
    track = load_track(name, geojson_fallback_path=fallback, half_width_m=hw)
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
        )
        await _broadcast({"type": "track", **track_to_dict(session.track)})
        await _broadcast({"type": "status", "state": "training"})
    elif kind == "load_and_drive":
        try:
            session.start_driving(cmd["circuit"], use_rays=bool(cmd.get("use_rays", True)))
            await _broadcast({"type": "track", **track_to_dict(session.track)})
            await _broadcast({"type": "status", "state": "driving"})
        except Exception as e:       # noqa: BLE001
            await ws.send_json({"type": "status", "state": "idle", "message": str(e)})
    elif kind == "stop":
        session.stop()
        await _broadcast({"type": "status", "state": "idle"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        # Bring a freshly-connected client up to speed.
        await ws.send_json({"type": "status", "state": session.mode})
        if session.track is not None:
            await ws.send_json({"type": "track", **track_to_dict(session.track)})
        while True:
            cmd = await ws.receive_json()
            await _handle(cmd, ws)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
