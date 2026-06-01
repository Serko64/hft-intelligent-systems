"""Web backend: a FastAPI server that exposes the RL simulation over HTTP + WebSocket.

This layer is a drop-in alternative to the pygame `ui/` package. It does NOT
touch `simulation/` or `learning/` — it reuses `trainer.train()` (with its
render/stats queues) and `F1Env` exactly as the pygame UI does, and forwards
the same per-frame data to a browser front-end (see ../../web) over a WebSocket.

Run with:  uvicorn f1_rl.server.app:app --reload   (or python -m f1_rl.server)
"""
