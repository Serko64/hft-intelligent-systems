"""Web backend: a FastAPI server that exposes the RL simulation over HTTP + WebSocket.

This layer does NOT touch `simulation/` or `learning/` internals — it reuses
`trainer.train()` / `qtable.q_learning_loop()` (with their render/stats queues)
and the functional car environment, and forwards the same per-frame data to a
browser front-end (see ../../web) over a WebSocket.

Run with:  uvicorn f1_rl.server.app:app --reload   (or python -m f1_rl.server)
"""
