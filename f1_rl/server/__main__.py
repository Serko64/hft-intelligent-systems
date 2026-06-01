"""Run the web backend:  python -m f1_rl.server

Serves the API + WebSocket on http://127.0.0.1:8000 . The React front-end
(in ../../web) connects to it. Use --reload during development.
"""
from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("f1_rl.server.app:app", host="127.0.0.1", port=8000, reload=False)
