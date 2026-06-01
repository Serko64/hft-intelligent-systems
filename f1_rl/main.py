"""Entry point for the F1 RL Simulator.

Run with:  python -m f1_rl.main   (or)   python f1_rl/main.py

All the real work lives in the sub-packages:
  simulation/  — the track + the gymnasium environment
  learning/    — the network and how it is trained (DQN + genetic algorithm)
  ui/          — the pygame interface

The ``if __name__ == "__main__"`` guard below is required: training spawns
worker processes, and on Windows each one re-imports this module.
"""
from __future__ import annotations

import os
import sys

# Allow `python f1_rl/main.py` (direct file run) to find the top-level package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Training logs use a few unicode characters (≈, ≥, →). Force UTF-8 so they
# never crash on a Windows console that defaults to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from f1_rl.ui.app import main

if __name__ == "__main__":
    main()
