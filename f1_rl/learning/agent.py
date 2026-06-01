"""Inference wrapper: turn a flat weight vector into a driving policy.

A NeuralAgent holds one network and exposes ``predict(obs)`` returning the
greedy action (the index with the highest Q-value). Used by the live display,
replay recording, and the "load & drive" view.
"""
from __future__ import annotations

import numpy as np
import torch

from f1_rl.learning.network import (
    flat_to_network, n_params, network_to_flat, random_weights,
)

# Inference is single-sample and runs in the UI thread — one thread is plenty
# and avoids torch spawning background threads that fight the render loop.
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

__all__ = ["NeuralAgent", "n_params", "random_weights"]


class NeuralAgent:
    """Wraps a network for action selection; predict() matches the SB3 interface."""

    def __init__(self, weights: np.ndarray):
        self._net = flat_to_network(np.asarray(weights, dtype=np.float32))
        self._net.eval()

    @property
    def weights(self) -> np.ndarray:
        return network_to_flat(self._net)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            action = int(self._net(x).argmax(dim=1).item())
        return action, None

    def save(self, path: str) -> None:
        npy = path if path.endswith(".npy") else path + ".npy"
        np.save(npy, self.weights)

    @classmethod
    def load(cls, path: str) -> "NeuralAgent":
        npy = path if path.endswith(".npy") else path + ".npy"
        return cls(np.load(npy))
