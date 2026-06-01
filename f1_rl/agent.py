"""Neural Q-network agent: architecture, weight helpers, and predict interface."""
from __future__ import annotations

import numpy as np

from .config import N_OBS, N_ACTIONS, NET_HIDDEN


# ── Network ───────────────────────────────────────────────────────────────────

class QNetwork:
    """Lightweight MLP implemented in pure numpy — no PyTorch needed at import time.

    Layout: N_OBS → hidden[0] → hidden[1] → N_ACTIONS  (ReLU activations).
    Weights are kept as a single flat float32 array for easy GA crossover/mutation.
    """

    def __init__(self, weights: np.ndarray | None = None):
        # Build layer size list: input → *hidden → output
        sizes = [N_OBS] + list(NET_HIDDEN) + [N_ACTIONS]
        self._shapes: list[tuple] = []   # (W_shape, b_shape) per layer
        n = 0
        for i in range(len(sizes) - 1):
            self._shapes.append(((sizes[i + 1], sizes[i]), (sizes[i + 1],)))
            n += sizes[i + 1] * sizes[i] + sizes[i + 1]
        self._n_params = n

        if weights is None:
            self._w = _xavier_weights(sizes)
        else:
            assert len(weights) == n, f"Expected {n} weights, got {len(weights)}"
            self._w = weights.astype(np.float32)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N_OBS,) → q-values: (N_ACTIONS,)"""
        h = x.astype(np.float32)
        idx = 0
        for k, (ws, bs) in enumerate(self._shapes):
            W = self._w[idx:idx + ws[0] * ws[1]].reshape(ws)
            idx += ws[0] * ws[1]
            b = self._w[idx:idx + bs[0]]
            idx += bs[0]
            h = W @ h + b
            if k < len(self._shapes) - 1:
                h = np.maximum(h, 0)  # ReLU (skip on output layer)
        return h

    def q_action(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.forward(obs)))

    @property
    def weights(self) -> np.ndarray:
        return self._w

    @weights.setter
    def weights(self, w: np.ndarray) -> None:
        self._w = w.astype(np.float32)

    @property
    def n_params(self) -> int:
        return self._n_params


def _xavier_weights(sizes: list[int]) -> np.ndarray:
    """Xavier-uniform init for all layers, concatenated into a flat array."""
    parts = []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = np.random.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)
        b = np.zeros(fan_out, dtype=np.float32)
        parts.append(W.ravel())
        parts.append(b)
    return np.concatenate(parts)


def n_params() -> int:
    """Total number of network parameters (depends on NET_HIDDEN and N_OBS/N_ACTIONS)."""
    return QNetwork().n_params


def random_weights() -> np.ndarray:
    return QNetwork().weights


# ── Agent wrapper ─────────────────────────────────────────────────────────────

class NeuralAgent:
    """Wraps a QNetwork; predict() matches the visualiser interface."""

    def __init__(self, weights: np.ndarray):
        self._net = QNetwork(weights)

    @property
    def weights(self) -> np.ndarray:
        return self._net.weights

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._net.q_action(obs), None

    def save(self, path: str) -> None:
        npy = path if path.endswith(".npy") else path + ".npy"
        np.save(npy, self._net.weights)

    @classmethod
    def load(cls, path: str) -> "NeuralAgent":
        npy = path if path.endswith(".npy") else path + ".npy"
        return cls(np.load(npy))
