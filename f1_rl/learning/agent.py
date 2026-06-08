"""Inference helpers: turn a flat weight vector into a driving policy.

A "policy" here is just a ready-to-use PyTorch network. A few small functions are
all the rest of the project needs:

  policy_from_weights(w) -> net   build a network from an in-memory weight vector
  load_policy(path)      -> net   same, but load the weights from a .npy file
  act(net, obs)          -> action  pick the greedy action (highest Q-value)

Used by the live display, replay recording, and the "load & drive" view.
"""
from __future__ import annotations

import numpy as np
import torch

from f1_rl.learning.network import flat_to_network

# Inference is single-sample and runs in the UI/server thread — one thread is
# plenty and avoids torch spawning background threads that fight the render loop.
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

__all__ = ["neuronal_net_from_weights", "neuronal_net_from_weight_file", "act", "forward_trace"]


# Definition eines Neuronalen Netzes auf basis von vordefinierten gewichten für die jeweilige instanz
def neuronal_net_from_weights(weights: np.ndarray) -> torch.nn.Module:
    """Build a ready-for-inference network from a flat weight vector."""
    net = flat_to_network(np.asarray(weights, dtype=np.float32))
    net.eval()
    return net


# Abrufen der gewichte für das neuronale netz aus einer Datei
def neuronal_net_from_weight_file(path: str) -> torch.nn.Module:
    """Load a flat weight vector from ``path`` (.npy) into a ready network."""
    npy = path if path.endswith(".npy") else path + ".npy"
    return neuronal_net_from_weights(np.load(npy))


def act(net: torch.nn.Module, obs: np.ndarray) -> int:
    """Return the greedy action (index of the highest Q-value) for one observation."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return int(net(x).argmax(dim=1).item())


def forward_trace(net: torch.nn.Module, obs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run one forward pass and return (q_values, hidden_activations).

    q_values is the output layer (one Q per action — the DQN "Q-table" for this
    state). hidden_activations is the post-ReLU activation vector of each hidden
    layer, so the UI can visualise the network lighting up. Cheap: a single
    sample through a tiny MLP.
    """
    import torch.nn as nn
    hidden: list[np.ndarray] = []
    with torch.no_grad():
        h = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        for layer in net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                hidden.append(h.squeeze(0).cpu().numpy().copy())
        q = h.squeeze(0).cpu().numpy()
    return q, hidden
