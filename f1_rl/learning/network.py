"""The one and only network definition — a small PyTorch MLP.

Everything in the project uses *this* network:
  - worker.py trains it with gradient descent (Double DQN),
  - agent.py runs it to choose actions,
  - the genetic algorithm stores its weights as a single flat float32 vector,
    which makes crossover and mutation simple array operations.

Two tiny helpers convert between the PyTorch module and that flat vector:
  flat_to_network(flat) -> nn.Module        (load weights into a network)
  network_to_flat(net)  -> np.ndarray       (read weights out as a vector)

Keeping a single definition here means there is no second hand-written copy of
the network to keep in sync — change the architecture in config.NET_HIDDEN and
the whole project follows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from f1_rl.config import N_ACTIONS, N_OBS, NET_HIDDEN


def build_network() -> nn.Sequential:
    """Create a fresh network: N_OBS -> *NET_HIDDEN -> N_ACTIONS, with ReLU between layers."""
    sizes = [N_OBS, *NET_HIDDEN, N_ACTIONS]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:           # no activation on the output layer
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def n_params() -> int:
    """Total number of weights+biases in one network (depends on config)."""
    return sum(p.numel() for p in build_network().parameters())


def random_weights() -> np.ndarray:
    """A fresh, randomly initialised weight vector (Xavier-uniform, zero biases)."""
    net = build_network()
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    return network_to_flat(net)


def flat_to_network(flat: np.ndarray, net: nn.Module | None = None,
                    device: str | torch.device = "cpu") -> nn.Module:
    """Load a flat weight vector into a network (creating one if not given)."""
    if net is None:
        net = build_network()
    net = net.to(device)
    flat = np.asarray(flat, dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear):
                fo, fi = module.weight.shape
                module.weight.copy_(
                    torch.from_numpy(flat[idx:idx + fo * fi].reshape(fo, fi))
                )
                idx += fo * fi
                module.bias.copy_(torch.from_numpy(flat[idx:idx + fo]))
                idx += fo
    return net


def network_to_flat(net: nn.Module) -> np.ndarray:
    """Read a network's weights out as a single flat float32 vector."""
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear):
                parts.append(module.weight.detach().cpu().numpy().ravel())
                parts.append(module.bias.detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)
