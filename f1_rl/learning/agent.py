import numpy as np
import torch

from f1_rl.learning.network import flat_to_network

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

__all__ = ["neural_net_from_weights",
           "load_neural_net", "act", "forward_trace"]


def neural_net_from_weights(weights: np.ndarray) -> torch.nn.Module:
    net = flat_to_network(np.asarray(weights, dtype=np.float32))
    net.eval()
    return net


def load_neural_net(path: str) -> torch.nn.Module:
    npy = path if path.endswith(".npy") else path + ".npy"
    return neural_net_from_weights(np.load(npy))


def act(net: torch.nn.Module, obs: np.ndarray) -> int:
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return int(net(x).argmax(dim=1).item())


def forward_trace(net: torch.nn.Module, obs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
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
