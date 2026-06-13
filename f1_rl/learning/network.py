import numpy as np
import torch
import torch.nn as nn

from f1_rl.config import N_ACTIONS, N_OBS, NET_DROPOUT, NET_HIDDEN


def build_network() -> nn.Sequential:
    sizes = [N_OBS, *NET_HIDDEN, N_ACTIONS]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:           # keine Aktivierung auf der Ausgabeschicht
            layers.append(nn.ReLU())
            if NET_DROPOUT > 0:
                layers.append(nn.Dropout(NET_DROPOUT))
    return nn.Sequential(*layers)


def n_params() -> int:
    return sum(p.numel() for p in build_network().parameters())


def random_weights() -> np.ndarray:
    net = build_network()
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    return network_to_flat(net)


def flat_to_network(flat: np.ndarray, net: nn.Module | None = None,
                    device: str | torch.device = "cpu") -> nn.Module:
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
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear):
                parts.append(module.weight.detach().cpu().numpy().ravel())
                parts.append(module.bias.detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)
