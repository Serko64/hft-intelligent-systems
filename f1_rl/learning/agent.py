"""Inferenz-Helfer: macht aus einem flachen Gewichtsvektor eine fahrbare Policy.

Eine „Policy" ist hier einfach ein fertiges PyTorch-Netz. Der Rest des Projekts
braucht nur diese kleinen Funktionen:

  neuronal_net_from_weights(w)        -> net   Netz aus einem Gewichtsvektor bauen
  neuronal_net_from_weight_file(path) -> net   dasselbe, Gewichte aus .npy-Datei laden
  act(net, obs)                       -> action  beste Aktion (höchster Q-Wert)
  forward_trace(net, obs)             -> (q_values, hidden)  Detail-Ansicht fürs UI

Genutzt von der Live-Anzeige, der Racing-Line-Aufzeichnung und der „Laden & Fahren"-Ansicht.
"""
from __future__ import annotations

import numpy as np
import torch

from f1_rl.learning.network import flat_to_network

# Inferenz läuft Einzel-Sample im UI-/Server-Thread — ein Thread reicht und
# verhindert, dass torch Hintergrund-Threads startet, die die Renderschleife stören.
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

__all__ = ["neuronal_net_from_weights", "neuronal_net_from_weight_file", "act", "forward_trace"]


# Baut ein neuronales Netz aus einem vorgegebenen Gewichtsvektor.
def neuronal_net_from_weights(weights: np.ndarray) -> torch.nn.Module:
    """Baut aus einem flachen Gewichtsvektor ein inferenzbereites Netz."""
    net = flat_to_network(np.asarray(weights, dtype=np.float32))
    net.eval()
    return net


# Lädt die Netz-Gewichte aus einer Datei.
def neuronal_net_from_weight_file(path: str) -> torch.nn.Module:
    """Lädt einen flachen Gewichtsvektor aus ``path`` (.npy) in ein fertiges Netz."""
    npy = path if path.endswith(".npy") else path + ".npy"
    return neuronal_net_from_weights(np.load(npy))


def act(net: torch.nn.Module, obs: np.ndarray) -> int:
    """Beste Aktion (Index des höchsten Q-Werts) für eine Beobachtung."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return int(net(x).argmax(dim=1).item())


def forward_trace(net: torch.nn.Module, obs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Ein Vorwärtsdurchlauf, gibt (q_values, hidden_activations) zurück.

    q_values ist die Ausgabeschicht (ein Q je Aktion — die DQN-„Q-Tabelle" dieses
    Zustands). hidden_activations ist der Post-ReLU-Aktivierungsvektor jeder
    versteckten Schicht, damit das UI das „Aufleuchten" des Netzes zeigen kann.
    Billig: ein Sample durch ein winziges MLP.
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
