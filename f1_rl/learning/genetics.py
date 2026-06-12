"""Genetische Operationen auf flachen Gewichtsvektoren.

Bewusst klein und rein (nur numpy): rein gehen Gewichtsvektoren / Fitnesswerte,
raus kommen neue Gewichtsvektoren. Das eigentliche Training jedes Individuums
passiert in worker.py, die Generationsschleife in trainer.py. Die GA-Mathematik
hier zu bündeln macht jedes Teil für sich gut lesbar.
"""
from __future__ import annotations

import numpy as np


# ── Basic operators ───────────────────────────────────────────────────────────

def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
    """Uniformes Crossover auf flachen Vektoren (jedes Gen von einem Zufallselternteil)."""
    from_parent_a = np.random.random(len(parent_a)) > 0.5   # bool-Maske pro Gewicht
    child = parent_b.copy()
    child[from_parent_a] = parent_a[from_parent_a]
    return child


def mutate(weights: np.ndarray, rate: float, noise: float) -> np.ndarray:
    """Addiert Gauß-Rauschen auf einen Zufallsanteil (`rate`) der Gewichte."""
    child = weights.copy()
    mutation_mask = np.random.rand(len(child)) < rate   # welche Gewichte bekommen Rauschen?
    child[mutation_mask] += np.random.normal(0, noise, mutation_mask.sum()).astype(np.float32)
    return child


def rank_select(weights_list: list) -> np.ndarray:
    """Rang-proportionale Selektion; weights_list muss absteigend sortiert sein."""
    n = len(weights_list)
    probs = np.arange(n, 0, -1, dtype=np.float64)
    probs /= probs.sum()
    return weights_list[np.random.choice(n, p=probs)]
