import numpy as np

# Genetische Operatoren auf flachen Gewichtsvektoren. Das Pendant für Q-Tabellen
# liegt in qtable.py, die Idee ist dieselbe.


def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
    # Uniformes Crossover: pro Gewicht entscheidet ein Münzwurf, von welchem
    # Elternteil der Wert übernommen wird.
    from_parent_a = np.random.random(
        len(parent_a)) > 0.5   # bool-Maske pro Gewicht
    child = parent_b.copy()
    child[from_parent_a] = parent_a[from_parent_a]
    return child


def mutate(weights: np.ndarray, rate: float, noise: float) -> np.ndarray:
    # Ein Bruchteil rate der Gewichte bekommt Gauß-Rauschen der Stärke noise dazu.
    child = weights.copy()
    mutation_mask = np.random.rand(len(child)) < rate
    child[mutation_mask] += np.random.normal(
        0, noise, mutation_mask.sum()).astype(np.float32)
    return child


def rank_select(weights_list: list):
    # Rang-Selektion auf einer absteigend sortierten Liste: der vorderste Eintrag wird
    # am wahrscheinlichsten gezogen, der hinterste am seltensten. Anders als bei der
    # Fitness-proportionalen Auswahl zählt nur die Reihenfolge, nicht der Abstand der Scores.
    n = len(weights_list)
    probs = np.arange(n, 0, -1, dtype=np.float64)
    probs /= probs.sum()
    return weights_list[np.random.choice(n, p=probs)]
