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


# ── Pack / swarm evolution ──────────────────────────────────────────────────────

def assign_packs(descriptors: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """Clustert Individuen in *k* Packs per winzigem numpy-k-Means.

    *descriptors* ist (N, D) — ein Verhaltens-Fingerabdruck pro Individuum (z. B.
    Endposition + Score). Spalten werden zuerst z-normiert, damit Position und
    Fitness auf vergleichbaren Skalen beitragen. Gibt ein (N,)-Array mit Pack-IDs
    in [0, k) zurück. Ähnlich fahrende Autos landen im selben Pack, sodass Packs
    in der Live-Ansicht zusammenhängende Gruppen sind.
    """
    desc = np.asarray(descriptors, dtype=np.float64)
    n = len(desc)
    k = max(1, min(k, n))
    if k == 1 or n == 0:
        return np.zeros(n, dtype=np.int64)

    std = desc.std(axis=0)
    std[std < 1e-9] = 1.0
    norm = (desc - desc.mean(axis=0)) / std

    centers = norm[np.random.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # Abstand jedes Individuums zu jedem Cluster-Zentrum, alle auf einmal:
        # norm hat Shape (n, D), centers (k, D). Durch das Einfügen leerer Achsen
        # ([:, None, :] bzw. [None, :, :]) rechnet numpy die Differenz für jede
        # (Individuum, Zentrum)-Kombination → Ergebnis-Shape (n, k).
        distances = np.linalg.norm(norm[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        new_labels = distances.argmin(axis=1)   # jedem Individuum das nächste Zentrum
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Jedes Zentrum auf den Mittelwert seiner Mitglieder verschieben.
        for cluster in range(k):
            members = labels == cluster
            if members.any():
                centers[cluster] = norm[members].mean(axis=0)
    return labels


def _select_in_pack(weights_out: list, members: np.ndarray) -> np.ndarray:
    """Rang-Selektion innerhalb eines Packs. *members* = globale Indizes in Fitness-desc-Reihenfolge."""
    sub = [weights_out[i] for i in members]
    return rank_select(sub)


def breed_packs(
    weights_out: list,
    fitnesses: list,
    pack_ids: np.ndarray,
    hof_weights: list,
    best_ever_w: np.ndarray | None,
    n_pop: int,
    *,
    pack_support: float,
    migration_rate: float,
    min_survivors: int,
    mutation_rate: float,
    mutation_noise: float,
) -> list:
    """Speziierte (Pack-)Zucht — Gruppen-Selektion, sodass schwache DNA über ihr Pack überlebt.

    *weights_out* / *fitnesses* sind global absteigend sortiert; *pack_ids* ist gleich ausgerichtet.

    Mechanik:
      1. Effektive Fitness hebt schwache Mitglieder zum Pack-Besten
         (``eff = own + pack_support·max(0, pack_best − own)``) — Kernidee:
         schwache DNA in einem starken Pack überlebt.
      2. Jedes Pack behält seine Top-``min_survivors`` unverändert → kein Pack
         stirbt abrupt aus, Vielfalt bleibt erhalten.
      3. Restplätze gehen an Packs ∝ fitness-geteilter Pack-Stärke (sum(eff)/size),
         damit große Packs nicht dominieren.
      4. Zucht ist Crossover innerhalb des Packs, mit ``migration_rate`` Genfluss zwischen Packs.
      5. Allzeit-Beste + Top-Hall-of-Fame-Eliten werden immer übernommen.
    """
    fitness_arr = np.asarray(fitnesses, dtype=np.float64)
    pack_ids = np.asarray(pack_ids)
    unique_packs = np.unique(pack_ids)

    # Bester Score je Pack, dann "effektive" Fitness: schwache Mitglieder werden
    # anteilig (pack_support) zum Pack-Besten hochgezogen.
    pack_best = {int(p): fitness_arr[pack_ids == p].max() for p in unique_packs}
    effective_fitness = np.array(
        [own + pack_support * max(0.0, pack_best[int(p)] - own)
         for own, p in zip(fitness_arr, pack_ids)])

    new_pop: list = []

    # ── Globale Eliten: Allzeit-Beste / Top-Hall-of-Fame nie verlieren ──
    if best_ever_w is not None:
        new_pop.append(best_ever_w.copy())
    for elite_weights in hof_weights[:2]:
        if len(new_pop) < n_pop:
            new_pop.append(elite_weights.copy())

    # ── Pro-Pack-Überlebende (Mitglieder behalten globale Fitness-desc-Reihenfolge) ──
    pack_members = {int(p): np.where(pack_ids == p)[0] for p in unique_packs}
    for members in pack_members.values():
        for member_idx in members[:min_survivors]:
            if len(new_pop) < n_pop:
                new_pop.append(weights_out[member_idx].copy())

    # ── Restplätze den Packs ∝ fitness-geteilter Stärke zuteilen ──
    strengths = np.array([effective_fitness[members].sum() / len(members)
                          for members in pack_members.values()])
    strengths = strengths - strengths.min() + 1e-6
    pack_probs = strengths / strengths.sum()
    pack_keys = list(pack_members.keys())

    while len(new_pop) < n_pop:
        if np.random.random() < migration_rate and len(pack_keys) > 1:
            # Crossover zwischen Packs — Genfluss
            idx_a, idx_b = np.random.choice(len(pack_keys), 2, replace=False)
            parent_a = _select_in_pack(weights_out, pack_members[pack_keys[idx_a]])
            parent_b = _select_in_pack(weights_out, pack_members[pack_keys[idx_b]])
        else:
            chosen_pack = pack_keys[np.random.choice(len(pack_keys), p=pack_probs)]
            members = pack_members[chosen_pack]
            parent_a = _select_in_pack(weights_out, members)
            parent_b = _select_in_pack(weights_out, members)
        child = crossover(parent_a, parent_b)
        child = mutate(child, mutation_rate, mutation_noise)
        new_pop.append(child)

    return new_pop[:n_pop]
