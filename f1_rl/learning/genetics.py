"""Genetic-algorithm operations on flat weight vectors.

These functions are deliberately small and pure (numpy only): they take weight
vectors / fitnesses in, and return new weight vectors out. The actual training
of each individual happens in worker.py; the generation loop lives in
trainer.py. Keeping the GA maths here makes each piece easy to read on its own.
"""
from __future__ import annotations

import numpy as np


# ── Basic operators ───────────────────────────────────────────────────────────

def crossover(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """Uniform crossover on flat weight vectors (each gene comes from a random parent)."""
    mask = np.random.random(len(wa)) > 0.5
    child = wb.copy()
    child[mask] = wa[mask]
    return child


def mutate(weights: np.ndarray, rate: float, noise: float) -> np.ndarray:
    """Add Gaussian noise to a random fraction (`rate`) of the weights."""
    child = weights.copy()
    mask  = np.random.rand(len(child)) < rate
    child[mask] += np.random.normal(0, noise, mask.sum()).astype(np.float32)
    return child


def rank_select(weights_list: list) -> np.ndarray:
    """Rank-proportional selection; weights_list must be sorted descending."""
    n = len(weights_list)
    probs = np.arange(n, 0, -1, dtype=np.float64)
    probs /= probs.sum()
    return weights_list[np.random.choice(n, p=probs)]


# ── Pack / swarm evolution ──────────────────────────────────────────────────────

def assign_packs(descriptors: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """Cluster individuals into *k* packs via a tiny numpy k-means.

    *descriptors* is (N, D) — a behavioural fingerprint per individual (e.g. where it
    ended up + how well it scored). Columns are z-normalised first so position and
    fitness contribute on comparable scales. Returns an (N,) array of pack ids in
    [0, k).  Cars that drive similarly land in the same pack, so packs are visually
    coherent groups in the live view.
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
        d = np.linalg.norm(norm[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = norm[mask].mean(axis=0)
    return labels


def _select_in_pack(weights_out: list, members: np.ndarray) -> np.ndarray:
    """Rank-select within one pack. *members* are global indices in fitness-desc order."""
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
    """Speciated (pack) breeding — group selection so weak DNA can survive via its pack.

    *weights_out* / *fitnesses* are sorted globally descending; *pack_ids* is aligned.

    Mechanics:
      1. Effective fitness lifts weak members toward their pack's best
         (``eff = own + pack_support·max(0, pack_best − own)``) — the headline:
         poor DNA in a strong pack survives.
      2. Each pack keeps its top ``min_survivors`` unchanged → no pack goes extinct
         abruptly, preserving diversity.
      3. Remaining slots go to packs ∝ fitness-shared pack strength (sum(eff)/size),
         so large packs don't dominate.
      4. Breeding is within-pack crossover, with ``migration_rate`` cross-pack gene flow.
      5. The all-time best + top hall-of-fame elites are always carried over.
    """
    fit = np.asarray(fitnesses, dtype=np.float64)
    pack_ids = np.asarray(pack_ids)
    uniq = np.unique(pack_ids)

    pack_best = {int(p): fit[pack_ids == p].max() for p in uniq}
    eff = np.array([f + pack_support * max(0.0, pack_best[int(p)] - f)
                    for f, p in zip(fit, pack_ids)])

    new_pop: list = []

    # ── Global elites: never lose the all-time best / top hall of fame ──
    if best_ever_w is not None:
        new_pop.append(best_ever_w.copy())
    for w in hof_weights[:2]:
        if len(new_pop) < n_pop:
            new_pop.append(w.copy())

    # ── Per-pack survivors (members preserve global fitness-desc order) ──
    pack_members = {int(p): np.where(pack_ids == p)[0] for p in uniq}
    for members in pack_members.values():
        for idx in members[:min_survivors]:
            if len(new_pop) < n_pop:
                new_pop.append(weights_out[idx].copy())

    # ── Allocate remaining slots to packs ∝ fitness-shared strength ──
    strengths = np.array([eff[m].sum() / len(m) for m in pack_members.values()])
    strengths = strengths - strengths.min() + 1e-6
    pack_probs = strengths / strengths.sum()
    pack_keys = list(pack_members.keys())

    while len(new_pop) < n_pop:
        if np.random.random() < migration_rate and len(pack_keys) > 1:
            # Cross-pack crossover — gene flow between packs
            ia, ib = np.random.choice(len(pack_keys), 2, replace=False)
            pa = _select_in_pack(weights_out, pack_members[pack_keys[ia]])
            pb = _select_in_pack(weights_out, pack_members[pack_keys[ib]])
        else:
            p = pack_keys[np.random.choice(len(pack_keys), p=pack_probs)]
            members = pack_members[p]
            pa = _select_in_pack(weights_out, members)
            pb = _select_in_pack(weights_out, members)
        child = crossover(pa, pb)
        child = mutate(child, mutation_rate, mutation_noise)
        new_pop.append(child)

    return new_pop[:n_pop]
