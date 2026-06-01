"""Genetic operations and the parallel DQN+GA worker process.

Architecture
------------
Each worker:
  1. Inherits weights from a parent (GA selection).
  2. Builds a PyTorch network from those weights.
  3. Runs DDQN training with a pre-allocated numpy replay buffer.
  4. Returns updated weights → GA crossover/mutation operates on these.

GPU is used automatically when CUDA is available (install NVIDIA driver to enable).
On CPU the pre-allocated numpy buffer avoids Python list overhead.
"""
from __future__ import annotations

import numpy as np

from .config import (
    BATCH_SIZE, BATCH_SIZE_GPU, GAMMA, GRAD_CLIP, LR,
    MUTATION_NOISE, MUTATION_RATE,
    REPLAY_CAPACITY, TARGET_UPDATE_FREQ, TRAIN_FREQ,
)


# ── Genetic operations ────────────────────────────────────────────────────────

def crossover(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """Uniform crossover on flat weight vectors."""
    mask = np.random.random(len(wa)) > 0.5
    child = wb.copy()
    child[mask] = wa[mask]
    return child


def mutate(weights: np.ndarray, rate: float, noise: float) -> np.ndarray:
    child = weights.copy()
    mask  = np.random.rand(len(child)) < rate
    child[mask] += np.random.normal(0, noise, mask.sum()).astype(np.float32)
    return child


def rank_select(weights_list: list, fitnesses: list | None = None) -> np.ndarray:
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


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple[float, np.ndarray, float, float]:
    """Hybrid DDQN + GA worker.

    CPU optimisations
    ~~~~~~~~~~~~~~~~~
    - Pre-allocated numpy arrays for the replay buffer (avoids repeated
      Python list allocations and zip/unpack overhead on every sample).
    - Batch sampled as contiguous numpy slices → single torch.from_numpy call.

    GPU support
    ~~~~~~~~~~~
    Auto-detects CUDA. When a driver is installed torch.cuda.is_available()
    returns True and tensors are moved to the GPU automatically.  A larger
    batch size (BATCH_SIZE_GPU) is used to saturate GPU throughput.
    """
    inherited_weights, track, n_steps, epsilon, eval_steps = args

    # ── Deferred imports (spawn-safe on Windows) ──────────────────────────
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from f1_rl.config import N_OBS, N_ACTIONS, NET_HIDDEN
    from f1_rl.env import F1Env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE

    # ── Build network from inherited weights ──────────────────────────────
    sizes = [N_OBS] + list(NET_HIDDEN) + [N_ACTIONS]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    online_net = nn.Sequential(*layers).to(device)

    idx = 0
    with torch.no_grad():
        for module in online_net.modules():
            if isinstance(module, nn.Linear):
                fo, fi = module.weight.shape
                module.weight.copy_(
                    torch.from_numpy(inherited_weights[idx:idx + fo * fi].reshape(fo, fi))
                )
                idx += fo * fi
                module.bias.copy_(torch.from_numpy(inherited_weights[idx:idx + fo]))
                idx += fo

    target_net = copy.deepcopy(online_net)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(online_net.parameters(), lr=LR)

    # ── Pre-allocated numpy replay buffer ─────────────────────────────────
    # Storing flat arrays instead of a list of tuples eliminates repeated
    # Python object creation and zip/unpack overhead on every sample call.
    buf_s  = np.zeros((REPLAY_CAPACITY, N_OBS), dtype=np.float32)
    buf_ns = np.zeros((REPLAY_CAPACITY, N_OBS), dtype=np.float32)
    buf_a  = np.zeros(REPLAY_CAPACITY, dtype=np.int64)
    buf_r  = np.zeros(REPLAY_CAPACITY, dtype=np.float32)
    buf_d  = np.zeros(REPLAY_CAPACITY, dtype=np.float32)
    buf_ptr  = 0
    buf_size = 0

    env = F1Env(track=track, render_mode=None)
    obs, _ = env.reset()

    # ── Training phase ────────────────────────────────────────────────────
    online_net.train()
    for step in range(n_steps):
        if np.random.random() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            with torch.no_grad():
                q = online_net(torch.from_numpy(obs).to(device).unsqueeze(0))
                action = int(q.argmax().item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buf_s[buf_ptr]  = obs
        buf_ns[buf_ptr] = next_obs
        buf_a[buf_ptr]  = action
        buf_r[buf_ptr]  = reward
        buf_d[buf_ptr]  = float(done)
        buf_ptr  = (buf_ptr + 1) % REPLAY_CAPACITY
        buf_size = min(buf_size + 1, REPLAY_CAPACITY)

        obs = next_obs
        if done:
            obs, _ = env.reset()

        # Gradient update — every TRAIN_FREQ steps
        if buf_size >= batch_size and step % TRAIN_FREQ == 0:
            idx_b = np.random.choice(buf_size, batch_size, replace=False)

            s_t  = torch.from_numpy(buf_s[idx_b]).to(device)
            ns_t = torch.from_numpy(buf_ns[idx_b]).to(device)
            a_t  = torch.from_numpy(buf_a[idx_b]).to(device)
            r_t  = torch.from_numpy(buf_r[idx_b]).to(device)
            d_t  = torch.from_numpy(buf_d[idx_b]).to(device)

            with torch.no_grad():
                # Double DQN: online selects action, target evaluates value
                next_a = online_net(ns_t).argmax(dim=1)
                q_next = target_net(ns_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                td_target = r_t + GAMMA * (1.0 - d_t) * q_next

            q_pred = online_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
            optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

    # ── Greedy evaluation phase (single episode) ─────────────────────────
    # Score = reward from one clean run from the start — directly reflects
    # how far the agent gets, comparable to what the live display shows.
    # eval_steps is budgeted to the track length so a full lap (and its goal
    # bonus) can actually be reached; the loop still breaks early on crash/lap.
    eval_reward = 0.0
    obs, _ = env.reset()
    online_net.eval()
    for _ in range(eval_steps):
        with torch.no_grad():
            a = int(online_net(torch.from_numpy(obs).to(device).unsqueeze(0)).argmax().item())
        obs, r, term, trunc, _ = env.step(a)
        eval_reward += r
        if term or trunc:
            break
    crash_x = env.x_m
    crash_y = env.y_m

    env.close()

    # Return updated weights as flat numpy array
    parts = []
    with torch.no_grad():
        for module in online_net.modules():
            if isinstance(module, nn.Linear):
                parts.append(module.weight.cpu().numpy().ravel())
                parts.append(module.bias.cpu().numpy())
    return eval_reward, np.concatenate(parts).astype(np.float32), crash_x, crash_y
