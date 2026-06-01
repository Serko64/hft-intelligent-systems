"""Train one GA individual with Double DQN, then score it.

``run_worker`` is executed in a separate process (one per individual per
generation). It:
  1. builds the network from the individual's inherited weights,
  2. runs Double-DQN gradient training with a replay buffer + ε-greedy,
  3. plays one greedy evaluation lap to measure fitness,
  4. returns (fitness, updated_weights, end_x, end_y).

The GA in trainer.py then selects/crosses/mutates the returned weight vectors.

CPU notes: the replay buffer is pre-allocated numpy (no per-step Python object
churn). GPU is used automatically when CUDA is available, with a larger batch.
"""
from __future__ import annotations

import numpy as np

from f1_rl.config import (
    BATCH_SIZE, BATCH_SIZE_GPU, GAMMA, GRAD_CLIP, LR, N_ACTIONS, N_OBS,
    REPLAY_CAPACITY, TARGET_UPDATE_FREQ, TRAIN_FREQ,
)


def run_worker(args: tuple) -> tuple[float, np.ndarray, float, float]:
    """Hybrid DDQN + GA worker. See module docstring."""
    inherited_weights, track, n_steps, epsilon, eval_steps, use_rays = args

    # ── Deferred imports (spawn-safe on Windows) ──────────────────────────
    import copy

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from f1_rl.learning.network import build_network, flat_to_network, network_to_flat
    from f1_rl.simulation.environment import F1Env

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE

    # ── Build network from inherited weights ──────────────────────────────
    online_net = flat_to_network(inherited_weights, build_network(), device)

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

    env = F1Env(track=track, render_mode=None, use_rays=use_rays)
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

    # Return updated weights as a flat numpy array (the GA operates on this).
    return eval_reward, network_to_flat(online_net), crash_x, crash_y
