"""Trainiert ein GA-Individuum mit Double DQN und bewertet es.

``run_worker`` läuft in einem eigenen Prozess (einer pro Individuum pro
Generation). Er:
  1. baut das Netz aus den geerbten Gewichten des Individuums,
  2. trainiert per Double-DQN-Gradienten mit Replay-Buffer + ε-greedy,
  3. spielt eine greedy Eval-Runde zur Fitness-Messung,
  4. gibt (fitness, updated_weights, end_x, end_y) zurück.

Der GA in trainer.py selektiert/kreuzt/mutiert dann die zurückgegebenen Vektoren.

CPU-Hinweis: Der Replay-Buffer ist vorab als numpy alloziert (keine Python-Objekte
pro Schritt). GPU wird bei verfügbarem CUDA automatisch mit größerem Batch genutzt.
"""
from __future__ import annotations

import numpy as np

from f1_rl.config import (
    BATCH_SIZE, BATCH_SIZE_GPU, GAMMA, GRAD_CLIP, LR, N_ACTIONS, N_OBS,
    REPLAY_CAPACITY, TARGET_UPDATE_FREQ, TRAIN_FREQ,
)


def run_worker(args: tuple) -> tuple[float, np.ndarray, float, float]:
    """Hybrider DDQN-+-GA-Worker. Siehe Modul-Docstring."""
    inherited_weights, track, n_steps, epsilon, eval_steps, use_rays = args

    # ── Aufgeschobene Imports (spawn-sicher auf Windows) ──────────────────
    import copy

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from f1_rl.learning.network import build_network, flat_to_network, network_to_flat
    from f1_rl.simulation.environment import create_car_env, reset_env, step_env

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE

    # ── Netz aus den geerbten Gewichten bauen ─────────────────────────────
    online_net = flat_to_network(inherited_weights, build_network(), device)

    target_net = copy.deepcopy(online_net)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(online_net.parameters(), lr=LR)

    # ── Vorab allozierter numpy-Replay-Buffer ─────────────────────────────
    # Der Replay-Buffer speichert die letzten REPLAY_CAPACITY Übergänge
    # (Zustand, Aktion, Reward, Folgezustand, fertig?) als fünf parallele
    # Arrays — Zeile i gehört überall zum selben Übergang. Flache Arrays statt
    # einer Liste von Tupeln vermeiden Python-Objekt-Erzeugung bei jedem Schritt.
    replay_states      = np.zeros((REPLAY_CAPACITY, N_OBS), dtype=np.float32)
    replay_next_states = np.zeros((REPLAY_CAPACITY, N_OBS), dtype=np.float32)
    replay_actions     = np.zeros(REPLAY_CAPACITY, dtype=np.int64)
    replay_rewards     = np.zeros(REPLAY_CAPACITY, dtype=np.float32)
    replay_dones       = np.zeros(REPLAY_CAPACITY, dtype=np.float32)
    write_pos   = 0   # nächste Schreibposition (Ringpuffer)
    buffer_size = 0   # wie viele Einträge schon gültig sind

    env = create_car_env(track, use_rays=use_rays)
    obs = reset_env(env)

    # ── Trainingsphase ────────────────────────────────────────────────────
    online_net.train()
    for step in range(n_steps):
        # ε-greedy: mit Wahrscheinlichkeit ε zufällig erkunden, sonst beste Aktion.
        if np.random.random() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            with torch.no_grad():
                q_values = online_net(torch.from_numpy(obs).to(device).unsqueeze(0))
                action = int(q_values.argmax().item())

        next_obs, reward, terminated, truncated, _ = step_env(env, action)
        done = terminated or truncated

        # Übergang in den Ringpuffer schreiben (älteste Einträge werden überschrieben).
        replay_states[write_pos]      = obs
        replay_next_states[write_pos] = next_obs
        replay_actions[write_pos]     = action
        replay_rewards[write_pos]     = reward
        replay_dones[write_pos]       = float(done)
        write_pos   = (write_pos + 1) % REPLAY_CAPACITY
        buffer_size = min(buffer_size + 1, REPLAY_CAPACITY)

        obs = next_obs
        if done:
            obs = reset_env(env)

        # Gradienten-Update — alle TRAIN_FREQ Schritte
        if buffer_size >= batch_size and step % TRAIN_FREQ == 0:
            # Zufälliges Mini-Batch aus dem Buffer ziehen.
            batch_indices = np.random.choice(buffer_size, batch_size, replace=False)

            batch_states      = torch.from_numpy(replay_states[batch_indices]).to(device)
            batch_next_states = torch.from_numpy(replay_next_states[batch_indices]).to(device)
            batch_actions     = torch.from_numpy(replay_actions[batch_indices]).to(device)
            batch_rewards     = torch.from_numpy(replay_rewards[batch_indices]).to(device)
            batch_dones       = torch.from_numpy(replay_dones[batch_indices]).to(device)

            with torch.no_grad():
                # Double DQN: das Online-Netz WÄHLT die Aktion, das Target-Netz BEWERTET sie.
                next_actions = online_net(batch_next_states).argmax(dim=1)
                next_q = target_net(batch_next_states).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                # Kein zukünftiger Wert, wenn die Episode vorbei war (1 - done).
                td_target = batch_rewards + GAMMA * (1.0 - batch_dones) * next_q

            predicted_q = online_net(batch_states).gather(
                1, batch_actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
            optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

    # ── Greedy-Eval-Phase (eine Episode) ─────────────────────────────────
    # Score = Reward einer sauberen Fahrt ab dem Start — spiegelt direkt, wie
    # weit der Agent kommt, vergleichbar mit der Live-Anzeige. eval_steps ist an
    # die Streckenlänge budgetiert, damit eine volle Runde (und ihr Ziel-Bonus)
    # erreichbar ist; die Schleife bricht bei Crash/Runde trotzdem früh ab.
    eval_reward = 0.0
    obs = reset_env(env)
    online_net.eval()
    for _ in range(eval_steps):
        with torch.no_grad():
            action = int(online_net(torch.from_numpy(obs).to(device).unsqueeze(0)).argmax().item())
        obs, reward, terminated, truncated, _ = step_env(env, action)
        eval_reward += reward
        if terminated or truncated:
            break
    crash_x = env["x_m"]
    crash_y = env["y_m"]

    # Aktualisierte Gewichte als flaches numpy-Array zurück (darauf arbeitet der GA).
    return eval_reward, network_to_flat(online_net), crash_x, crash_y
