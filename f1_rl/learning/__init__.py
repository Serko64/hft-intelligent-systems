"""Learning layer: the driving brain and how it is trained.

- network.py   — the single PyTorch network definition used everywhere
- agent.py     — wraps a trained network for inference (pick an action)
- genetics.py  — genetic-algorithm operations (selection, crossover, mutation)
- worker.py    — trains one network with Double DQN (one GA individual)
- trainer.py   — the generation loop that ties DQN + GA together
"""
