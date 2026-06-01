"""Central configuration — all tuneable constants, paths, and UI settings."""
from __future__ import annotations

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = os.path.join(os.path.dirname(__file__), "..")
CIRCUITS_DIR   = os.path.join(ROOT, "circuits")
MODEL_DIR      = os.path.join(ROOT, "model")
STATE_PATH     = os.path.join(MODEL_DIR, "training_state.json")
MODEL_PATH     = os.path.join(MODEL_DIR, "model.npy")   # flat neural-net weights
QTABLE_PATH    = MODEL_PATH                              # alias kept for compatibility
REPLAY_DIR     = os.path.join(MODEL_DIR, "replays")
BEST_LAPS_PATH = os.path.join(MODEL_DIR, "best_laps.json")

# ── Observation / action space ────────────────────────────────────────────────
# obs: [0]x  [1]y  [2]hdg_diff/π  [3]speed  [4]progress
#      [5]dist_left  [6]dist_right
#      [7]ray-75°  [8]ray-45°  [9]ray-20°  [10]ray0°
#      [11]ray+20°  [12]ray+45°  [13]ray+75°
N_OBS     = 14   # must match env.observation_space
N_ACTIONS = 20   # must match len(DISCRETE_ACTIONS) in env.py

# ── Neural network architecture ───────────────────────────────────────────────
NET_HIDDEN = (128, 128)   # hidden layer widths; change to (256, 256) for harder circuits

# ── DQN (per-worker gradient training) ───────────────────────────────────────
LR                 = 3e-4   # Adam learning rate
GAMMA              = 0.97   # discount factor
REPLAY_CAPACITY    = 12_000 # transitions kept per worker (pre-allocated numpy)
BATCH_SIZE         = 64     # mini-batch size on CPU
BATCH_SIZE_GPU     = 256    # mini-batch size on GPU (better utilisation)
TRAIN_FREQ         = 8      # gradient update every N steps (8 = ~50% less grad overhead vs 4)
TARGET_UPDATE_FREQ = 600    # hard-copy online → target every N steps
GRAD_CLIP          = 10.0   # max gradient norm

# ── ε-greedy exploration schedule ────────────────────────────────────────────
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EXPLORE_FRAC  = 0.3    # fraction of gens to anneal ε over

# ── Genetic algorithm ─────────────────────────────────────────────────────────
N_POP           = 40
HALL_OF_FAME_K  = 8
N_BEST_CLONES   = 4       # light-mutation copies of all-time best injected each gen
STEPS_PER_GEN   = 5_000   # env steps each worker runs per generation
EVAL_STEPS      = 2_000   # floor for greedy evaluation steps (honest fitness signal)
# The eval episode must be long enough for a car to actually complete a lap, otherwise
# the lap/goal bonus can never register in fitness.  The real budget is scaled to the
# track length (see train._eval_step_budget) using these knobs:
EVAL_AVG_SPEED_MS = 25.0   # assumed avg lap speed when budgeting eval steps
EVAL_STEP_BUFFER  = 1.35   # safety margin on the lap-length estimate
EVAL_STEP_CAP     = 15_000 # hard upper bound so eval never runs away
STAGNATION_GENS = 15
MUTATION_RATE   = 0.02    # fraction of weights perturbed per offspring
MUTATION_NOISE  = 0.10    # std of Gaussian noise added during mutation
SAVE_EVERY      = 5
MAX_REPLAYS     = 10

# ── Reward shaping ─────────────────────────────────────────────────────────────
LAP_BONUS       = 5_000.0  # reward for completing a full lap (episode then terminates)

# ── Pack / swarm evolution ─────────────────────────────────────────────────────
# "Rudel-Evolution": instead of pure individual survival-of-the-fittest, the
# population is clustered into packs and selection is partly group-level, so weak
# DNA can survive by belonging to a strong pack (group / kin selection).
EVOLUTION_MODE_DEFAULT = "classic"   # "classic" | "pack"
N_PACKS            = 5     # number of packs (behavioural clusters) per generation
PACK_SUPPORT       = 0.5   # how strongly weak members are pulled toward the pack best
MIGRATION_RATE     = 0.10  # chance an offspring is a cross-pack crossover (gene flow)
PACK_MIN_SURVIVORS = 1     # elites guaranteed to survive per pack (no abrupt extinction)

# ── UI ────────────────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1600, 1000
FPS = 60

CIRCUITS = [
    # query (osmnx geocode)              geojson fallback (optional)                     half-width m
    ("Circuit de Monaco",                 os.path.join(CIRCUITS_DIR, "mc-1929.geojson"), 10.0),
    ("Silverstone Circuit",               os.path.join(CIRCUITS_DIR, "gb-1948.geojson"), 10.0),
    ("Autodromo Nazionale Monza",         os.path.join(CIRCUITS_DIR, "it-1922.geojson"), 10.0),
    ("Circuit de Spa-Francorchamps",      os.path.join(CIRCUITS_DIR, "be-1925.geojson"), 10.0),
    ("Suzuka Circuit",                    os.path.join(CIRCUITS_DIR, "jp-1962.geojson"), 10.0),
    # Additional circuits — loaded live via osmnx on first use, then cached as .pkl.
    ("Red Bull Ring",                     None, 12.0),
    ("Hungaroring",                       None, 11.0),
    ("Circuit Zandvoort",                 None, 11.0),
    ("Autódromo José Carlos Pace",        None, 11.0),  # Interlagos
    ("Circuit of the Americas",           None, 12.0),
    ("Bahrain International Circuit",      None, 12.0),
    ("Autodromo Enzo e Dino Ferrari",     None, 11.0),  # Imola
    ("Circuit de Barcelona-Catalunya",    None, 12.0),
    ("Marina Bay Street Circuit",         None, 10.0),
    ("Jeddah Corniche Circuit",           None, 11.0),
]

STEPS_PRESETS = [500, 1_000, 2_000, 3_000, 5_000, 8_000, 12_000, 20_000,
                 50_000, 100_000, 250_000, 500_000, 1_000_000]
GENS_PRESETS  = [5, 10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

BG     = (12, 12, 18)
ACCENT = (200, 30, 30)
TEXT   = (220, 220, 220)
DIM    = (110, 110, 130)
SEL_BG = (55, 25, 75)
HOV_BG = (35, 35, 55)
GOLD   = (255, 215, 0)
GREEN  = (50, 220, 80)
