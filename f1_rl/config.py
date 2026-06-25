import os

# ── Pfade ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
CIRCUITS_DIR = os.path.join(ROOT, "circuits")
MODEL_DIR = os.path.join(ROOT, "model")
STATE_PATH = os.path.join(MODEL_DIR, "training_state.json")
# flache Netz-Gewichte (DQN)
MODEL_PATH = os.path.join(MODEL_DIR, "model.npy")

# ── Beobachtungs- und Aktionsraum ─────────────────────────────────────────────
# obs: [0]x  [1]y  [2]hdg_diff/π  [3]speed  [4]progress
#      [5]dist_left  [6]dist_right
#      [7]ray-75°  [8]ray-45°  [9]ray-20°  [10]ray0°
#      [11]ray+20°  [12]ray+45°  [13]ray+75°
N_OBS = 14   # muss zur Länge des Beobachtungsvektors in environment._get_obs passen
N_ACTIONS = 20   # muss len(DISCRETE_ACTIONS) in environment.py entsprechen

# ── Netzarchitektur ───────────────────────────────────────────────────────────
# Breite der versteckten Schichten, für schwerere Strecken auf (256, 256) erhöhen
NET_HIDDEN = (128, 128)
# Dropout-Wahrscheinlichkeit auf den versteckten Schichten (0.0 = aus). Bringt keine
# zusätzlichen Gewichte, model.npy bleibt also kompatibel. Aktiv nur in der Gradienten-
# phase eines Workers, Inferenz, Anzeige und Eval laufen mit Dropout AUS (net.eval()).
# Achtung: Dropout in wertbasiertem RL (DQN) ist unüblich und kann das Lernen
# destabilisieren, am besten mit einem kleinen Wert (~0.1) testen.
NET_DROPOUT = 0.0

# ── DQN (Gradiententraining je Worker) ────────────────────────────────────────
LR = 3e-4   # Adam-Lernrate
GAMMA = 0.97   # Diskontfaktor
REPLAY_CAPACITY = 12_000  # gespeicherte Übergänge je Worker (vorab als numpy alloziert)
BATCH_SIZE = 64     # Mini-Batch-Größe auf der CPU
BATCH_SIZE_GPU = 256    # Mini-Batch-Größe auf der GPU (bessere Auslastung)
# Gradienten-Update alle N Schritte (8 = ~50% weniger Grad-Overhead als bei 4)
TRAIN_FREQ = 8
TARGET_UPDATE_FREQ = 600    # online nach N Schritten hart aufs Target kopieren
GRAD_CLIP = 10.0   # maximale Gradientennorm

# ── ε-greedy-Erkundungsplan ───────────────────────────────────────────────────
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EXPLORE_FRAC = 0.3    # Anteil der Generationen, über den ε abgesenkt wird

# ── Genetischer Algorithmus ───────────────────────────────────────────────────
N_POP = 40
# Q-table backend population: tabular agents carry a sparse dict per individual
# (more RAM than a flat weight vector), so a smaller population keeps the parallel
# evaluation light while still giving the GA something to select/cross/mutate.
QTABLE_N_POP = 24
HALL_OF_FAME_K = 12
N_BEST_CLONES = 6       # leicht mutierte Kopien des Allzeit-Besten, je Generation eingespeist
STEPS_PER_GEN = 5_000   # Umgebungsschritte je Worker und Generation
# Untergrenze für die Greedy-Eval-Schritte (ehrliches Fitness-Signal)
EVAL_STEPS = 2_000
# Die Eval-Episode muss lang genug sein, dass ein Auto wirklich eine Runde schafft, sonst
# kann der Runden- bzw. Zielbonus nie in die Fitness einfließen. Das echte Budget wird über
# die Streckenlänge skaliert (siehe evolution.eval_step_budget), und zwar mit diesen Reglern:
EVAL_AVG_SPEED_MS = 25.0   # angenommene Durchschnittsgeschwindigkeit beim Budgetieren
EVAL_STEP_BUFFER = 1.35   # Sicherheitsmarge auf die geschätzte Rundenlänge
EVAL_STEP_CAP = 15_000  # harte Obergrenze, damit die Eval nie davonläuft
# Multi-Start-Eval (optional, per Frontend-Schalter): statt nur am Start/Ziel wird
# der Eval-Score über mehrere, gleichmäßig auf der Centerline verteilte Startpunkte
# gemittelt, ein robusteres Fitness-Signal, das nicht auf "Strecke ab Start" überanpasst.
EVAL_START_POSITIONS = 4   # Anzahl Startpunkte bei aktivem Multi-Start-Eval
STAGNATION_GENS = 15
MUTATION_RATE = 0.02    # Anteil der Gewichte, die je Nachkomme verändert werden
MUTATION_NOISE = 0.10    # Standardabweichung des Gauß-Rauschens bei der Mutation
SAVE_EVERY = 5

# ── Reward-Gestaltung ──────────────────────────────────────────────────────────
LAP_BONUS = 5_000.0  # Belohnung für eine komplett gefahrene Runde

# Curriculum (Variante B): eine Episode läuft über mehrere Runden, statt nach der
# ersten zu enden. Runde 1 wird nur danach bewertet, ob das Auto überhaupt herumkommt
# (Basis-Rewards). Die Optimierungs-Rewards (Tempo, Langsam-Strafe, Lenken, Zeitdruck)
# schalten sich erst zu, sobald ein Auto CURRICULUM_AFTER_LAP Runde(n) geschafft hat,
# erst überleben lernen, dann schnell werden. Die Reward-REGEL bleibt jede Generation
# gleich, die Fitness bleibt über den GA hinweg vergleichbar (kein Hall-of-Fame-Reset nötig).
LAPS_PER_EPISODE = 2   # Episode endet nach so vielen Runden (oder bei einem Crash)
CURRICULUM_AFTER_LAP = 1   # Optimierungs-Rewards aktiv ab so vielen gefahrenen Runden


# ── Benutzeroberfläche ─────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1600, 1000
FPS = 60

# Live-Geschwindigkeit: wie viele Simulations-Teilschritte pro gerendertem Bild laufen.
# Die Anzeige liefert weiter 60 Bilder pro Sekunde (flüssig), aber jedes Bild rückt die
# Sim um so viele Schritte vor, die Autos bewegen sich also schneller und halten Schritt
# mit dem Trainingsfortschritt. Live über die UI verstellbar (1 = Echtzeit, höher = flotter).
SIM_SPEED_DEFAULT = 3
SIM_SPEED_MAX = 20    # Obergrenze des manuellen Reglers
# Die Auto-Geschwindigkeit soll nur eine flüssige, lebendige Bewegung liefern. Die harte
# Generationssynchronisation garantiert inzwischen MAX_DISPLAY_LAG (Autos springen auf die
# neueste Generation). Eine niedrigere Obergrenze hält die Animation flüssig, statt den
# Anzeige-Thread mit 80 Autos zu überlasten.
SIM_SPEED_AUTO_CAP = 12    # Obergrenze der automatischen (generationssynchronen) Geschwindigkeit

CIRCUITS = [
    # Suchbegriff (osmnx-Geocode)        GeoJSON-Fallback (optional)                     halbe Breite m
    ("Circuit de Monaco",                 os.path.join(
        CIRCUITS_DIR, "mc-1929.geojson"), 10.0),
    ("Silverstone Circuit",               os.path.join(
        CIRCUITS_DIR, "gb-1948.geojson"), 10.0),
    ("Autodromo Nazionale Monza",         os.path.join(
        CIRCUITS_DIR, "it-1922.geojson"), 10.0),
    ("Circuit de Spa-Francorchamps",
     os.path.join(CIRCUITS_DIR, "be-1925.geojson"), 10.0),
    ("Suzuka Circuit",                    os.path.join(
        CIRCUITS_DIR, "jp-1962.geojson"), 10.0),
    # Weitere Strecken, beim ersten Gebrauch live über osmnx geladen, danach als .pkl gecacht.
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
GENS_PRESETS = [5, 10, 25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

BG = (12, 12, 18)
ACCENT = (200, 30, 30)
TEXT = (220, 220, 220)
DIM = (110, 110, 130)
SEL_BG = (55, 25, 75)
HOV_BG = (35, 35, 55)
GOLD = (255, 215, 0)
GREEN = (50, 220, 80)
