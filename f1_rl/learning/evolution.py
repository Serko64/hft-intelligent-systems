"""Geteilte Generations-Bausteine für beide Trainings-Backends.

DQN-Trainer (trainer.py) und Q-Table-Backend (qtable.py) durchlaufen pro
Generation dieselben Schritte: Epsilon absenken, Hall of Fame pflegen,
Stagnation erkennen, Anzeige-Geschwindigkeit anpassen. Diese kleinen
Schritte leben hier EINMAL — die beiden
Hauptschleifen selbst bleiben bewusst getrennt und explizit lesbar.
"""
from __future__ import annotations

import math

from f1_rl.config import (
    EPSILON_MIN, EPSILON_START, EXPLORE_FRAC, HALL_OF_FAME_K,
    SIM_SPEED_AUTO_CAP, STAGNATION_GENS,
)


def epsilon_for_generation(gen: int, start_gen: int, total_gens: int) -> float:
    """Erkundungsrate für diese Generation: sinkt linear von EPSILON_START auf
    EPSILON_MIN über die ersten EXPLORE_FRAC (z. B. 30 %) der Generationen,
    danach bleibt sie auf dem Minimum (erst viel erkunden, später greedy)."""
    training_progress = (gen - start_gen) / total_gens   # 0.0 (Start) … 1.0 (Ende)
    if training_progress < EXPLORE_FRAC:
        return EPSILON_START - (EPSILON_START - EPSILON_MIN) * training_progress / EXPLORE_FRAC
    return EPSILON_MIN


def eval_step_budget(track) -> int:
    """Länge der Bewertungs-Episode, skaliert an die Strecke, damit eine volle
    Runde wirklich geschafft werden kann.

    Eine feste Obergrenze (z. B. 2000 Schritte) ist für die meisten Strecken zu
    kurz — der Runden-Bonus könnte nie in die Fitness einfließen. Also: Schritte
    für eine Runde bei angenommenem Durchschnittstempo schätzen, Sicherheitsmarge
    drauf, und zwischen EVAL_STEPS und EVAL_STEP_CAP einklemmen.
    """
    from f1_rl.config import (
        EVAL_AVG_SPEED_MS, EVAL_STEP_BUFFER, EVAL_STEP_CAP, EVAL_STEPS, LAPS_PER_EPISODE,
    )
    steps_for_lap = track["total_length_m"] / EVAL_AVG_SPEED_MS * 60.0 * EVAL_STEP_BUFFER
    # Budget für die ganze Mehrrunden-Episode, damit alle LAPS_PER_EPISODE Runden passen.
    return int(min(EVAL_STEP_CAP, max(EVAL_STEPS, steps_for_lap * LAPS_PER_EPISODE)))


def update_hall_of_fame(hall_of_fame: list, fitnesses: list, individuals: list,
                        *, copy_individual) -> None:
    """Pflegt die Allzeit-Bestenliste (wird in place verändert), best first.

    ``fitnesses``/``individuals`` müssen best-first sortiert sein — dann darf die
    Schleife abbrechen, sobald ein Kandidat den schlechtesten Platz nicht mehr
    verdrängen kann.

    ``copy_individual`` entscheidet, wie ein Individuum aufbewahrt wird:
      - DQN: Gewichtsvektoren werden KOPIERT (``vector.copy``), weil die GA-
        Operatoren sie später weiterverwenden und verändern könnten.
      - Q-Table: Tabellen werden per REFERENZ aufbewahrt (Identitätsfunktion),
        denn fertige Tabellen werden nie mehr verändert (Invariante in qtable.py)
        — Kopieren wäre nur unnötige Arbeit pro Generation.
    """
    for fitness, individual in zip(fitnesses, individuals):
        if len(hall_of_fame) < HALL_OF_FAME_K:
            hall_of_fame.append((fitness, copy_individual(individual)))
            hall_of_fame.sort(key=lambda entry: entry[0], reverse=True)
        elif fitness > hall_of_fame[-1][0]:
            hall_of_fame[-1] = (fitness, copy_individual(individual))
            hall_of_fame.sort(key=lambda entry: entry[0], reverse=True)
        else:
            break


def update_top_scores(top_scores: list, best_fitness: float, gen: int) -> None:
    """Pflegt das Scoreboard der zehn besten Generations-Ergebnisse (in place).

    Jeder Eintrag ist ``(score, generation)``, best first — die Web-Anzeige
    (Hud) und die alte Pygame-Ansicht rendern daraus die Top-10-Liste.
    """
    if len(top_scores) < 10 or best_fitness > top_scores[-1][0]:
        top_scores.append((best_fitness, gen))
        top_scores.sort(key=lambda entry: entry[0], reverse=True)
        if len(top_scores) > 10:
            top_scores.pop()


def update_stagnation(prev_best: float, stagnation_count: int,
                      best_fitness: float) -> tuple[float, int, float]:
    """Stagnations-Erkennung: bleibt der Bestwert stecken, wird die Mutation
    hochgeregelt, um aus dem Plateau auszubrechen.

    Gibt (neues prev_best, neuer Zähler, Mutations-Boost) zurück.
    Der Boost wächst von 1.0 bis maximal 3.0.
    """
    if best_fitness > prev_best + 0.5:
        return best_fitness, 0, 1.0
    stagnation_count += 1
    stagnation_boost = 1.0 + min(2.0, stagnation_count / STAGNATION_GENS)
    return prev_best, stagnation_count, stagnation_boost


def update_auto_speed(speed_holder: list, eval_steps: int, gen_seconds: float) -> None:
    """Koppelt die Anzeige-Geschwindigkeit an die Rechenzeit pro Generation.

    Ziel: in der Zeit, die eine Generation zum Rechnen braucht, soll die
    Live-Anzeige etwa eine volle Episode schaffen — dann sind die Autos einer
    Generation durch, bevor die nächste ankommt.
      Schritte/Sekunde der Anzeige = 60 · sub_steps  →  sub_steps = eval_steps / (60 · T)
    """
    if gen_seconds > 0:
        target = math.ceil(eval_steps / (60.0 * gen_seconds))
        speed_holder[0] = max(1, min(SIM_SPEED_AUTO_CAP, target))
