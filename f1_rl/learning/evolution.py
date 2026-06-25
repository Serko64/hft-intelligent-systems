import math

from f1_rl.config import (
    EPSILON_MIN, EPSILON_START, EXPLORE_FRAC, HALL_OF_FAME_K,
    SIM_SPEED_AUTO_CAP, STAGNATION_GENS,
)


# ε sinkt über den ersten EXPLORE_FRAC-Anteil des Trainings linear von EPSILON_START
# auf EPSILON_MIN und bleibt danach unten: erst viel erkunden, später greedy fahren.
def epsilon_for_generation(gen: int, start_gen: int, total_gens: int) -> float:
    training_progress = (gen - start_gen) / total_gens
    if training_progress < EXPLORE_FRAC:
        return EPSILON_START - (EPSILON_START - EPSILON_MIN) * training_progress / EXPLORE_FRAC
    return EPSILON_MIN


def eval_step_budget(track) -> int:
    # Wie viele Schritte eine Eval-Episode laufen darf, an die Streckenlänge gekoppelt,
    # damit die geplanten Runden erreichbar bleiben, aber zwischen EVAL_STEPS und
    # EVAL_STEP_CAP geklemmt.
    from f1_rl.config import (
        EVAL_AVG_SPEED_MS, EVAL_STEP_BUFFER, EVAL_STEP_CAP, EVAL_STEPS, LAPS_PER_EPISODE,
    )
    steps_for_lap = track["total_length_m"] / \
        EVAL_AVG_SPEED_MS * 60.0 * EVAL_STEP_BUFFER
    return int(min(EVAL_STEP_CAP, max(EVAL_STEPS, steps_for_lap * LAPS_PER_EPISODE)))


# Allzeit-Bestenliste der besten HALL_OF_FAME_K Individuen, absteigend sortiert.
# Die Liste kommt schon best-first an, beim ersten zu schwachen Eintrag kann die
# Schleife also abbrechen. copy_individual bestimmt, ob kopiert oder referenziert wird.
def update_hall_of_fame(hall_of_fame: list, fitnesses: list, individuals: list,
                        *, copy_individual) -> None:
    for fitness, individual in zip(fitnesses, individuals):
        if len(hall_of_fame) < HALL_OF_FAME_K:
            hall_of_fame.append((fitness, copy_individual(individual)))
            hall_of_fame.sort(key=lambda entry: entry[0], reverse=True)
        elif fitness > hall_of_fame[-1][0]:
            hall_of_fame[-1] = (fitness, copy_individual(individual))
            hall_of_fame.sort(key=lambda entry: entry[0], reverse=True)
        else:
            break


# Scoreboard fürs UI: die zehn besten je erreichten Scores mit ihrer Generation.
def update_top_scores(top_scores: list, best_fitness: float, gen: int) -> None:
    if len(top_scores) < 10 or best_fitness > top_scores[-1][0]:
        top_scores.append((best_fitness, gen))
        top_scores.sort(key=lambda entry: entry[0], reverse=True)
        if len(top_scores) > 10:
            top_scores.pop()


# Steckt der Bestwert fest, zählt der Stagnationszähler hoch und der zurückgegebene
# Boost-Faktor regelt die Mutation hoch (bis 3-fach), um vom Plateau wegzukommen. Ein
# echter Fortschritt (> 0.5) setzt alles zurück.
def update_stagnation(prev_best: float, stagnation_count: int,
                      best_fitness: float) -> tuple[float, int, float]:
    if best_fitness > prev_best + 0.5:
        return best_fitness, 0, 1.0
    stagnation_count += 1
    stagnation_boost = 1.0 + min(2.0, stagnation_count / STAGNATION_GENS)
    return prev_best, stagnation_count, stagnation_boost


# Anzeige-Tempo so wählen, dass eine Eval-Episode etwa in der Rechenzeit einer
# Generation durchläuft, damit Animation und Trainingsfortschritt zusammenpassen.
def update_auto_speed(speed_holder: list, eval_steps: int, gen_seconds: float) -> None:
    if gen_seconds > 0:
        target = math.ceil(eval_steps / (60.0 * gen_seconds))
        speed_holder[0] = max(1, min(SIM_SPEED_AUTO_CAP, target))
