from queue import Empty, Full, Queue

# Diese Queues transportieren Live-Zustand (Bilder, Stats) zwischen den Threads. Dabei
# zählt immer nur der neueste Eintrag, ein Rückstau alter Bilder würde die Anzeige nur
# verzögern. Beide Helfer halten die Queue deshalb bewusst leer bis auf das Aktuellste.


def put_latest(target_queue: Queue | None, item) -> None:
    # Vor dem Einreihen alle wartenden Einträge verwerfen, nur item bleibt übrig.
    if target_queue is None:
        return
    try:
        while not target_queue.empty():
            target_queue.get_nowait()
    except Empty:
        pass
    try:
        target_queue.put_nowait(item)
    except Full:
        pass


def get_latest(source_queue: Queue | None):
    # Die Queue leerräumen und nur den zuletzt eingereihten Eintrag zurückgeben.
    if source_queue is None:
        return None
    newest = None
    try:
        while True:
            newest = source_queue.get_nowait()
    except Empty:
        pass
    return newest
