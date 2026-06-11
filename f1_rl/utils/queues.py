"""Queue-Helfer für die Live-Anzeige.

Die Simulation produziert Daten schneller, als das UI sie anzeigen kann.
Das UI interessiert sich aber immer nur für den NEUESTEN Stand (das aktuellste
Bild, die aktuellste Statistik). Deshalb gilt überall dasselbe Muster:
alte Einträge wegwerfen, nur den neuesten behalten.

Beide Funktionen schlucken Fehler bewusst — eine volle oder gerade leerlaufende
Queue darf weder die Simulation noch den Server zum Absturz bringen.
"""
from __future__ import annotations

from queue import Empty, Full, Queue


def put_latest(target_queue: Queue | None, item) -> None:
    """Verwirft alle alten Einträge und reiht nur ``item`` ein.

    Sicher gegen ``None`` (keine Queue vorhanden) und gegen eine volle Queue —
    in beiden Fällen passiert einfach nichts.
    """
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
    """Holt alle wartenden Einträge ab und gibt nur den neuesten zurück.

    Gibt ``None`` zurück, wenn die Queue leer oder nicht vorhanden ist.
    """
    if source_queue is None:
        return None
    newest = None
    try:
        while True:
            newest = source_queue.get_nowait()
    except Empty:
        pass
    return newest
