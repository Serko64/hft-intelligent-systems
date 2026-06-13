from queue import Empty, Full, Queue


def put_latest(target_queue: Queue | None, item) -> None:
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
    if source_queue is None:
        return None
    newest = None
    try:
        while True:
            newest = source_queue.get_nowait()
    except Empty:
        pass
    return newest
