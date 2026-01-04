from queue import Queue
from typing import Callable

from bookmeta.monitoring import MonitoredQueue, QueueMonitor

QueueFactory = Callable[[str, int | None], Queue]


def make_queue_factory(default_size: int | None, monitor: QueueMonitor) -> QueueFactory:
    """
    Return a factory that creates queues with optional monitoring support.
    """

    def _queue_size(preferred: int | None) -> int:
        size = preferred if preferred is not None else default_size
        return 0 if size is None else size

    def _make(name: str, preferred: int | None) -> Queue:
        maxsize = _queue_size(preferred)
        if monitor.enabled:
            return MonitoredQueue(name, maxsize=maxsize, monitor=monitor)
        return Queue(maxsize=maxsize)

    return _make
