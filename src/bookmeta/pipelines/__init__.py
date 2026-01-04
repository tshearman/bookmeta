import logging
import time
from queue import Empty, Queue
from threading import Event, Thread, current_thread
from typing import Callable, TypeVar

from bookmeta.monitoring import TimedItem

InT = TypeVar("InT")
OutT = TypeVar("OutT")

LOGGER = logging.getLogger(__name__)


def start_stage_workers(
    in_queue: Queue[InT],
    out_queue: Queue[OutT | TimedItem[OutT]],
    process: Callable[[InT], OutT | None],
    upstream_done: Event,
    *,
    workers: int = 2,
    timeout: float = 0.5,
) -> list[Thread]:
    """
    Generic queue-to-queue worker launcher.

    Workers exit once the upstream producer signals completion and the input queue is empty.
    """

    def _worker() -> None:
        thread_name = current_thread().name
        LOGGER.debug("Worker %s started for queue %s", thread_name, in_queue)
        while True:
            try:
                item = in_queue.get(timeout=timeout)
            except Empty:
                if upstream_done.is_set() and in_queue.empty():
                    break
                continue
            start_time = time.perf_counter()
            monitor = getattr(in_queue, "monitor", None)
            queue_name = getattr(in_queue, "name", None)
            try:
                from bookmeta.monitoring import TimedItem

                if isinstance(item, TimedItem):
                    payload: InT = item.obj
                    enqueue_time = item.enqueued_time
                else:
                    payload: InT = item
                    enqueue_time = None
                if monitor and queue_name:
                    monitor.work_started(queue_name)
                    if enqueue_time is not None:
                        monitor.record_wait(queue_name, start_time - enqueue_time)
                result = process(payload)
                if result is None:
                    continue
                if monitor:
                    out_queue.put(TimedItem(result, time.perf_counter()))
                else:
                    out_queue.put(result)
            except Exception:
                LOGGER.exception(
                    "Worker %s encountered error processing item from %s",
                    thread_name,
                    in_queue,
                )
            finally:
                duration = time.perf_counter() - start_time
                if monitor and queue_name:
                    monitor.record_duration(queue_name, duration)
                    monitor.work_finished(queue_name)
                in_queue.task_done()

    threads: list[Thread] = [
        Thread(target=_worker, daemon=True) for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    return threads


from .batch import PipelineConfig, run_pipeline
from .batching import start_llm_batch_pipeline
from .discovery import discover_pdfs, produce_pdfs, start_discovery
from .extraction import start_extraction_pipeline
from .llm import start_llm_pipeline
from .ocr import start_ocr_pipeline
from .persist import persist_llm_task_batch, start_persist_batch_pipeline

__all__ = [
    "discover_pdfs",
    "produce_pdfs",
    "start_discovery",
    "start_stage_workers",
    "start_ocr_pipeline",
    "start_extraction_pipeline",
    "start_llm_batch_pipeline",
    "start_persist_batch_pipeline",
    "persist_llm_task_batch",
    "start_llm_pipeline",
    "PipelineConfig",
    "run_pipeline",
]
