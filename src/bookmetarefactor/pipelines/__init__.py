from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


def start_stage_workers(
    in_queue: Queue[InT],
    out_queue: Queue[OutT],
    process: Callable[[InT], OutT | None],
    upstream_done: Event,
    *,
    workers: int = 2,
    timeout: float | None = None,
) -> list[Thread]:
    """
    Generic queue-to-queue worker launcher.

    Workers exit once the upstream producer signals completion and the input queue is empty.
    """

    def _worker() -> None:
        while True:
            try:
                item = in_queue.get(timeout=timeout)
            except Empty:
                if upstream_done.is_set() and in_queue.empty():
                    break
                continue
            try:
                result = process(item)
                if result is None:
                    continue
                out_queue.put(result)
            finally:
                in_queue.task_done()

    threads: list[Thread] = [
        Thread(target=_worker, daemon=True) for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    return threads


from .discovery import discover_pdfs, produce_pdfs
from .ocr import start_ocr_pipeline
from .extraction import start_extraction_pipeline
from .batching import start_llm_batch_pipeline
from .persist import start_persist_batch_pipeline, persist_llm_task_batch
from .llm import start_llm_pipeline
from .batch import PipelineConfig, run_pipeline

__all__ = [
    "discover_pdfs",
    "produce_pdfs",
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
