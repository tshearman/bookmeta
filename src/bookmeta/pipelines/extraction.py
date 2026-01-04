from queue import Queue
from threading import Event, Thread

from bookmeta.extraction import execute_extraction_task
from bookmeta.monitoring import TimedItem
from bookmeta.pipelines import start_stage_workers
from bookmeta.types.extraction import ExtractionTask, LLMTask


def start_extraction_pipeline(
    extraction_queue: Queue[ExtractionTask],
    llm_queue: Queue[LLMTask | TimedItem[LLMTask]],
    *,
    workers: int = 2,
    upstream_done: Event,
    timeout: float,
) -> list[Thread]:
    """
    Start worker threads that consume ExtractionTasks and produce LLMTasks.

    Workers exit once the upstream producer signals completion and the extraction queue is empty.
    """
    return start_stage_workers(
        extraction_queue,
        llm_queue,
        execute_extraction_task,
        upstream_done,
        workers=workers,
        timeout=timeout,
    )
