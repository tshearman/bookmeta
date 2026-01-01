from queue import Queue
from threading import Event, Thread

from bookmetarefactor.extraction import execute_llm_task
from bookmetarefactor.pipelines import start_stage_workers
from bookmetarefactor.types.bookinfo import BookInfoResult
from bookmetarefactor.types.extraction import LLMTask


def start_llm_pipeline(
    llm_queue: Queue[LLMTask],
    bookinfo_queue: Queue[BookInfoResult],
    *,
    workers: int = 2,
    upstream_done: Event,
    timeout: float | None = None,
) -> list[Thread]:
    """
    Consume LLMTasks, execute them, and enqueue BookInfoResponse results.

    Tasks that return None are skipped. Workers exit once upstream_done is set and
    the input queue has been drained.
    """
    return start_stage_workers(
        llm_queue,
        bookinfo_queue,
        execute_llm_task,
        upstream_done,
        workers=workers,
        timeout=timeout,
    )
