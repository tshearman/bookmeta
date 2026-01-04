from queue import Queue
from threading import Event, Thread

from bookmeta.extraction import execute_llm_task
from bookmeta.monitoring import TimedItem
from bookmeta.pipelines import start_stage_workers
from bookmeta.pipelines.cache import load_cached_bookinfo, save_cached_bookinfo
from bookmeta.types.bookinfo import BookInfoResult
from bookmeta.types.extraction import LLMTask


def start_llm_pipeline(
    llm_queue: Queue[LLMTask],
    bookinfo_queue: Queue[BookInfoResult | TimedItem[BookInfoResult]],
    *,
    workers: int = 2,
    upstream_done: Event,
    timeout: float,
    resume: bool = False,
) -> list[Thread]:
    """
    Consume LLMTasks, execute them, and enqueue BookInfoResponse results.

    Tasks that return None are skipped. Workers exit once upstream_done is set and
    the input queue has been drained.
    """

    def _process(task: LLMTask) -> BookInfoResult | None:
        if resume:
            cached = load_cached_bookinfo(task.pdf)
            if cached:
                return cached
        result = execute_llm_task(task)
        if result:
            save_cached_bookinfo(result)
        return result

    return start_stage_workers(
        llm_queue,
        bookinfo_queue,
        _process,
        upstream_done,
        workers=workers,
        timeout=timeout,
    )
