import json
from queue import Empty, Queue
from threading import Event, Thread

from bookmetarefactor.config import MAX_BATCH_BYTES, MAX_REQUESTS_PER_BATCH
from bookmetarefactor.types.extraction import LLMTask, LLMTaskBatch


def _task_size_bytes(task: LLMTask) -> int:
    """Compute byte size of a single LLMTask payload when serialized."""
    return len((json.dumps(task.payload, ensure_ascii=False) + "\n").encode("utf-8"))


def _worker(
    llm_queue: Queue[LLMTask],
    batch_queue: Queue[LLMTaskBatch],
    upstream_done: Event,
    timeout: float | None,
) -> None:

    current: LLMTaskBatch = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        batch_queue.put(current)
        current = []
        current_bytes = 0

    while True:
        try:
            task = llm_queue.get(timeout=timeout)
        except Empty:
            if upstream_done.is_set() and llm_queue.empty():
                flush()
                break
            continue

        try:
            line_bytes = _task_size_bytes(task)
            if current and (
                len(current) >= MAX_REQUESTS_PER_BATCH
                or current_bytes + line_bytes > MAX_BATCH_BYTES
            ):
                flush()
            current.append(task)
            current_bytes += line_bytes
        finally:
            llm_queue.task_done()

    flush()


def start_llm_batch_pipeline(
    llm_queue: Queue[LLMTask],
    batch_queue: Queue[LLMTaskBatch],
    *,
    upstream_done: Event,
    timeout: float | None = None,
) -> Thread:
    worker = Thread(
        target=_worker,
        args=(llm_queue, batch_queue, upstream_done, timeout),
        daemon=True,
    )
    worker.start()
    return worker
