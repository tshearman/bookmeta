from queue import Queue
from threading import Event, Thread

from bookmetarefactor.ocr import OcrConfig, execute_task, ocr_tasks
from bookmetarefactor.pipelines import start_stage_workers
from bookmetarefactor.types import Pdf
from bookmetarefactor.types.extraction import ExtractionConfig, ExtractionTask
from bookmetarefactor.types.ocr import OcrResult, OcrResults


def _tasks_for_pdf(pdf: Pdf, config: OcrConfig) -> OcrResults:
    """Spawn threads for each OCR task and yield collected results."""
    tasks = list(ocr_tasks(pdf, config))
    results: list[OcrResult] = []

    def _run(task_idx: int) -> None:
        results.append(execute_task(tasks[task_idx]))

    threads = [
        Thread(target=_run, args=(idx,), daemon=True) for idx in range(len(tasks))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return results


def start_ocr_pipeline(
    pdf_queue: Queue[Pdf],
    extraction_queue: Queue[ExtractionTask],
    ocr_config: OcrConfig,
    extraction_config: ExtractionConfig,
    *,
    workers: int = 2,
    upstream_done: Event,
    timeout: float | None = None,
) -> list[Thread]:
    """
    Start OCR worker threads that consume Pdfs and produce ExtractionTasks.

    Workers exit once the upstream producer signals completion and the PDF queue is empty.
    """

    def _to_extraction(pdf: Pdf) -> ExtractionTask:
        return ExtractionTask(pdf, _tasks_for_pdf(pdf, ocr_config), extraction_config)

    return start_stage_workers(
        pdf_queue,
        extraction_queue,
        _to_extraction,
        upstream_done,
        workers=workers,
        timeout=timeout,
    )
