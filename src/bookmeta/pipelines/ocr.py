from queue import Queue
from threading import Event, Thread

from bookmeta.monitoring import TimedItem
from bookmeta.ocr import OcrConfig, execute_ocr, execute_task, ocr_tasks
from bookmeta.pipelines import start_stage_workers
from bookmeta.types import Pdf
from bookmeta.types.extraction import ExtractionConfig, ExtractionTask
from bookmeta.types.ocr import OcrResult, OcrResults


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
    pdf_queue: Queue[Pdf | TimedItem[Pdf]],
    extraction_queue: Queue[ExtractionTask | TimedItem[ExtractionTask]],
    ocr_config: OcrConfig,
    extraction_config: ExtractionConfig,
    *,
    workers: int = 2,
    upstream_done: Event,
    timeout: float,
) -> list[Thread]:
    """
    Start OCR worker threads that consume Pdfs and produce ExtractionTasks.

    Workers exit once the upstream producer signals completion and the PDF queue is empty.
    """

    def _to_extraction(item: Pdf | TimedItem[Pdf]) -> ExtractionTask:
        pdf = item.obj if isinstance(item, TimedItem) else item
        return ExtractionTask(pdf, execute_ocr(pdf, ocr_config), extraction_config)

    return start_stage_workers(
        pdf_queue,
        extraction_queue,
        _to_extraction,
        upstream_done,
        workers=workers,
        timeout=timeout,
    )
