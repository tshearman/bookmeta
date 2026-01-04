from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Iterable

from bookmeta.config import DEFAULT_DB_PATH
from bookmeta.monitoring import QueueMonitor
from bookmeta.ocr import OcrConfig
from bookmeta.persistence import (
    execute_persist_bookinfo,
    has_existing_result,
)
from bookmeta.pipelines import (
    produce_pdfs,
    start_extraction_pipeline,
    start_llm_pipeline,
    start_ocr_pipeline,
    start_stage_workers,
)
from bookmeta.pipelines.metadata import write_metadata
from bookmeta.pipelines.queues import make_queue_factory
from bookmeta.types.bookinfo import (
    BookInfoResult,
    DetailedBookInfoResult,
)
from bookmeta.types.extraction import (
    ContextLimits,
    ExtractionConfig,
    ProviderConfig,
)


@dataclass
class PipelineConfig:
    roots: Iterable[Path | str]
    prompt: str
    provider_config: ProviderConfig
    ocr_config: OcrConfig = field(default_factory=OcrConfig)
    context_limits: ContextLimits = field(default_factory=ContextLimits)
    queue_size: int = 32
    dedupe: bool = True
    limit: int | None = None
    ocr_workers: int = 2
    extraction_workers: int = 2
    llm_workers: int = 2
    persist_workers: int = 1
    stage_timeout: float = 0.5
    results_db: Path = DEFAULT_DB_PATH
    collect_results: bool = False
    monitor_queues: bool = False
    writer_bin: Path | None = None
    output_dir: Path | None = None
    resume: bool = False

    @property
    def extraction_config(self) -> ExtractionConfig:
        return ExtractionConfig(self.prompt, self.provider_config, self.context_limits)


def run_pipeline(cfg: PipelineConfig) -> list[DetailedBookInfoResult]:
    """
    Orchestrate the full pipeline from PDF discovery through LLM execution
    and persist BookInfo results into the configured SQLite database.
    """

    extraction_config = ExtractionConfig(
        prompt=cfg.prompt,
        provider_config=cfg.provider_config,
        context_limits=cfg.context_limits,
    )

    provider = cfg.provider_config.provider
    model = cfg.provider_config.model

    def _skip_processed(pdf) -> bool:
        return has_existing_result(
            cfg.results_db, pdf.hash, provider=provider, model=model
        )

    monitor = QueueMonitor(
        title="BookMeta Queue Monitor",
        refresh_interval=0.2,
        enabled=cfg.monitor_queues,
    )
    monitor.start()
    make_queue = make_queue_factory(cfg.queue_size, monitor)

    pdf_queue: Queue = make_queue("ocr", None)
    extraction_queue: Queue = make_queue("extraction", None)
    llm_queue: Queue = make_queue("llm", None)
    bookinfo_queue: Queue = make_queue("bookinfo", None)
    persist_queue: Queue = make_queue("persist", None)
    metadata_queue: Queue = make_queue("metadata", None)
    completed_queue: Queue = make_queue("completed", None)

    producer_done = Event()
    ocr_done = Event()
    extraction_done = Event()
    llm_done = Event()

    def _produce() -> None:
        produce_pdfs(
            cfg.roots,
            pdf_queue,
            dedupe=cfg.dedupe,
            limit=cfg.limit,
            skip_processed=_skip_processed,
        )
        producer_done.set()

    producer_thread = Thread(target=_produce, daemon=True)
    producer_thread.start()

    ocr_threads = start_ocr_pipeline(
        pdf_queue,
        extraction_queue,
        cfg.ocr_config,
        extraction_config,
        workers=cfg.ocr_workers,
        upstream_done=producer_done,
        timeout=cfg.stage_timeout,
    )

    extraction_threads = start_extraction_pipeline(
        extraction_queue,
        llm_queue,
        workers=cfg.extraction_workers,
        upstream_done=ocr_done,
        timeout=cfg.stage_timeout,
    )

    llm_threads = start_llm_pipeline(
        llm_queue,
        bookinfo_queue,
        workers=cfg.llm_workers,
        upstream_done=extraction_done,
        timeout=cfg.stage_timeout,
        resume=cfg.resume,
    )

    def _to_detailed(result: BookInfoResult) -> DetailedBookInfoResult | None:
        try:
            return DetailedBookInfoResult(
                pdf=result.pdf, detailed=result.bookinfo.info.as_detailed_book_info()
            )
        except Exception:
            return None

    detailed_threads = start_stage_workers(
        bookinfo_queue,
        persist_queue,
        _to_detailed,
        llm_done,
        workers=cfg.persist_workers,
        timeout=cfg.stage_timeout,
    )

    persist_threads = start_stage_workers(
        persist_queue,
        metadata_queue,
        execute_persist_bookinfo(cfg),
        llm_done,
        workers=cfg.persist_workers,
        timeout=cfg.stage_timeout,
    )

    metadata_threads: list[Thread] = []
    metadata_threads = start_stage_workers(
        metadata_queue,
        completed_queue,
        write_metadata(cfg),
        llm_done,
        workers=cfg.persist_workers,
        timeout=cfg.stage_timeout,
    )

    # Producer Stage
    producer_thread.join()

    # OCR Stages
    pdf_queue.join()
    for thread in ocr_threads:
        thread.join()
    ocr_done.set()
    monitor.mark_done("ocr")

    # Extraction Stage
    extraction_queue.join()
    for thread in extraction_threads:
        thread.join()
    extraction_done.set()
    monitor.mark_done("extraction")

    # LLM Stage
    llm_queue.join()
    for thread in llm_threads:
        thread.join()
    llm_done.set()
    monitor.mark_done("llm")

    # Conversion and Persistence Stage
    bookinfo_queue.join()
    for thread in detailed_threads:
        thread.join()
    monitor.mark_done("bookinfo")

    persist_queue.join()
    for thread in persist_threads:
        thread.join()
    monitor.mark_done("detailed")
    monitor.mark_done("persist")

    metadata_queue.join()
    for thread in metadata_threads:
        thread.join()
    monitor.mark_done("metadata")

    bookinfo_results: list[DetailedBookInfoResult] = []
    while not completed_queue.empty():
        item = completed_queue.get()
        if item is not None and cfg.collect_results:
            payload = getattr(item, "obj", item)
            bookinfo_results.append(payload)
        completed_queue.task_done()
    completed_queue.join()
    monitor.mark_done("completed")
    monitor.stop()

    return bookinfo_results
