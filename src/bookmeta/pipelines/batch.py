import hashlib
import json
import logging
from pathlib import Path
from queue import Queue
from threading import Event, Thread

from bookmeta.config.pipeline import PipelineConfig
from bookmeta.monitoring import QueueMonitor
from bookmeta.ocr import OcrConfig
from bookmeta.persistence import (
    has_persisted_batch_result,
    serialize_batch_pipeline_config,
)
from bookmeta.pipelines.batching import start_llm_batch_pipeline
from bookmeta.pipelines.discovery import start_discovery
from bookmeta.pipelines.extraction import start_extraction_pipeline
from bookmeta.pipelines.ocr import start_ocr_pipeline
from bookmeta.pipelines.persist import start_persist_batch_pipeline
from bookmeta.pipelines.queues import make_queue_factory
from bookmeta.types.extraction import ExtractionConfig, PersistedBatch

LOGGER = logging.getLogger(__name__)


def _pipeline_hash(
    cfg: PipelineConfig, extraction_config: ExtractionConfig, ocr_config: OcrConfig
) -> str:
    payload = {
        "extraction_hash": extraction_config.hash,
        "ocr": {
            "num_first_pages": ocr_config.num_first_pages,
            "num_last_pages": ocr_config.num_last_pages,
            "methods": sorted(method.name for method in ocr_config.methods),
        },
        "roots": sorted(str(Path(root)) for root in cfg.runtime.roots),
        "dedupe": cfg.runtime.dedupe,
        "limit": cfg.runtime.limit,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def run_pipeline(cfg: PipelineConfig) -> list[PersistedBatch]:
    """
    Orchestrate the full pipeline from PDF discovery through batching persistence.

    Returns persisted batches generated from the input PDFs.
    """

    provider_config = cfg.pdf.provider_config

    ocr_methods = tuple(cfg.pdf.ocr_config.methods)
    ocr_config = OcrConfig(
        num_first_pages=cfg.pdf.ocr_config.num_first_pages,
        num_last_pages=cfg.pdf.ocr_config.num_last_pages,
        methods=ocr_methods,
    )

    extraction_config = ExtractionConfig(
        prompt=cfg.pdf.prompt,
        provider_config=provider_config,
        context_limits=cfg.pdf.context_limits,
    )
    pipeline_hash = _pipeline_hash(cfg, extraction_config, ocr_config)
    pipeline_config_json = serialize_batch_pipeline_config(cfg)
    results_db = cfg.runtime.results_db

    monitor = QueueMonitor(
        title="BookMeta Queue Monitor",
        refresh_interval=0.2,
        enabled=cfg.runtime.monitor_queues,
    )
    monitor.start()

    make_queue = make_queue_factory(cfg.runtime.queue_size, monitor)

    pdf_queue: Queue = make_queue("ocr", cfg.runtime.pdf_queue_size)
    extraction_queue: Queue = make_queue(
        "extraction", cfg.runtime.extraction_queue_size
    )
    llm_queue: Queue = make_queue("batching", cfg.runtime.llm_queue_size)
    batch_queue: Queue = make_queue("persist", cfg.runtime.batch_queue_size)
    persisted_queue: Queue = make_queue("completed", cfg.runtime.persist_queue_size)

    discovery_done = Event()
    ocr_done = Event()
    extraction_done = Event()
    batch_thread: Thread | None = None
    persist_thread: Thread | None = None
    batch_done = Event()

    skip_processed = None
    if cfg.runtime.resume:

        def _skip_processed(pdf) -> bool:
            return has_persisted_batch_result(
                results_db, pdf.hash, config_hash=pipeline_hash
            )

        skip_processed = _skip_processed

    producer_thread = start_discovery(
        cfg.runtime.roots,
        pdf_queue,
        dedupe=cfg.runtime.dedupe,
        limit=cfg.runtime.limit,
        done_event=discovery_done,
        skip_processed=skip_processed,
    )

    ocr_threads = start_ocr_pipeline(
        pdf_queue,
        extraction_queue,
        ocr_config,
        extraction_config,
        workers=cfg.runtime.ocr_workers,
        upstream_done=discovery_done,
        timeout=cfg.runtime.stage_timeout,
    )

    extraction_threads = start_extraction_pipeline(
        extraction_queue,
        llm_queue,
        workers=cfg.runtime.extraction_workers,
        upstream_done=ocr_done,
        timeout=cfg.runtime.stage_timeout,
    )

    batch_thread = start_llm_batch_pipeline(
        llm_queue,
        batch_queue,
        upstream_done=extraction_done,
        timeout=cfg.runtime.stage_timeout,
    )

    persist_thread = start_persist_batch_pipeline(
        batch_queue,
        persisted_queue,
        cfg.runtime.batch_output_dir,
        pipeline_hash,
        db_path=results_db,
        pipeline_config=pipeline_config_json,
        upstream_done=batch_done,
        timeout=cfg.runtime.stage_timeout,
    )

    # Producer Stage
    producer_thread.join()
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
    llm_queue.join()
    batch_thread.join()
    batch_done.set()
    monitor.mark_done("batching")
    batch_queue.join()
    persist_thread.join()
    monitor.mark_done("persist")

    # Collect outputs
    persisted_batches: list[PersistedBatch] = []
    while not persisted_queue.empty():
        batch = persisted_queue.get()
        try:
            from bookmeta.monitoring import TimedItem

            if isinstance(batch, TimedItem):
                batch_obj = batch.obj
            else:
                batch_obj = batch
            if batch_obj:
                persisted_batches.append(batch_obj)
        finally:
            persisted_queue.task_done()
    persisted_queue.join()
    monitor.mark_done("completed")

    monitor.stop()

    return persisted_batches
