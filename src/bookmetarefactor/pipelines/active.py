import json
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Iterable, Literal

from bookmetarefactor.ocr import OcrConfig
from bookmetarefactor.pipelines import (
    produce_pdfs,
    start_extraction_pipeline,
    start_llm_pipeline,
    start_ocr_pipeline,
)
from bookmetarefactor.types.bookinfo import BookInfoResult
from bookmetarefactor.types.extraction import (
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
    stage_timeout: float | None = None
    mode: Literal["llm", "batch"] = "llm"


def load_openai_provider_config(secrets_path: Path) -> ProviderConfig:
    """Load OpenAI provider settings from a secrets.json file."""
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    api_key = secrets.get("OPENAI_API_KEY")
    project = secrets.get("OPENAI_PROJECT_ID")
    if not api_key:
        raise ValueError(f"Missing OPENAI_API_KEY in {secrets_path}")
    client_config = {"api_key": api_key}
    if project:
        client_config["project"] = project
    return ProviderConfig(
        provider="openai", model="gpt-4.1", client_config=client_config
    )


def run_pipeline(cfg: PipelineConfig) -> list[BookInfoResult]:
    """
    Orchestrate the full pipeline from PDF discovery through LLM execution or batching.

    Returns (bookinfo_results, llm_task_batches). Only one of the outputs will be populated
    depending on cfg.mode ("llm" for immediate execution, "batch" for offline submission).
    """

    provider_config = cfg.provider_config

    extraction_config = ExtractionConfig(
        prompt=cfg.prompt,
        provider_config=provider_config,
        context_limits=cfg.context_limits,
    )

    pdf_queue: Queue = Queue(maxsize=cfg.queue_size)
    extraction_queue: Queue = Queue(maxsize=cfg.queue_size)
    llm_queue: Queue = Queue(maxsize=cfg.queue_size)
    bookinfo_queue: Queue = Queue(maxsize=cfg.queue_size)

    producer_done = Event()
    ocr_done = Event()
    extraction_done = Event()
    llm_done = Event()

    def _produce() -> None:
        produce_pdfs(cfg.roots, pdf_queue, dedupe=cfg.dedupe, limit=cfg.limit)
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
    )

    # Producer Stage
    producer_thread.join()

    # OCR Stages
    pdf_queue.join()
    for thread in ocr_threads:
        thread.join()
    ocr_done.set()

    # Extraction Stage
    extraction_queue.join()
    for thread in extraction_threads:
        thread.join()
    extraction_done.set()

    # LLM Stage
    llm_queue.join()
    for thread in llm_threads:
        thread.join()
    llm_done.set()

    # Bookinfo Stage
    bookinfo_results: list[BookInfoResult] = []
    while not bookinfo_queue.empty():
        item = bookinfo_queue.get()
        if item is not None:
            bookinfo_results.append(item)
        bookinfo_queue.task_done()
    bookinfo_queue.join()

    return bookinfo_results
