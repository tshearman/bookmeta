import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterable, Literal

import joblib
import ollama

from bookmeta.config.settings import DEFAULT_DB_PATH, PIPELINE_CACHE_DIR
from bookmeta.data.sqlite import (
    _compute_pdf_hash,
    persist_run,
    serialize_pipeline_config,
)
from bookmeta.services.bookinfo import BOOK_PROMPT, BOOK_PROMPT_TTRPG
from bookmeta.services.bookinfo.blocks import ContextLimits
from bookmeta.services.bookinfo.book_info import DetailedBookInfo
from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.bookinfo.pipeline import (
    BookInfoPipelineConfig,
)
from bookmeta.services.bookinfo.pipeline import (
    generate_pipeline as generate_bookinfo_pipeline,
)
from bookmeta.services.booksearch import BookSearchMethod
from bookmeta.services.booksearch.pipeline import (
    BookSearchPipelineConfig,
)
from bookmeta.services.booksearch.pipeline import (
    generate_pipeline as generate_booksearch_pipeline,
)
from bookmeta.services.booksearch.providers.googlebooks import (
    GoogleBooksClientConfig,
    googlebooks_search,
)
from bookmeta.services.booksearch.providers.hardcover import (
    HardcoverClientConfig,
    hardcover_search,
)
from bookmeta.services.ocr.ocr import (
    native_ocr_method,
    ollama_ocr_method,
    tesseract_ocr_method,
)
from bookmeta.services.ocr.pdf_ocr_results import PdfOcrResults
from bookmeta.services.ocr.pipeline import OcrPipelineConfig
from bookmeta.services.ocr.pipeline import generate_pipeline as generate_ocr_pipeline
from bookmeta.services.rank.pipeline import (
    BookInfoSelectionPipelineConfig,
    generate_selection_pipeline,
)


@dataclass
class PipelineConfig:
    ocr_config: OcrPipelineConfig
    extraction_config: BookInfoPipelineConfig
    selection_config: BookInfoSelectionPipelineConfig
    booksearch_config: BookSearchPipelineConfig
    mode: Literal["full", "bookinfo_only"] = "full"
    queue_size: int = 16
    ocr_workers: int = 2
    bookinfo_workers: int = 2
    result_workers: int = 1


@dataclass
class PdfTask:
    pdf_path: Path


@dataclass
class OcrOutput:
    pdf_path: Path
    ocr_results: PdfOcrResults


@dataclass
class BookInfoOutput:
    pdf_path: Path
    info: BookInfoResponse


@dataclass
class Failure:
    pdf_path: Path
    stage: str
    error: str


@dataclass
class Result:
    pdf_path: Path
    detailed: DetailedBookInfo | None
    failure: Failure | None = None


@dataclass
class StageMetrics:
    name: str
    start_time: float = field(default_factory=time.time)
    count: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)

    def increment(self) -> int:
        with self.lock:
            self.count += 1
            return self.count

    def summary(self) -> str:
        with self.lock:
            elapsed = max(1e-6, time.time() - self.start_time)
            rate = self.count / elapsed
            return f"{self.name}: {self.count} ({rate:.2f}/s)"


@dataclass
class PipelineStage:
    """Worker stage that pulls items from an input queue, processes them, and pushes to an output queue."""

    name: str
    handler: Callable[[Any], Any | None]
    input_queue: Queue
    output_queue: Queue
    workers: int = 1
    propagate_failures: bool = True
    upstream_done: Event | None = None
    timeout_seconds: float = 15.0

    def start(self) -> list[Thread]:
        threads: list[Thread] = []
        for _ in range(max(1, self.workers)):
            thread = Thread(target=self._worker, daemon=True)
            thread.start()
            threads.append(thread)
        return threads

    def _worker(self) -> None:
        while True:
            try:
                item = self.input_queue.get(timeout=self.timeout_seconds)
            except Empty:
                if (
                    self.upstream_done
                    and self.upstream_done.is_set()
                    and self.input_queue.empty()
                ):
                    return
                continue
            try:
                if isinstance(item, Failure):
                    if self.propagate_failures:
                        self.output_queue.put(item)
                    continue
                try:
                    result = self.handler(item)
                except Exception as exc:  # pragma: no cover - defensive
                    pdf_path = getattr(item, "pdf_path", None)
                    error = str(exc)
                    LOGGER.exception(f"{self.name} stage failed for {pdf_path}: {exc}")
                    self.output_queue.put(
                        Failure(
                            pdf_path=pdf_path or Path(""), stage=self.name, error=error
                        )
                    )
                    continue
                if result is not None:
                    self.output_queue.put(result)
            finally:
                self.input_queue.task_done()


class NoOcrTextError(RuntimeError):
    """Raised when no OCR method extracted any text from a PDF."""


LOGGER = logging.getLogger(__name__)
BookMetaPipeline = Callable[[Path], DetailedBookInfo]

PDF_MEMORY = joblib.Memory(PIPELINE_CACHE_DIR, verbose=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a PDF and produce refined book metadata."
    )
    parser.add_argument(
        "pdf_path",
        nargs="+",
        type=Path,
        help="Path(s) to the input PDF(s). Supports globs like 'files/books_*'.",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.json"),
        help="Path to secrets JSON with API keys.",
    )
    parser.add_argument(
        "--num-front-ocr-pages",
        type=int,
        default=5,
        help="Number of pages from the front of the document to process.",
    )
    parser.add_argument(
        "--num-back-ocr-pages",
        type=int,
        default=3,
        help="Number of pages from the back of the document to process.",
    )
    parser.add_argument(
        "--ocr-ollama-model",
        type=str,
        default=None,
        help="Provider for the initial BookInfo extraction stage.",
    )
    parser.add_argument(
        "--extraction-provider",
        choices=("openai", "ollama"),
        default="openai",
        help="Provider for the initial BookInfo extraction stage.",
    )
    parser.add_argument(
        "--extraction-model",
        default=None,
        help="Optional model override for the BookInfo extraction stage.",
    )
    parser.add_argument(
        "--selection-provider",
        choices=("openai", "ollama"),
        default="openai",
        help="Provider for the BookInfo selection stage.",
    )
    parser.add_argument(
        "--selection-model",
        default=None,
        help="Optional model override for the BookInfo selection stage.",
    )
    parser.add_argument(
        "--search-max-results",
        type=int,
        default=3,
        help="Maximum Book Search results to fetch during book search.",
    )
    parser.add_argument(
        "--context-first-images",
        type=int,
        default=None,
        help="Number of first page images to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--context-last-images",
        type=int,
        default=None,
        help="Number of last page images to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--context-first-ocr-pages",
        type=int,
        default=None,
        help="Number of first OCR pages to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--context-last-ocr-pages",
        type=int,
        default=None,
        help="Number of last OCR pages to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite DB used to log pipeline runs.",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        help="Logging level (DEBUG, INFO, WARN, ERROR).",
    )
    parser.add_argument(
        "--book-prompt",
        choices=("default", "ttrpg"),
        default="default",
        help="BookInfo prompt variant to use (default or ttrpg-focused).",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("full", "bookinfo-only"),
        default="full",
        help="Choose full pipeline (with search/selection) or bookinfo-only.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=16,
        help="Max items buffered per pipeline stage (bookinfo-only mode).",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Worker threads for OCR stage (bookinfo-only mode).",
    )
    parser.add_argument(
        "--bookinfo-workers",
        type=int,
        default=None,
        help="Worker threads for BookInfo stage (bookinfo-only mode). Defaults to ocr-workers.",
    )
    parser.add_argument(
        "--result-workers",
        type=int,
        default=None,
        help="Worker threads for result stage (bookinfo-only mode). Defaults to ocr-workers.",
    )
    args = parser.parse_args()
    for name in (
        "context_first_images",
        "context_last_images",
        "context_first_ocr_pages",
        "context_last_ocr_pages",
        "queue_size",
        "ocr_workers",
        "bookinfo_workers",
        "result_workers",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    return args


def _read_secrets(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def _client_config_for(provider: str, secrets: dict[str, Any]) -> dict[str, Any]:
    if provider == "openai":
        api_key = secrets.get("OPENAI_API_KEY")
        project = secrets.get("OPENAI_PROJECT_ID")
        return {"api_key": api_key, "project": project}
    if provider == "ollama":
        host = secrets.get("OLLAMA_HOST")
        return {"host": host}
    raise ValueError(f"Unsupported provider: {provider}")


def _has_ocr_text(ocr_results: PdfOcrResults) -> bool:
    combined = (ocr_results.combined_text or "").strip()
    if combined:
        return True
    for page in ocr_results.pages:
        for result in page.ocr_results:
            text = result.text
            if text and text.strip():
                return True
    return False


def _default_booksearch_methods(
    args: argparse.Namespace, secrets: dict[str, Any]
) -> list[BookSearchMethod]:

    methods: list[BookSearchMethod] = []
    if "GOOGLE_BOOKS_API_KEY" in secrets:
        api_key = secrets["GOOGLE_BOOKS_API_KEY"]
        methods.append(
            googlebooks_search(
                GoogleBooksClientConfig(
                    api_key=api_key, max_results=args.search_max_results
                )
            )
        )

    # if "HARDCOVER_API_KEY" in secrets:
    #     api_key = secrets["HARDCOVER_API_KEY"]
    #     methods.append(
    #         hardcover_search(
    #             HardcoverClientConfig(api_key=api_key, per_page=args.search_max_results)
    #         )
    #     )

    return methods


def build_pipeline_config(
    args: argparse.Namespace, secrets: dict[str, Any]
) -> PipelineConfig:

    prompt = BOOK_PROMPT_TTRPG if args.book_prompt == "ttrpg" else BOOK_PROMPT
    context_limits = ContextLimits(
        num_first_images=args.context_first_images,
        num_last_images=args.context_last_images,
        num_first_ocr_pages=args.context_first_ocr_pages,
        num_last_ocr_pages=args.context_last_ocr_pages,
    )
    LOGGER.debug(
        "Configured context limits: first_images=%s last_images=%s "
        "first_ocr_pages=%s last_ocr_pages=%s",
        context_limits.num_first_images,
        context_limits.num_last_images,
        context_limits.num_first_ocr_pages,
        context_limits.num_last_ocr_pages,
    )

    ocr_methods = [native_ocr_method, tesseract_ocr_method]
    if args.ocr_ollama_model is not None:
        llm_ocr = ollama_ocr_method(
            ollama.Client(secrets["OLLAMA_HOST"]), args.ocr_ollama_model
        )
        ocr_methods.append(llm_ocr)  # type: ignore

    ocr_config = OcrPipelineConfig(ocr_methods=ocr_methods)
    extraction_config = BookInfoPipelineConfig(
        provider=args.extraction_provider,
        client_config=_client_config_for(args.extraction_provider, secrets),
        model=args.extraction_model,
        context_limits=context_limits,
        prompt=prompt,
    )
    selection_config = BookInfoSelectionPipelineConfig(
        provider=args.selection_provider,
        client_config=_client_config_for(args.selection_provider, secrets),
        model=args.selection_model,
        context_limits=context_limits,
    )
    search_methods = _default_booksearch_methods(args, secrets)
    booksearch_config = BookSearchPipelineConfig(
        search_methods=search_methods, num_responses=args.search_max_results
    )
    mode = "bookinfo_only" if args.pipeline_mode == "bookinfo-only" else "full"
    queue_size = max(1, args.queue_size)
    ocr_workers = max(1, args.ocr_workers)
    bookinfo_workers = max(1, args.bookinfo_workers or ocr_workers)
    result_workers = max(1, args.result_workers or ocr_workers)

    return PipelineConfig(
        ocr_config=ocr_config,
        extraction_config=extraction_config,
        selection_config=selection_config,
        booksearch_config=booksearch_config,
        mode=mode,
        queue_size=queue_size,
        ocr_workers=ocr_workers,
        bookinfo_workers=bookinfo_workers,
        result_workers=result_workers,
    )


def _full_pipeline(config: PipelineConfig) -> BookMetaPipeline:
    ocr_pipeline = generate_ocr_pipeline(config.ocr_config)
    info_pipeline = generate_bookinfo_pipeline(config.extraction_config)
    search_pipeline = generate_booksearch_pipeline(config.booksearch_config)
    selection_pipeline = generate_selection_pipeline(config.selection_config)

    def _inner_(pdf_path: Path) -> DetailedBookInfo:
        LOGGER.info(f"\n\nFULL PIPELINE::::::::::::::::::\n\t{str(pdf_path)}\n")
        ocr_results = ocr_pipeline(pdf_path)
        if not _has_ocr_text(ocr_results):
            LOGGER.warning(f"Skipping {pdf_path} because OCR produced no text.")
            raise NoOcrTextError(f"No OCR text extracted for {pdf_path}")
        search_results = search_pipeline(info_pipeline(ocr_results))
        return selection_pipeline(ocr_results, search_results)

    return _inner_


def bookinfo_only_pipeline(config: PipelineConfig) -> BookMetaPipeline:
    """Simplified pipeline that stops after OCR + BookInfo extraction."""

    ocr_pipeline = generate_ocr_pipeline(config.ocr_config)
    info_pipeline = generate_bookinfo_pipeline(config.extraction_config)

    def _inner_(pdf_path: Path) -> DetailedBookInfo:
        ocr_results = ocr_pipeline(pdf_path)
        if not _has_ocr_text(ocr_results):
            LOGGER.warning(f"Skipping {pdf_path} because OCR produced no text.")
            raise NoOcrTextError(f"No OCR text extracted for {pdf_path}")
        info_response = info_pipeline(ocr_results)
        if info_response is None:
            raise RuntimeError(f"BookInfo extraction returned no result for {pdf_path}")
        # Promote BookInfoResponse to DetailedBookInfo shape for signature parity.
        return info_response.info.as_detailed_book_info()

    return _inner_


def pipeline(config: PipelineConfig) -> BookMetaPipeline:
    """Return the configured pipeline (full or bookinfo-only)."""
    if config.mode == "bookinfo_only":
        return bookinfo_only_pipeline(config)
    return _full_pipeline(config)


def _init_bookinfo_queues(queue_size: int) -> tuple[Queue, Queue, Queue, Queue]:
    size = max(1, queue_size)
    return (
        Queue(maxsize=size),
        Queue(maxsize=size),
        Queue(maxsize=size),
        Queue(maxsize=size),
    )


def _bookinfo_only_handlers(
    ocr_runner: Callable[[Path], PdfOcrResults],
    info_runner: Callable[[PdfOcrResults], BookInfoResponse | None],
    results_db: Path,
    config: PipelineConfig,
    *,
    ocr_metrics: StageMetrics,
    bookinfo_metrics: StageMetrics,
    persist_metrics: StageMetrics,
) -> tuple[
    Callable[[PdfTask], OcrOutput | Failure | None],
    Callable[[OcrOutput], BookInfoOutput | Failure | None],
    Callable[[BookInfoOutput], Result | Failure | None],
]:
    def ocr_handler(task: PdfTask) -> OcrOutput | Failure | None:
        ocr_results = ocr_runner(task.pdf_path)
        ocr_metrics.increment()
        if not _has_ocr_text(ocr_results):
            LOGGER.warning(f"Skipping {task.pdf_path} because OCR produced no text.")
            return Failure(pdf_path=task.pdf_path, stage="ocr", error="no_ocr_text")
        LOGGER.info("STAGE ocr: %s processed", task.pdf_path.name[-50:])
        return OcrOutput(pdf_path=task.pdf_path, ocr_results=ocr_results)

    def bookinfo_handler(output: OcrOutput) -> BookInfoOutput | Failure | None:
        info_response = info_runner(output.ocr_results)
        bookinfo_metrics.increment()
        if info_response is None:
            LOGGER.warning(
                f"BookInfo extraction returned no result for {output.pdf_path}"
            )
            return Failure(
                pdf_path=output.pdf_path, stage="bookinfo", error="no_bookinfo_result"
            )
        LOGGER.info("STAGE bookinfo: %s processed", output.pdf_path.name[-50:])
        return BookInfoOutput(pdf_path=output.pdf_path, info=info_response)

    def persist_handler(item: BookInfoOutput) -> Result | Failure | None:
        try:
            detailed = item.info.info.as_detailed_book_info()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception(f"Failed to convert BookInfo for {item.pdf_path}: {exc}")
            return Failure(pdf_path=item.pdf_path, stage="bookinfo", error=str(exc))
        try:
            LOGGER.debug(f"Persisting pipeline run for {item.pdf_path}")
            persist_run(results_db, item.pdf_path, config, detailed)
            LOGGER.debug(f"Persisted pipeline run for {item.pdf_path}")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception(f"Failed to persist pipeline run for {item.pdf_path}")
            return Failure(pdf_path=item.pdf_path, stage="persist", error=str(exc))
        persist_metrics.increment()
        LOGGER.info(
            "STAGE persist: %s processed: \n%s",
            item.pdf_path.name[-50:],
            detailed.model_dump_json(indent=2, ensure_ascii=False),
        )
        return Result(pdf_path=item.pdf_path, detailed=detailed, failure=None)

    return ocr_handler, bookinfo_handler, persist_handler


def _start_stages(stages: Iterable[PipelineStage]) -> list[Thread]:
    threads: list[Thread] = []
    for stage in stages:
        threads.extend(stage.start())
    return threads


def run_bookinfo_only_pipeline(
    pdf_paths: Iterable[Path],
    config: PipelineConfig,
    results_db: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    total_expected: int | None = None,
    enqueue_limit: int | None = None,
    dedupe: bool = True,
) -> list[Result]:
    """
    Execute bookinfo-only pipeline with staged worker queues:
    PdfTask -> OCR -> BookInfo -> persistence/Result.
    """

    producer_paths = iter(pdf_paths)
    enqueued_lock = Lock()
    enqueued_count = 0
    seen_hashes: set[str] = set()
    total_pdfs = total_expected or enqueue_limit

    # If nothing to enqueue, short-circuit early.
    try:
        first = next(producer_paths)
    except StopIteration:
        return []
    pdf_queue, ocr_queue, bookinfo_queue, final_queue = _init_bookinfo_queues(
        config.queue_size
    )

    ocr_runner = generate_ocr_pipeline(config.ocr_config)
    info_runner = generate_bookinfo_pipeline(config.extraction_config)
    ocr_metrics = StageMetrics("ocr")
    bookinfo_metrics = StageMetrics("bookinfo")
    persist_metrics = StageMetrics("persist")
    ocr_handler, bookinfo_handler, persist_handler = _bookinfo_only_handlers(
        ocr_runner,
        info_runner,
        results_db,
        config,
        ocr_metrics=ocr_metrics,
        bookinfo_metrics=bookinfo_metrics,
        persist_metrics=persist_metrics,
    )

    results: list[Result] = []
    failures: list[Failure] = []
    result_lock = Lock()
    processed_count = 0

    producer_done = Event()
    ocr_done = Event()
    bookinfo_done = Event()
    persist_done = Event()

    ocr_stage = PipelineStage(
        name="ocr",
        handler=ocr_handler,
        input_queue=pdf_queue,
        output_queue=ocr_queue,
        workers=config.ocr_workers,
        upstream_done=producer_done,
    )
    bookinfo_stage = PipelineStage(
        name="bookinfo",
        handler=bookinfo_handler,
        input_queue=ocr_queue,
        output_queue=bookinfo_queue,
        workers=config.bookinfo_workers,
        upstream_done=ocr_done,
    )
    persist_stage = PipelineStage(
        name="persist",
        handler=persist_handler,
        input_queue=bookinfo_queue,
        output_queue=final_queue,
        workers=config.result_workers,
        upstream_done=bookinfo_done,
    )

    stop_event = Event()

    def _monitor() -> None:
        logged_idle = False
        while not stop_event.wait(60):
            with result_lock:
                collected = processed_count
                target_total = total_pdfs or enqueued_count or collected
            ocr_sz = ocr_queue.qsize()
            book_sz = bookinfo_queue.qsize()
            final_sz = final_queue.qsize()
            total_count = (
                ocr_metrics.count + bookinfo_metrics.count + persist_metrics.count
            )
            if total_count == 0 and ocr_sz == 0 and book_sz == 0 and final_sz == 0:
                if not logged_idle:
                    LOGGER.debug("Pipeline waiting for work; queues empty")
                    logged_idle = True
                continue
            logged_idle = False
            LOGGER.debug(
                "Pipeline progress | %s | %s | %s | queues: ocr=%d bookinfo=%d final=%d collected=%d/%d",
                ocr_metrics.summary(),
                bookinfo_metrics.summary(),
                persist_metrics.summary(),
                ocr_sz,
                book_sz,
                final_sz,
                collected,
                target_total,
            )

    ocr_threads = ocr_stage.start()
    bookinfo_threads = bookinfo_stage.start()
    persist_threads = persist_stage.start()
    monitor = Thread(target=_monitor, daemon=True)
    monitor.start()

    report_rate = 100

    def _collector() -> None:
        nonlocal processed_count
        idle_ticks = 0

        def _total_target() -> int:
            return total_pdfs or enqueued_count or processed_count

        while True:
            try:
                item = final_queue.get(timeout=30)
            except Empty:
                if persist_done.is_set() and final_queue.empty():
                    break
                idle_ticks += 1
                continue
            try:
                if isinstance(item, Failure):
                    with result_lock:
                        failures.append(item)
                        results.append(
                            Result(pdf_path=item.pdf_path, detailed=None, failure=item)
                        )
                        processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count, _total_target())
                    LOGGER.warning(
                        f"Pipeline failed for {item.pdf_path} at stage={item.stage}: {item.error}"
                    )
                    if processed_count % report_rate == 0:
                        LOGGER.debug(
                            f"Collected {processed_count} results (including failures)"
                        )
                    continue
                if isinstance(item, Result):
                    if item.failure:
                        with result_lock:
                            failures.append(item.failure)
                            results.append(item)
                            processed_count += 1
                        if progress_callback:
                            progress_callback(processed_count, _total_target())
                        LOGGER.warning(
                            f"Pipeline failed for {item.pdf_path} at stage={item.failure.stage}: {item.failure.error}"
                        )
                        if processed_count % report_rate == 0:
                            LOGGER.debug(
                                f"Collected {processed_count} results (including failures)"
                            )
                        continue
                    if item.detailed:
                        with result_lock:
                            results.append(item)
                            processed_count += 1
                        LOGGER.info("RESULT: %s generated", item.pdf_path)
                        if progress_callback:
                            progress_callback(processed_count, _total_target())
                        if processed_count % report_rate == 0:
                            LOGGER.debug(
                                f"Collected {processed_count} results (including failures)"
                            )
            finally:
                final_queue.task_done()

    collector = Thread(target=_collector, daemon=True)
    collector.start()

    def _producer() -> None:
        nonlocal enqueued_count, total_pdfs
        # push the first item already peeked
        for pdf_path in chain((first,), producer_paths):
            if enqueue_limit is not None and enqueued_count >= enqueue_limit:
                break
            try:
                pdf_hash = _compute_pdf_hash(pdf_path)
            except Exception as exc:
                LOGGER.warning(f"Skipping {pdf_path} due to hash error: {exc}")
                continue
            if dedupe and pdf_hash in seen_hashes:
                continue
            if dedupe:
                seen_hashes.add(pdf_hash)
            with enqueued_lock:
                enqueued_count += 1
            LOGGER.info("QUEUE: %s enqueued", pdf_path)
            pdf_queue.put(PdfTask(pdf_path))
        if total_pdfs is None:
            total_pdfs = enqueued_count
        producer_done.set()

    producer_thread = Thread(target=_producer, daemon=True)
    producer_thread.start()

    # Wait for producer to finish enqueueing, then drain queues while allowing stages to exit via timeouts.
    producer_thread.join()
    pdf_queue.join()
    for thread in ocr_threads:
        thread.join()
    ocr_done.set()

    ocr_queue.join()
    for thread in bookinfo_threads:
        thread.join()
    bookinfo_done.set()

    bookinfo_queue.join()
    for thread in persist_threads:
        thread.join()
    persist_done.set()

    final_queue.join()

    stop_event.set()
    monitor.join()
    collector.join()

    if failures:
        LOGGER.info(f"Encountered {len(failures)} failure(s) during bookinfo-only run")

    return results


@PDF_MEMORY.cache(ignore=["config"])
def execute_pipeline(
    pdf: Path, config: PipelineConfig, config_signature: str
) -> DetailedBookInfo:
    return pipeline(config)(pdf)


def process_pdf(
    pdf: Path,
    config: PipelineConfig,
    results_db: Path,
) -> DetailedBookInfo | None:
    LOGGER.info(f"Running pipeline on {pdf}")
    config_signature = json.dumps(serialize_pipeline_config(config), sort_keys=True)
    try:
        final_info = execute_pipeline(pdf, config, config_signature)
    except NoOcrTextError as exc:
        LOGGER.warning(f"Skipping {pdf}: {exc}")
        return

    LOGGER.info(f"Final BookInfo for {pdf}:\n{final_info}")
    try:
        persist_run(results_db, pdf, config, final_info)
        LOGGER.info(f"Persisted pipeline run to {results_db} for {pdf}")
    except Exception:
        LOGGER.exception(f"Failed to persist pipeline run for {pdf}")
    return final_info


def _expand_paths(patterns: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for pattern in patterns:
        pattern = pattern.expanduser()
        text = str(pattern)
        if any(ch in text for ch in "*?[]"):
            matches = list(Path().glob(text))
            if not matches:
                LOGGER.warning(f"No files matched pattern: {pattern}")
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    return expanded


def main() -> list[DetailedBookInfo]:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    config = build_pipeline_config(args, secrets)
    pdf_paths = _expand_paths(args.pdf_path)
    if not pdf_paths:
        raise FileNotFoundError("No PDF paths matched the provided arguments.")

    if config.mode == "bookinfo_only":
        results = run_bookinfo_only_pipeline(pdf_paths, config, args.results_db)
        detailed_results: list[DetailedBookInfo] = []
        for result in results:
            if result.detailed:
                print(json.dumps(result.detailed.model_dump(), indent=2))
                detailed_results.append(result.detailed)
        return detailed_results

    results: list[DetailedBookInfo] = []
    for pdf_path in pdf_paths:
        result = process_pdf(pdf_path, config, args.results_db)
        if result is not None:
            print(json.dumps(result.model_dump(), indent=2))
            results.append(result)
    return results


if __name__ == "__main__":
    out = main()
