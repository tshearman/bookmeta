import argparse
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, Iterable
from openai import OpenAI
from bookmeta.cli.utils import discover_pdfs
from bookmeta.config.settings import DEFAULT_DB_PATH
from bookmeta.data.sqlite import _compute_pdf_hash
from bookmeta.services.bookinfo import BOOK_PROMPT
from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.bookinfo.blocks import (
    ContextLimits,
    construct_blocks,
    get_img_blocks,
    get_text_blocks,
)
from bookmeta.services.ocr.pdf_ocr_results import PdfOcrResults
from bookmeta.services.llm import LLM_MEMORY
from bookmeta.services.ocr.pipeline import OcrPipelineConfig, generate_pipeline
from bookmeta.services.ocr.rendering import img_to_url
from bookmeta.cli.openai_batch_collector import run_collector

LOGGER = logging.getLogger(__name__)


def _read_secrets(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    return json.loads(path.read_text())


def _expand_dirs(patterns: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        pattern = pattern.expanduser()
        if any(ch in str(pattern) for ch in "*?[]"):
            out.extend(Path().glob(str(pattern)))
        else:
            out.append(pattern)
    return [p for p in out if p.exists()]


@dataclass
class BatchConfig:
    input_dirs: list[Path]
    output_dir: Path
    max_pdfs: int | None
    ocr_workers: int
    request_workers: int
    submit_workers: int
    model: str
    batch_size: int
    queue_size: int
    secrets: dict
    completion_window: str = "24h"
    api_endpoint: str = "/v1/chat/completions"
    timeout_seconds: float = 300.0
    context_limits: ContextLimits = ContextLimits()


@dataclass
class PdfTask:
    path: Path


@dataclass
class OcrResultItem:
    path: Path
    ocr: PdfOcrResults


@dataclass
class RequestBatch:
    request_path: Path
    mapping_path: Path
    count: int
    hashes: list[str]
    request_hash: str
    mapping_hash: str


@dataclass
class SubmittedBatch:
    request_path: Path
    mapping_path: Path
    response_path: Path
    batch_obj: dict


@dataclass
class CompletedBatch:
    batch_id: str
    mapping_path: Path
    output_path: Path
    batch_obj: dict


@dataclass
class ParsedResult:
    pdf_path: Path
    pdf_hash: str
    detailed: dict


def _ocr_text_from_pdf(
    pdf_path: Path, ocr_runner: Callable[[Path], any]
) -> OcrResultItem | None:
    try:
        ocr = ocr_runner(pdf_path)
    except Exception as exc:
        LOGGER.warning("OCR failed for %s: %s", pdf_path, exc)
        return None
    text = (ocr.combined_text or "").strip()
    if not text:
        LOGGER.info("No OCR text for %s; skipping", pdf_path)
        return None
    return OcrResultItem(path=pdf_path, ocr=ocr)


def _request_line(custom_id: str, model: str, messages: list[dict[str, any]]) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
    }


def _write_request_files(
    batch_items: list[OcrResultItem],
    model: str,
    output_dir: Path,
    limits: ContextLimits,
) -> list[RequestBatch]:
    MAX_REQUESTS_PER_BATCH = 50_000
    MAX_BATCH_BYTES = 100 * 1024 * 1024
    ts = int(time.time())
    token = uuid.uuid4().hex[:8]
    batches: list[RequestBatch] = []

    def flush(chunk: list[OcrResultItem], idx: int) -> RequestBatch:
        base = output_dir / f"batch_{ts}_{token}_p{idx}"
        req_path = base.with_suffix(".requests.jsonl")
        map_path = base.with_suffix(".pdfs.jsonl")
        with (
            req_path.open("w", encoding="utf-8") as req_f,
            map_path.open("w", encoding="utf-8") as map_f,
        ):
            for it in chunk:
                custom_id = _compute_pdf_hash(it.path)
                blocks = construct_blocks(it.ocr, BOOK_PROMPT, limits=limits)
                messages = get_text_blocks(blocks) + [
                    {"type": b["type"], "image_url": img_to_url(b["image"])}
                    for b in get_img_blocks(blocks)
                ]
                req_line = _request_line(custom_id, model, messages)
                req_f.write(json.dumps(req_line, ensure_ascii=False) + "\n")
                map_f.write(
                    json.dumps({"custom_id": custom_id, "pdf_path": str(it.path)})
                    + "\n"
                )
        hashes = [
            _compute_pdf_hash(it.path)
            for it in chunk
            if it and it.path and it.path.exists()
        ]
        request_hash = _compute_pdf_hash(req_path)
        mapping_hash = _compute_pdf_hash(map_path)
        return RequestBatch(
            request_path=req_path,
            mapping_path=map_path,
            count=len(chunk),
            hashes=hashes,
            request_hash=request_hash,
            mapping_hash=mapping_hash,
        )

    current: list[OcrResultItem] = []
    current_bytes = 0
    part_idx = 0
    for item in batch_items:
        custom_id = _compute_pdf_hash(item.path)
        blocks = construct_blocks(item.ocr, BOOK_PROMPT, limits=limits)
        messages = get_text_blocks(blocks) + [
            {"type": b["type"], "image_url": img_to_url(b["image"])}
            for b in get_img_blocks(blocks)
        ]
        req_line = _request_line(custom_id, model, messages)
        map_line = {"custom_id": custom_id, "pdf_path": str(item.path)}
        line_bytes = len(
            (json.dumps(req_line, ensure_ascii=False) + "\n").encode("utf-8")
        )
        line_bytes += len((json.dumps(map_line) + "\n").encode("utf-8"))

        if current and (
            len(current) >= MAX_REQUESTS_PER_BATCH
            or current_bytes + line_bytes > MAX_BATCH_BYTES
        ):
            batches.append(flush(current, part_idx))
            part_idx += 1
            current = []
            current_bytes = 0

        current.append(item)
        current_bytes += line_bytes

    if current:
        batches.append(flush(current, part_idx))

    return batches


@LLM_MEMORY.cache(ignore=["client"])
def _submit_batch(
    client: OpenAI, req_batch: RequestBatch, completion_window: str, endpoint: str
) -> SubmittedBatch:
    with req_batch.request_path.open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    response_path = req_batch.request_path.with_suffix(".batch_response.json")
    response_path.write_text(
        json.dumps(batch.to_dict_recursive(), indent=2), encoding="utf-8"
    )
    return SubmittedBatch(
        request_path=req_batch.request_path,
        mapping_path=req_batch.mapping_path,
        response_path=response_path,
        batch_obj=batch.to_dict_recursive(),
    )


def run_pipeline(cfg: BatchConfig) -> list[SubmittedBatch]:
    pdf_queue: Queue = Queue()
    ocr_queue: Queue = Queue(maxsize=cfg.queue_size)
    request_queue: Queue = Queue(maxsize=cfg.queue_size)
    submit_queue: Queue = Queue(maxsize=cfg.queue_size)
    results: list[SubmittedBatch] = []

    ocr_runner = generate_pipeline(OcrPipelineConfig())
    client = OpenAI(
        api_key=cfg.secrets.get("OPENAI_API_KEY"),
        project=cfg.secrets.get("OPENAI_PROJECT_ID"),
    )

    producer_done = Event()
    ocr_done = Event()
    request_done = Event()

    def producer() -> None:
        seen: set[str] = set()
        count = 0
        for root in cfg.input_dirs:
            for pdf in discover_pdfs(root):
                if cfg.max_pdfs is not None and count >= cfg.max_pdfs:
                    producer_done.set()
                    return
                try:
                    pdf_hash = _compute_pdf_hash(pdf)
                except Exception:
                    continue
                if pdf_hash in seen:
                    continue
                seen.add(pdf_hash)
                pdf_queue.put(PdfTask(pdf))
                LOGGER.info("QUEUE: %s enqueued", pdf.name[-50:])
                count += 1
        producer_done.set()

    def ocr_worker() -> None:
        idle = 0
        while True:
            try:
                task: PdfTask = pdf_queue.get(timeout=cfg.timeout_seconds)
            except Empty:
                if producer_done.is_set() and pdf_queue.empty():
                    break
                idle += 1
                continue
            try:
                result = _ocr_text_from_pdf(task.path, ocr_runner)
                if result:
                    LOGGER.info("STAGE ocr: %s processed", task.path.name[-50:])
                    ocr_queue.put(result)
            finally:
                pdf_queue.task_done()
        ocr_done.set()

    def request_worker() -> None:
        buffer: list[OcrResultItem] = []
        while True:
            try:
                item: OcrResultItem = ocr_queue.get(timeout=cfg.timeout_seconds)
            except Empty:
                if ocr_done.is_set() and ocr_queue.empty():
                    break
                continue
            try:
                buffer.append(item)
                if len(buffer) >= cfg.batch_size:
                    batches = _write_request_files(
                        buffer, cfg.model, cfg.output_dir, cfg.context_limits
                    )
                    for batch in batches:
                        LOGGER.info(
                            "STAGE request: wrote %s and %s",
                            batch.request_path,
                            batch.mapping_path,
                        )
                        request_queue.put(batch)
                    buffer = []
            finally:
                ocr_queue.task_done()
        if buffer:
            batches = _write_request_files(
                buffer, cfg.model, cfg.output_dir, cfg.context_limits
            )
            for batch in batches:
                LOGGER.info(
                    "STAGE request: wrote %s and %s",
                    batch.request_path,
                    batch.mapping_path,
                )
                request_queue.put(batch)
        request_done.set()

    def submit_worker() -> None:
        while True:
            try:
                batch: RequestBatch = request_queue.get(timeout=cfg.timeout_seconds)
            except Empty:
                if request_done.is_set() and request_queue.empty():
                    break
                continue
            try:
                submitted = _submit_batch(
                    client, batch, cfg.completion_window, cfg.api_endpoint
                )
                LOGGER.info("STAGE submit: %s submitted", batch.request_path)
                submit_queue.put(submitted)
            finally:
                request_queue.task_done()

    def collect_worker() -> None:
        idle = 0
        while True:
            try:
                submitted: SubmittedBatch = submit_queue.get(
                    timeout=cfg.timeout_seconds
                )
            except Empty:
                if request_done.is_set() and submit_queue.empty():
                    break
                idle += 1
                continue
            results.append(submitted)
            LOGGER.info("RESULT: %s written", submitted.response_path)
            submit_queue.task_done()

    threads: list[Thread] = []
    threads.append(Thread(target=producer, daemon=True))
    for _ in range(cfg.ocr_workers):
        threads.append(Thread(target=ocr_worker, daemon=True))
    for _ in range(cfg.request_workers):
        threads.append(Thread(target=request_worker, daemon=True))
    for _ in range(cfg.submit_workers):
        threads.append(Thread(target=submit_worker, daemon=True))
    collector = Thread(target=collect_worker, daemon=True)

    for t in threads:
        t.start()
    collector.start()

    for t in threads:
        t.join()
    collector.join()

    return results


def _persist_batch_result(
    db_path: Path, pdf_path: Path, pdf_hash: str, result: dict, model: str
) -> None:
    _ensure_db(db_path)
    pipeline_config = {
        "mode": "openai_batch",
        "model": model,
    }
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    payload = (
        pdf_path.name,
        pdf_hash,
        json.dumps(pipeline_config, indent=2),
        json.dumps(result, indent=2),
        timestamp,
    )
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (pdf_name, pdf_hash, pipeline_config, result, run_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()


def run_collector(
    output_dir: Path,
    client: OpenAI,
    *,
    poll_interval: int = 300,
    db_path: Path = DEFAULT_DB_PATH,
    model: str = "openai-batch",
    watch: bool = False,
    scan_interval: int = 300,
) -> None:
    collect_queue: Queue = Queue(maxsize=128)
    output_queue: Queue = Queue(maxsize=128)
    parse_queue: Queue = Queue(maxsize=128)
    stop_event = Event()
    seen_responses: set[Path] = set()

    def discover_batches() -> None:
        for resp_path in output_dir.glob("*.batch_response.json"):
            if resp_path in seen_responses:
                continue
            try:
                obj = json.loads(resp_path.read_text())
            except Exception:
                continue
            seen_responses.add(resp_path)
            batch_id = obj.get("id")
            status = obj.get("status")
            output_path = resp_path.with_suffix(".batch_output.jsonl")
            if status == "completed" and output_path.exists():
                collect_queue.put(
                    CompletedBatch(
                        batch_id=batch_id,
                        mapping_path=resp_path.with_suffix(".pdfs.jsonl"),
                        output_path=output_path,
                        batch_obj=obj,
                    )
                )
                continue
            if batch_id:
                collect_queue.put(resp_path)

    def poll_worker() -> None:
        while not stop_event.is_set():
            try:
                resp_path = collect_queue.get(timeout=poll_interval)
            except Empty:
                if stop_event.is_set() and collect_queue.empty():
                    break
                continue
            try:
                obj = json.loads(resp_path.read_text())
                batch_id = obj.get("id")
                mapping_path = resp_path.with_suffix(".pdfs.jsonl")
                if not batch_id:
                    continue
                latest = client.batches.retrieve(batch_id)
                status = latest.status
                output_file_id = latest.output_file_id
                latest_path = resp_path
                latest_path.write_text(
                    json.dumps(latest.to_dict_recursive(), indent=2), encoding="utf-8"
                )
                if status != "completed" or not output_file_id:
                    if not stop_event.is_set():
                        time.sleep(poll_interval)
                        collect_queue.put(resp_path)
                    continue
                # download output file
                output_path = resp_path.with_suffix(".batch_output.jsonl")
                content = client.files.content(output_file_id)
                output_path.write_bytes(content.read())
                output_queue.put(
                    CompletedBatch(
                        batch_id=batch_id,
                        mapping_path=mapping_path,
                        output_path=output_path,
                        batch_obj=latest.to_dict_recursive(),
                    )
                )
            finally:
                collect_queue.task_done()

    def parse_worker() -> None:
        while not stop_event.is_set():
            try:
                batch: CompletedBatch = output_queue.get(timeout=poll_interval)
            except Empty:
                if stop_event.is_set() and output_queue.empty():
                    break
                continue
            try:
                mapping: dict[str, str] = {}
                with batch.mapping_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        cid = rec.get("custom_id")
                        path = rec.get("pdf_path")
                        if cid and path:
                            mapping[cid] = path
                with batch.output_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        custom_id = rec.get("custom_id")
                        resp = rec.get("response") or {}
                        choices = resp.get("choices") or []
                        if not custom_id or not choices:
                            continue
                        message = choices[0].get("message") or {}
                        content = message.get("content") or ""
                        try:
                            bir = BookInfoResponse.model_validate_json(content)
                            detailed = bir.info.as_detailed_book_info()
                            pdf_path = Path(mapping.get(custom_id, ""))
                            pdf_hash = custom_id
                            parsed = ParsedResult(
                                pdf_path=pdf_path,
                                pdf_hash=pdf_hash,
                                detailed=detailed.model_dump(),
                            )
                            parse_queue.put(parsed)
                        except Exception as exc:
                            LOGGER.warning(
                                "Failed to parse batch output for %s: %s",
                                custom_id,
                                exc,
                            )
                            continue
            finally:
                output_queue.task_done()

    def persist_worker() -> None:
        while not stop_event.is_set():
            try:
                parsed: ParsedResult = parse_queue.get(timeout=poll_interval)
            except Empty:
                if stop_event.is_set() and parse_queue.empty():
                    break
                continue
            try:
                if parsed.pdf_path:
                    _persist_batch_result(
                        db_path,
                        parsed.pdf_path,
                        parsed.pdf_hash,
                        parsed.detailed,
                        model,
                    )
                    LOGGER.info("Persisted %s", parsed.pdf_path)
            finally:
                parse_queue.task_done()

    def watcher() -> None:
        while not stop_event.is_set():
            discover_batches()
            for _ in range(scan_interval):
                if stop_event.is_set():
                    break
                time.sleep(1)

    def wait_for_enter() -> None:
        try:
            input("Press Enter to stop watching...\n")
        except EOFError:
            pass
        stop_event.set()

    threads: list[Thread] = [
        Thread(target=poll_worker, daemon=True),
        Thread(target=parse_worker, daemon=True),
        Thread(target=persist_worker, daemon=True),
    ]
    discover_batches()
    if watch:
        threads.append(Thread(target=watcher, daemon=True))
        threads.append(Thread(target=wait_for_enter, daemon=True))
    for t in threads:
        t.start()
    collect_queue.join()
    output_queue.join()
    parse_queue.join()
    stop_event.set()
    for t in threads:
        t.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OpenAI batch pipeline for BookInfo extraction."
    )
    parser.add_argument(
        "input_dirs", nargs="+", type=Path, help="Directories containing PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openai_batches"),
        help="Where to write JSONL files",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.json"),
        help="Path to secrets with OPENAI_API_KEY",
    )
    parser.add_argument(
        "--max-pdfs", type=int, default=None, help="Cap number of PDFs to process"
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="OCR worker count",
    )
    parser.add_argument(
        "--request-workers", type=int, default=2, help="Request writer worker count"
    )
    parser.add_argument(
        "--submit-workers", type=int, default=1, help="Batch submit worker count"
    )
    parser.add_argument(
        "--batch-size", type=int, default=25, help="Number of OCR items per JSONL batch"
    )
    parser.add_argument(
        "--queue-size", type=int, default=64, help="Max items per internal queue"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model id for chat completions",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="OpenAI batch completion window",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--collect-results",
        action="store_true",
        help="Collect completed batches and persist results to the DB.",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite DB to persist results.",
    )
    parser.add_argument("--context-first-images", type=int, default=None)
    parser.add_argument("--context-last-images", type=int, default=None)
    parser.add_argument("--context-first-ocr-pages", type=int, default=None)
    parser.add_argument("--context-last-ocr-pages", type=int, default=None)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch the output dir for new batch responses and collect them until Enter is pressed.",
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=300,
        help="Seconds between rescans of the output directory when --watch is enabled.",
    )
    return parser.parse_args()


def main() -> list[SubmittedBatch]:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    input_dirs = _expand_dirs(args.input_dirs)
    if not input_dirs:
        raise FileNotFoundError("No input directories found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = BatchConfig(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        max_pdfs=args.max_pdfs,
        ocr_workers=max(1, args.ocr_workers),
        request_workers=max(1, args.request_workers),
        submit_workers=max(1, args.submit_workers),
        model=args.model,
        batch_size=max(1, args.batch_size),
        queue_size=max(1, args.queue_size),
        secrets=secrets,
        completion_window=args.completion_window,
        context_limits=ContextLimits(
            num_first_images=args.context_first_images,
            num_last_images=args.context_last_images,
            num_first_ocr_pages=args.context_first_ocr_pages,
            num_last_ocr_pages=args.context_last_ocr_pages,
        ),
    )
    submissions = run_pipeline(cfg)
    if args.collect_results:
        client = OpenAI(
            api_key=secrets.get("OPENAI_API_KEY"),
            project=secrets.get("OPENAI_PROJECT_ID"),
        )
        run_collector(
            output_dir=args.output_dir,
            client=client,
            db_path=args.results_db,
            model=args.model,
            watch=args.watch,
            scan_interval=args.scan_interval,
        )
    return submissions


if __name__ == "__main__":
    main()
