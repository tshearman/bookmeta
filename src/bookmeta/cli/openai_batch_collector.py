import argparse
import json
import logging
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from openai import OpenAI

from bookmeta.config.settings import DEFAULT_DB_PATH
from bookmeta.data.sqlite import _compute_pdf_hash, _ensure_db
from bookmeta.services.bookinfo.book_info_response import BookInfoResponse

LOGGER = logging.getLogger(__name__)


def _read_secrets(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    return json.loads(path.read_text())


def _persist_batch_result(
    db_path: Path, pdf_path: Path, pdf_hash: str, result: dict, model: str
) -> None:
    _ensure_db(db_path)
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    pipeline_config = {"mode": "openai_batch", "model": model}
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
                    {
                        "type": "completed",
                        "batch_id": batch_id,
                        "mapping_path": resp_path.with_suffix(".pdfs.jsonl"),
                        "output_path": output_path,
                        "batch_obj": obj,
                    }
                )
                continue
            if batch_id:
                collect_queue.put({"type": "pending", "resp_path": resp_path})

    def poll_worker() -> None:
        while not stop_event.is_set():
            try:
                item = collect_queue.get(timeout=poll_interval)
            except Empty:
                if stop_event.is_set() and collect_queue.empty():
                    break
                continue
            try:
                if isinstance(item, dict) and item.get("type") == "completed":
                    output_queue.put(item)
                    continue
                resp_path: Path = item["resp_path"]
                obj = json.loads(resp_path.read_text())
                batch_id = obj.get("id")
                mapping_path = resp_path.with_suffix(".pdfs.jsonl")
                if not batch_id:
                    continue
                latest = client.batches.retrieve(batch_id)
                status = latest.status
                output_file_id = latest.output_file_id
                resp_path.write_text(json.dumps(latest.to_dict_recursive(), indent=2), encoding="utf-8")
                if status != "completed" or not output_file_id:
                    if not stop_event.is_set():
                        time.sleep(poll_interval)
                        collect_queue.put(item)
                    continue
                output_path = resp_path.with_suffix(".batch_output.jsonl")
                content = client.files.content(output_file_id)
                output_path.write_bytes(content.read())
                output_queue.put(
                    {
                        "type": "completed",
                        "batch_id": batch_id,
                        "mapping_path": mapping_path,
                        "output_path": output_path,
                        "batch_obj": latest.to_dict_recursive(),
                    }
                )
            finally:
                collect_queue.task_done()

    def parse_worker() -> None:
        while not stop_event.is_set():
            try:
                item = output_queue.get(timeout=poll_interval)
            except Empty:
                if stop_event.is_set() and output_queue.empty():
                    break
                continue
            try:
                mapping: dict[str, str] = {}
                with Path(item["mapping_path"]).open("r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        cid = rec.get("custom_id")
                        path = rec.get("pdf_path")
                        if cid and path:
                            mapping[cid] = path
                with Path(item["output_path"]).open("r", encoding="utf-8") as fh:
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
                            detailed = bir.info.as_detailed_book_info().model_dump()
                            pdf_path = Path(mapping.get(custom_id, ""))
                            pdf_hash = custom_id
                            parse_queue.put(
                                {
                                    "pdf_path": pdf_path,
                                    "pdf_hash": pdf_hash,
                                    "detailed": detailed,
                                }
                            )
                        except Exception as exc:
                            LOGGER.warning(
                                "Failed to parse batch output for %s: %s", custom_id, exc
                            )
                            continue
            finally:
                output_queue.task_done()

    def persist_worker() -> None:
        while not stop_event.is_set():
            try:
                item = parse_queue.get(timeout=poll_interval)
            except Empty:
                if stop_event.is_set() and parse_queue.empty():
                    break
                continue
            try:
                pdf_path = item["pdf_path"]
                pdf_hash = item["pdf_hash"]
                detailed = item["detailed"]
                if pdf_path:
                    _persist_batch_result(
                        db_path, pdf_path, pdf_hash, detailed, model
                    )
                    LOGGER.info("Persisted %s", pdf_path)
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
        description="Collect OpenAI batch outputs and persist to the DB."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("openai_batches"))
    parser.add_argument("--secrets", type=Path, default=Path("secrets.json"))
    parser.add_argument("--results-db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--scan-interval", type=int, default=300)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    client = OpenAI(
        api_key=secrets.get("OPENAI_API_KEY"),
        project=secrets.get("OPENAI_PROJECT_ID"),
    )
    run_collector(
        output_dir=args.output_dir,
        client=client,
        poll_interval=args.poll_interval,
        db_path=args.results_db,
        model=args.model,
        watch=args.watch,
        scan_interval=args.scan_interval,
    )


if __name__ == "__main__":
    main()
