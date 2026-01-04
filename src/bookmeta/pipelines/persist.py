import hashlib
import json
import logging
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

from bookmeta.persistence import persist_batch_run
from bookmeta.pipelines import start_stage_workers
from bookmeta.types.extraction import LLMTaskBatch, PersistedBatch

LOGGER = logging.getLogger(__name__)


def _batch_digest(pdf_hashes: list[str], pipeline_hash: str) -> str:
    payload = {"pdf_hashes": sorted(pdf_hashes), "pipeline_hash": pipeline_hash}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _serialize_batch(batch: LLMTaskBatch) -> list[dict]:
    """Convert batch items to JSON-serializable dictionaries."""
    serialized: list[dict] = []
    for task in sorted(batch, key=lambda t: (t.pdf.hash, str(t.pdf.path))):
        serialized.append(
            {
                "pdf_path": str(task.pdf.path),
                "pdf_hash": task.pdf.hash,
                "provider": task.config.provider,
                "model": task.config.model,
                "payload": task.payload,
            }
        )
    return serialized


def persist_llm_task_batch(
    batch: LLMTaskBatch,
    output_dir: Path,
    pipeline_hash: str,
) -> PersistedBatch:
    pdfs = [t.pdf for t in batch]
    pdf_hashes = sorted([pdf.hash for pdf in pdfs])
    digest = _batch_digest(pdf_hashes, pipeline_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"batch_{digest}.json"

    submission: dict[str, Any] | None = None

    if not path.exists():
        payload = {
            "pipeline_hash": pipeline_hash,
            "pdf_hashes": pdf_hashes,
            "count": len(batch),
            "tasks": _serialize_batch(batch),
            "submission": None,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            submission = existing.get("submission")
        except Exception:
            LOGGER.warning("Unable to load existing submission info from %s", path)

    return PersistedBatch(
        path=path,
        digest=digest,
        pdfs=pdfs,
        pipeline_hash=pipeline_hash,
        count=len(batch),
        submission=submission,
    )


def start_persist_batch_pipeline(
    batch_queue: Queue[Any],
    persisted_queue: Queue[Any],
    output_dir: Path,
    pipeline_hash: str,
    *,
    db_path: Path | None,
    pipeline_config: str | None,
    upstream_done: Event,
    timeout: float | None = None,
) -> Thread:
    """
    Consume LLMTaskBatches, persist them to disk with deterministic filenames,
    and enqueue PersistedBatch metadata.
    """

    effective_timeout = 0.5 if timeout is None else timeout

    def _process(batch: LLMTaskBatch) -> PersistedBatch | None:
        started_at = time.perf_counter()
        try:
            persisted = persist_llm_task_batch(batch, output_dir, pipeline_hash)
            LOGGER.debug("Persist worker enqueued persisted batch %s", persisted.path)
            duration = time.perf_counter() - started_at
            if db_path:
                persist_batch_run(
                    db_path,
                    pdf_hashes=sorted([p.hash for p in persisted.pdfs]),
                    batch_id=persisted.digest,
                    batch_file=persisted.path,
                    process_time=duration,
                    pipeline_config=pipeline_config or "{}",
                    config_hash=pipeline_hash,
                )
            return persisted
        except Exception:
            LOGGER.exception("Persist worker failed to persist batch")
            return None

    threads = start_stage_workers(
        batch_queue,
        persisted_queue,
        _process,
        upstream_done,
        workers=1,
        timeout=effective_timeout,
    )

    return threads[0]


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _persist_submission_metadata(path: Path, submission: dict[str, Any]) -> None:
    payload = _load_payload(path)
    payload["submission"] = submission
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def submit_persisted_batch_to_openai(
    persisted: PersistedBatch,
    client,
    *,
    completion_window: str = "24h",
    endpoint: str = "/v1/chat/completions",
    expires_seconds: int = 86400,
) -> dict[str, Any]:
    """
    Upload a persisted batch file and create an OpenAI batch request.

    Skips submission if the batch is already marked as submitted. Returns submission metadata.
    """
    if persisted.submission and persisted.submission.get("submitted"):
        return persisted.submission

    with open(persisted.path, "rb") as f:
        upload = client.files.create(
            file=f,
            purpose="batch",
            expires_after={"anchor": "created_at", "seconds": expires_seconds},
        )

    try:
        upload_payload: dict[str, Any] = upload.model_dump(exclude_none=False)
    except Exception:
        try:
            upload_payload = json.loads(upload.model_dump_json(exclude_none=False))
        except Exception:
            upload_payload = {
                "id": getattr(upload, "id", None),
                "object": getattr(upload, "object", None),
                "bytes": getattr(upload, "bytes", None),
                "created_at": getattr(upload, "created_at", None),
                "expires_at": getattr(upload, "expires_at", None),
                "filename": getattr(upload, "filename", None),
                "purpose": getattr(upload, "purpose", None),
            }

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )

    # Best-effort conversion of the batch object into a JSON-serializable dict.
    try:
        batch_payload: dict[str, Any] = batch.model_dump(exclude_none=False)
    except Exception:
        try:
            batch_payload = json.loads(batch.model_dump_json(exclude_none=False))
        except Exception:
            # Manual extraction of common fields as a fallback.
            batch_payload = {
                "id": getattr(batch, "id", None),
                "object": getattr(batch, "object", None),
                "endpoint": getattr(batch, "endpoint", None),
                "errors": getattr(batch, "errors", None),
                "input_file_id": getattr(batch, "input_file_id", None),
                "completion_window": getattr(batch, "completion_window", None),
                "status": getattr(batch, "status", None),
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "created_at": getattr(batch, "created_at", None),
                "in_progress_at": getattr(batch, "in_progress_at", None),
                "expires_at": getattr(batch, "expires_at", None),
                "finalizing_at": getattr(batch, "finalizing_at", None),
                "completed_at": getattr(batch, "completed_at", None),
                "failed_at": getattr(batch, "failed_at", None),
                "expired_at": getattr(batch, "expired_at", None),
                "cancelling_at": getattr(batch, "cancelling_at", None),
                "cancelled_at": getattr(batch, "cancelled_at", None),
                "request_counts": getattr(batch, "request_counts", None),
                "metadata": getattr(batch, "metadata", None),
            }

    submission = {
        "submitted": True,
        "submitted_at": time.time(),
        "file": upload_payload,
        "batch": batch_payload,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }

    _persist_submission_metadata(persisted.path, submission)
    return submission
