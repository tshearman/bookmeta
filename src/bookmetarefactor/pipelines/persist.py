import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from bookmetarefactor.types.extraction import LLMTaskBatch, PersistedBatch


def _batch_digest(pdf_hashes: list[str], pipeline_hash: str) -> str:
    payload = {"pdf_hashes": sorted(pdf_hashes), "pipeline_hash": pipeline_hash}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _serialize_batch(batch: LLMTaskBatch) -> list[dict]:
    """Convert batch items to JSON-serializable dictionaries."""
    serialized: list[dict] = []
    for task in sorted(
        batch, key=lambda t: (t.pdf.hash, str(t.pdf.path))
    ):
        serialized.append(
            {
                "pdf_path": str(task.pdf.path),
                "pdf_hash": task.pdf.hash,
                "provider": task.config.provider,
                "model": task.config.model,
                "client_config": task.config.client_config,
                "payload": task.payload,
            }
        )
    return serialized


def persist_llm_task_batch(
    batch: LLMTaskBatch,
    output_dir: Path,
    pipeline_hash: str,
) -> PersistedBatch:
    pdf_hashes = [task.pdf.hash for task in batch]
    digest = _batch_digest(pdf_hashes, pipeline_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"batch_{digest}.json"

    if not path.exists():
        payload = {
            "pipeline_hash": pipeline_hash,
            "pdf_hashes": sorted(pdf_hashes),
            "count": len(batch),
            "tasks": _serialize_batch(batch),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return PersistedBatch(
        path=path,
        digest=digest,
        pdf_hashes=tuple(sorted(pdf_hashes)),
        pipeline_hash=pipeline_hash,
        count=len(batch),
    )


def start_persist_batch_pipeline(
    batch_queue: Queue[LLMTaskBatch],
    persisted_queue: Queue[PersistedBatch],
    output_dir: Path,
    pipeline_hash: str,
    *,
    upstream_done: Event,
    timeout: float | None = None,
) -> Thread:
    """
    Consume LLMTaskBatches, persist them to disk with deterministic filenames,
    and enqueue PersistedBatch metadata.
    """

    def _worker() -> None:
        while True:
            try:
                batch = batch_queue.get(timeout=timeout)
            except Empty:
                if upstream_done.is_set() and batch_queue.empty():
                    break
                continue
            try:
                persisted = persist_llm_task_batch(batch, output_dir, pipeline_hash)
                persisted_queue.put(persisted)
            finally:
                batch_queue.task_done()

    thread = Thread(
        target=_worker,
        daemon=True,
    )
    thread.start()
    return thread
