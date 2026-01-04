from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from bookmeta.config.pipeline import PipelineConfig as BatchPipelineConfig
    from bookmeta.pipelines.active import PipelineConfig
    from bookmeta.types.bookinfo import DetailedBookInfoResult
    from bookmeta.types.extraction import ContextLimits

LOGGER = logging.getLogger(__name__)


def _sanitize_client_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in config.items():
        if value is None:
            continue
        if any(token in key.lower() for token in ("key", "secret", "token")):
            sanitized[key] = "<redacted>"
        else:
            sanitized[key] = value
    return sanitized


def _serialize_context_limits(limits: ContextLimits) -> dict[str, Any]:
    return {
        "num_first_images": limits.num_first_images,
        "num_last_images": limits.num_last_images,
        "num_first_ocr_pages": limits.num_first_ocr_pages,
        "num_last_ocr_pages": limits.num_last_ocr_pages,
    }


def _serialize_pipeline_config(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "mode": "active",
        "prompt": cfg.prompt,
        "provider": cfg.provider_config.provider,
        "model": cfg.provider_config.model,
        "client_config": _sanitize_client_config(cfg.provider_config.client_config),
        "context_limits": _serialize_context_limits(cfg.context_limits),
        "ocr_config": {
            "num_first_pages": cfg.ocr_config.num_first_pages,
            "num_last_pages": cfg.ocr_config.num_last_pages,
            "methods": [method.name for method in cfg.ocr_config.methods],
        },
        "roots": [str(root) for root in cfg.roots],
        "dedupe": cfg.dedupe,
        "limit": cfg.limit,
        "queue_size": cfg.queue_size,
        "workers": {
            "ocr": cfg.ocr_workers,
            "extraction": cfg.extraction_workers,
            "llm": cfg.llm_workers,
            "persist": cfg.persist_workers,
        },
    }


def serialize_pipeline_config(cfg: PipelineConfig) -> str:
    """
    Return a stable JSON representation of the pipeline config used for persistence.
    """
    return json.dumps(
        _serialize_pipeline_config(cfg), ensure_ascii=False, indent=2, sort_keys=False
    )


def _ensure_run_timestamp_column(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(pipeline_runs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "run_timestamp" not in columns:
        conn.execute("ALTER TABLE pipeline_runs ADD COLUMN run_timestamp TEXT")


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pdf_hash ON pipeline_runs(pdf_hash)"
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_batch_runs_pdf_config
        ON batch_runs(pdf_hash, config_hash)
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_batch_runs_batch_id ON batch_runs(batch_id)"
    )


def _ensure_batch_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_hash TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            batch_file TEXT NOT NULL,
            process_time REAL NOT NULL,
            pipeline_config TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT NOT NULL,
                pdf_hash TEXT NOT NULL,
                pipeline_config TEXT NOT NULL,
                result TEXT NOT NULL,
                run_timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_run_timestamp_column(conn)
        _ensure_batch_runs_table(conn)
        _ensure_indexes(conn)
        conn.commit()


def execute_persist_bookinfo(config: PipelineConfig):
    def inner(bookinfo):
        return persist_bookinfo_result(bookinfo, config)

    return inner


def has_existing_result(
    db_path: Path, pdf_hash: str, *, provider: str, model: str
) -> bool:
    """
    Return True if a run already exists for the given PDF hash and provider/model.
    Uses a read-only SQLite connection to avoid locking the database during discovery.
    """
    if not db_path.exists():
        return False

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.execute(
                """
                SELECT 1
                FROM pipeline_runs
                WHERE pdf_hash = ?
                  AND json_extract(pipeline_config, '$.provider') = ?
                  AND json_extract(pipeline_config, '$.model') = ?
                LIMIT 1
                """,
                (pdf_hash, provider, model),
            )
            return cursor.fetchone() is not None
    except sqlite3.OperationalError as exc:
        # Database exists but table is missing or JSON1 is unavailable.
        message = str(exc).lower()
        if "no such table" in message:
            return False
        LOGGER.exception(
            "SQLite operational error checking existing result for hash %s: %s",
            pdf_hash,
            exc,
        )
        return False
    except Exception:
        LOGGER.exception(
            "Failed to check for existing result for hash %s with provider=%s model=%s",
            pdf_hash,
            provider,
            model,
        )
        return False


def persist_bookinfo_result(
    result: "DetailedBookInfoResult", cfg: PipelineConfig
) -> "DetailedBookInfoResult | None":
    ensure_db(cfg.results_db)

    payload = (
        result.pdf.path.name,
        result.pdf.hash,
        serialize_pipeline_config(cfg),
        json.dumps(result.detailed.model_dump(), ensure_ascii=False, indent=2),
        datetime.now(timezone.utc).isoformat(),
    )

    try:
        with sqlite3.connect(cfg.results_db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (
                    pdf_name, pdf_hash, pipeline_config, result, run_timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
        LOGGER.debug("Persisted BookInfo for %s", result.pdf.path)
        return result
    except Exception:
        LOGGER.exception("Failed to persist BookInfo for %s", result.pdf.path)
        return None


def serialize_batch_pipeline_config(cfg: "BatchPipelineConfig") -> str:
    """
    Return a stable JSON representation of the batch pipeline config used for persistence.
    """
    payload = {
        "mode": "batch",
        "prompt": cfg.pdf.prompt,
        "provider": cfg.pdf.provider_config.provider,
        "model": cfg.pdf.provider_config.model,
        "client_config": _sanitize_client_config(cfg.pdf.provider_config.client_config),
        "context_limits": _serialize_context_limits(cfg.pdf.context_limits),
        "ocr_config": {
            "num_first_pages": cfg.pdf.ocr_config.num_first_pages,
            "num_last_pages": cfg.pdf.ocr_config.num_last_pages,
            "methods": [method.name for method in cfg.pdf.ocr_config.methods],
        },
        "roots": [str(root) for root in cfg.runtime.roots],
        "dedupe": cfg.runtime.dedupe,
        "limit": cfg.runtime.limit,
        "queue_size": cfg.runtime.queue_size,
        "workers": {
            "ocr": cfg.runtime.ocr_workers,
            "extraction": cfg.runtime.extraction_workers,
            "llm": cfg.runtime.llm_workers,
        },
        "stage_timeout": cfg.runtime.stage_timeout,
        "batch_output_dir": str(cfg.runtime.batch_output_dir),
        "monitor_queues": cfg.runtime.monitor_queues,
        "results_db": str(cfg.runtime.results_db),
        "resume": cfg.runtime.resume,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def persist_batch_run(
    db_path: Path,
    *,
    pdf_hashes: Iterable[str],
    batch_id: str,
    batch_file: Path,
    process_time: float,
    pipeline_config: str,
    config_hash: str,
) -> None:
    """
    Record the PDFs contained in a persisted batch so they can be skipped on resume.
    """
    ensure_db(db_path)
    rows = [
        (
            pdf_hash,
            batch_id,
            str(batch_file),
            process_time,
            pipeline_config,
            config_hash,
        )
        for pdf_hash in pdf_hashes
    ]
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO batch_runs (
                    pdf_hash, batch_id, batch_file, process_time, pipeline_config, config_hash
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
    except Exception:
        LOGGER.exception("Failed to persist batch metadata for batch %s", batch_id)


def has_persisted_batch_result(
    db_path: Path, pdf_hash: str, *, config_hash: str
) -> bool:
    """
    Return True if the PDF has already been placed in a persisted batch for the config.
    Uses a read-only connection and tolerates missing tables for backward compatibility.
    """
    if not db_path.exists():
        return False

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.execute(
                """
                SELECT 1
                FROM batch_runs
                WHERE pdf_hash = ?
                  AND config_hash = ?
                LIMIT 1
                """,
                (pdf_hash, config_hash),
            )
            return cursor.fetchone() is not None
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "no such table" in message:
            return False
        LOGGER.exception(
            "SQLite operational error checking persisted batch for hash %s: %s",
            pdf_hash,
            exc,
        )
        return False
    except Exception:
        LOGGER.exception(
            "Failed to check persisted batch for hash %s with config hash %s",
            pdf_hash,
            config_hash,
        )
        return False
