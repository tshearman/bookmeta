from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bookinfo.book_info import DetailedBookInfo

DEFAULT_DB_PATH = Path("bookmeta.db")


def _callable_name(candidate: Any) -> str:
    if hasattr(candidate, "__qualname__"):
        return getattr(candidate, "__qualname__")
    if hasattr(candidate, "__name__"):
        return getattr(candidate, "__name__")
    return repr(candidate)


def _sanitize_client_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in config.items():
        if value is None:
            continue
        if any(token in key.lower() for token in ("key", "secret", "token")):
            sanitized[key] = "<redacted>"
        else:
            sanitized[key] = value
    return sanitized


def serialize_pipeline_config(config: Any) -> dict[str, Any]:
    return {
        "ocr_config": {
            "num_first_pages": config.ocr_config.num_first_pages,
            "num_last_pages": config.ocr_config.num_last_pages,
            "ocr_methods": [
                _callable_name(method) for method in config.ocr_config.ocr_methods
            ],
        },
        "extraction_config": {
            "provider": config.extraction_config.provider,
            "model": config.extraction_config.model,
            "client_config": _sanitize_client_config(
                config.extraction_config.client_config
            ),
        },
        "selection_config": {
            "provider": config.selection_config.provider,
            "model": config.selection_config.model,
            "client_config": _sanitize_client_config(
                config.selection_config.client_config
            ),
        },
        "booksearch_config": {
            "search_methods": [
                _callable_name(method)
                for method in config.booksearch_config.search_methods
            ]
        },
    }


def _compute_pdf_hash(pdf_path: Path, chunk_size: int = 64 * 1024) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_db(db_path: Path) -> None:
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
        conn.commit()


def _ensure_run_timestamp_column(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(pipeline_runs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "run_timestamp" not in columns:
        conn.execute("ALTER TABLE pipeline_runs ADD COLUMN run_timestamp TEXT")


def persist_run(
    db_path: Path, pdf_path: Path, config: Any, result: DetailedBookInfo
) -> None:
    _ensure_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = (
        pdf_path.name,
        _compute_pdf_hash(pdf_path),
        json.dumps(serialize_pipeline_config(config), indent=2),
        json.dumps(result.model_dump(), indent=2),
        timestamp,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (pdf_name, pdf_hash, pipeline_config, result, run_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
