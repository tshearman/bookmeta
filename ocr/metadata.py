from __future__ import annotations

from pathlib import Path
from typing import Dict

import fitz


Metadata = Dict[str, str | None]


def load_pdf_metadata(pdf_path: str | Path) -> Metadata:
    """Return sanitized metadata for the given PDF."""
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        meta = doc.metadata or {}
    return {
        "title": meta.get("title"),
        "author": meta.get("author"),
        "subject": meta.get("subject"),
        "keywords": meta.get("keywords"),
    }
