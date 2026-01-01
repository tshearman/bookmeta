import hashlib
from pathlib import Path
import re
from typing import Any
from openai.types.responses import ResponseInputParam


def hash_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sanitize(obj) -> Any:
    def clean_string(value: str) -> str:
        """Remove surrogate code points while preserving readable characters."""
        replaced = value.encode("utf-8", "replace").decode("utf-8")
        return replaced.encode("utf-8", "ignore").decode("utf-8")

    if isinstance(obj, str):
        return clean_string(obj)
    if isinstance(obj, list):
        return [sanitize(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize(item) for item in obj)
    if isinstance(obj, dict):
        return {key: sanitize(val) for key, val in obj.items()}
    return obj


def split_authors(author_text: str) -> list[str]:
    """Split a free-form author string on commas and the word 'and'."""
    parts = re.split(r",\s*|\band\b", author_text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]
