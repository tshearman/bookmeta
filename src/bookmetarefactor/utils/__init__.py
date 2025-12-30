import hashlib
from pathlib import Path


def hash_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sanitize_payload(obj):
    def clean_string(value: str) -> str:
        """Remove surrogate code points while preserving readable characters."""
        replaced = value.encode("utf-8", "replace").decode("utf-8")
        return replaced.encode("utf-8", "ignore").decode("utf-8")

    if isinstance(obj, str):
        return clean_string(obj)
    if isinstance(obj, list):
        return [sanitize_payload(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_payload(item) for item in obj)
    if isinstance(obj, dict):
        return {key: sanitize_payload(val) for key, val in obj.items()}
    return obj
