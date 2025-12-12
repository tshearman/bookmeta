import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import PipelineResult


def compute_pdf_hash(pdf_path: Path) -> str:
    pdf_path = Path(pdf_path)
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_store(store_path: Path) -> dict:
    store_path = Path(store_path)
    if store_path.exists():
        with open(store_path, "r") as f:
            return json.load(f)
    return {}


def save_store(data: dict, store_path: Path) -> None:
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(data, f, indent=2)


def persist_pipeline_result(
    result: "PipelineResult",
    store_path: Path,
) -> Optional[str]:
    """
    Save the extracted metadata for a PDF into the JSON store.
    """
    if result.book_info is None:
        return None

    pdf_hash = result.pdf_hash
    store = load_store(store_path)
    entry = store.setdefault(
        pdf_hash, {"pdf_path": str(result.pdf_path), "records": {}}
    )
    entry["pdf_path"] = str(result.pdf_path)
    records = entry.setdefault("records", {})
    record_key = f"{result.provider}:{result.model}"
    records[record_key] = result.serialize()
    save_store(store, store_path)
    return f"{pdf_hash}:{record_key}"
