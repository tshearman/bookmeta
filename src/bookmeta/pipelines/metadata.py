import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from bookmeta.types.bookinfo import DetailedBookInfo, DetailedBookInfoResult


def _list_or_none(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    cleaned = [value for value in values if value]
    return cleaned or None


def writer_payload(info: DetailedBookInfo) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": info.title,
        "subtitle": info.subtitle,
        "description": info.description,
        "publisher": info.publisher,
        "authors": _list_or_none(
            [info.author] if isinstance(info.author, str) else info.author
        ),
        # Map keywords to categories/tags for broader compatibility.
        "categories": _list_or_none(info.keywords),
        "tags": _list_or_none(info.keywords),
        "isbn_identifiers": _list_or_none(info.isbn_identifiers),
        "nsfw": info.nsfw,
    }
    return {key: value for key, value in payload.items() if value not in (None, [], "")}


def write_metadata_with_cli(
    source_pdf: Path,
    destination_pdf: Path,
    info: DetailedBookInfo,
    writer_bin: Path,
) -> None:
    # Resolve writer binary: allow pointing to the install root or the executable.
    if writer_bin.is_dir():
        candidate = writer_bin / "bin" / "pdf-metadata-writer-cli"
        if candidate.exists():
            writer_bin = candidate
    if not writer_bin.exists():
        raise FileNotFoundError(f"Writer binary not found at {writer_bin}")

    payload = writer_payload(info)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        metadata_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                str(writer_bin),
                str(source_pdf),
                str(destination_pdf),
                str(metadata_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"pdf-metadata-writer CLI failed: {exc.stderr or exc.stdout}"
        ) from exc
    finally:
        metadata_path.unlink(missing_ok=True)


def write_metadata(cfg):
    def inner(result: DetailedBookInfoResult) -> DetailedBookInfoResult | None:
        from bookmeta.monitoring import TimedItem

        payload = result.obj if isinstance(result, TimedItem) else result
        if payload is None or payload.detailed.nsfw:
            return None
        if not cfg.writer_bin or not cfg.output_dir:
            return payload
        pdf_path = payload.pdf.path
        rel_pdf = pdf_path.name
        for root in getattr(cfg, "roots", []):
            try:
                rel_pdf = pdf_path.resolve().relative_to(Path(root).resolve())
                break
            except Exception:
                continue
        destination = cfg.output_dir / rel_pdf
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_metadata_with_cli(
            payload.pdf.path, destination, payload.detailed, cfg.writer_bin
        )
        return payload

    return inner
