#!/usr/bin/env python
"""
Deduplicate PDFs by comparing first page text, last page text, and page count.

Usage:
    python dedupe_pdf_by_content.py /path/to/pdfs --dest /tmp/duplicates
    python dedupe_pdf_by_content.py /path/to/pdfs --rm
"""

import argparse
import hashlib
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PyPDF2 import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect duplicate PDFs by page count plus first/last page text and "
            "either remove them or move them into a destination directory."
        )
    )
    parser.add_argument(
        "pdf_directory",
        type=Path,
        help="Root directory to scan recursively for PDFs.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dest",
        type=Path,
        help="Directory where duplicate PDFs should be moved.",
    )
    group.add_argument(
        "--rm",
        action="store_true",
        help="Delete duplicate PDFs instead of moving them.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Number of workers to use in parallel.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARN, ERROR).",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be a positive integer.")
    return args


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip().lower()


def _page_image_digest(page) -> str:
    hasher = hashlib.sha256()
    try:
        images = getattr(page, "images", []) or []
        for image in images:
            data = getattr(image, "data", b"") or b""
            width = getattr(image, "width", None)
            height = getattr(image, "height", None)
            if width is not None and height is not None:
                hasher.update(f"{width}x{height}".encode())
            hasher.update(data)
    except Exception as exc:
        logging.debug("Failed to hash images for page: %s", exc)
    return hasher.hexdigest()


def _pdf_signature(path: Path) -> Tuple[int, str, str, str, str]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    first_page = reader.pages[0]
    last_page = reader.pages[-1]
    first_text = _normalize_text(first_page.extract_text() or "")
    last_text = _normalize_text(last_page.extract_text() or "")
    first_images = _page_image_digest(first_page)
    last_images = _page_image_digest(last_page)
    return page_count, first_text, first_images, last_text, last_images


def _hash_pdf_by_content(
    path: Path,
) -> Tuple[Path, Optional[Tuple[int, str, str, str, str]]]:
    try:
        return path, _pdf_signature(path)
    except Exception as exc:
        logging.warning("Failed to read %s: %s", path, exc)
        return path, None


def _discover_pdfs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"PDF directory is not a directory: {root}")
    yield from (path for path in root.rglob("*.pdf") if path.is_file())


def _dedupe_pdfs(
    paths: Iterable[Path],
    *,
    log_interval: int = 25,
    workers: int,
) -> Tuple[list[Path], list[Path]]:
    seen: dict[Tuple[int, str, str, str, str], Path] = {}
    unique: list[Path] = []
    duplicates: list[Path] = []
    processed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for pdf, signature in executor.map(_hash_pdf_by_content, paths):
            processed += 1
            if signature is None:
                unique.append(pdf)
                continue
            if signature in seen:
                logging.info("Duplicate detected: %s (matches %s)", pdf, seen[signature])
                duplicates.append(pdf)
            else:
                seen[signature] = pdf
                unique.append(pdf)
            if processed % log_interval == 0:
                logging.info("Processed %d PDFs...", processed)
    logging.info("Finished processing %d PDFs.", processed)
    return unique, duplicates


def _ensure_destination(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return dest.resolve()


def _move_duplicates(duplicates: list[Path], dest: Path, root: Path) -> None:
    dest = _ensure_destination(dest)
    for idx, pdf in enumerate(duplicates, start=1):
        try:
            relative_pdf = pdf.relative_to(root)
        except ValueError:
            logging.warning(
                "PDF %s is outside root %s; placing at destination root.", pdf, root
            )
            relative_pdf = Path(pdf.name)

        target = dest / relative_pdf
        target.parent.mkdir(parents=True, exist_ok=True)
        base_target = target
        counter = 1
        while target.exists():
            target = (
                base_target.parent / f"{base_target.stem}_{counter}{base_target.suffix}"
            )
            counter += 1
        shutil.move(str(pdf), target)
        logging.info("Moved duplicate %s -> %s", pdf, target)
        logging.info("Moved %d/%d duplicates.", idx, len(duplicates))


def _remove_duplicates(duplicates: list[Path]) -> None:
    total = len(duplicates)
    for idx, pdf in enumerate(duplicates, start=1):
        pdf.unlink(missing_ok=True)
        logging.info("Removed duplicate %s", pdf)
        logging.info("Removed %d/%d duplicates.", idx, total)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    pdf_iter = _discover_pdfs(args.pdf_directory)
    logging.info("Processing PDFs with %d workers.", args.workers)
    unique, duplicates = _dedupe_pdfs(pdf_iter, workers=args.workers)
    logging.info(
        "Found %d unique PDFs and %d duplicates.", len(unique), len(duplicates)
    )
    if not duplicates:
        logging.info("Nothing to do.")
        return 0

    if args.rm:
        _remove_duplicates(duplicates)
    else:
        _move_duplicates(duplicates, args.dest, args.pdf_directory)

    logging.info("Finished deduplicating PDFs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
