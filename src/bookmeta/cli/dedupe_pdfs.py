import argparse
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Iterable

from bookmeta.cli.utils import discover_pdfs
from bookmeta.data.sqlite import _compute_pdf_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect duplicate PDFs in a directory by hash and either remove "
            "them or move them into a quarantine directory."
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
        help="Number of hashing workers to use in parallel.",
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


def _hash_pdf(path: Path) -> tuple[Path, str]:
    return path, _compute_pdf_hash(path)


def _dedupe_pdfs(
    paths: Iterable[Path],
    *,
    log_interval: int = 50,
    workers: int,
) -> tuple[list[Path], list[Path]]:
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    duplicates: list[Path] = []
    hashed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for pdf, pdf_hash in executor.map(_hash_pdf, paths):
            hashed += 1
            if pdf_hash in seen:
                logging.info("Duplicate detected: %s (matches %s)", pdf, seen[pdf_hash])
                duplicates.append(pdf)
            else:
                seen[pdf_hash] = pdf
                unique.append(pdf)
            if hashed % log_interval == 0:
                logging.info("Hashed %d PDFs...", hashed)
    logging.info("Finished hashing %d PDFs.", hashed)
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
            target = base_target.parent / f"{base_target.stem}_{counter}{base_target.suffix}"
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

    pdf_iter = discover_pdfs(args.pdf_directory)
    logging.info("Hashing PDFs with %d workers.", args.workers)
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
