#!/usr/bin/env python3
import argparse
import hashlib
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm


def _compute_pdf_hash(pdf_path: Path, /) -> str:
    with pdf_path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


LOGGER = logging.getLogger("copy_pdfs")

INCLUDE_PREFIX = "The Ultimate Trove - Books "
LANG_TAG_RE = re.compile(r"\[[A-Za-z]{2}\]")


def iter_pdfs(root: Path) -> Iterator[Path]:
    """Yield all PDF files under root (case-insensitive)."""
    for path in root.rglob("*.pdf"):
        if path.is_file():
            yield path


def allowed(path: Path) -> bool:
    """Apply include/exclude rules to a PDF path."""
    parts = path.parts
    # Include only if a directory component starts with the target prefix.
    if not any(part.startswith(INCLUDE_PREFIX) for part in parts):
        return False
    # Exclude any path segment named LANG.
    if any(part == "LANG" for part in parts):
        return False
    # Exclude if any component contains a [xx] language code.
    if LANG_TAG_RE.search(str(path)):
        return False
    if "2025-04 (Apr) Update" in str(path):
        return False
    return True


def _load_existing_hashes(dest: Path) -> dict[str, Path]:
    """Return a map of pdf_hash -> destination path for PDFs already in dest."""
    hashes: dict[str, Path] = {}
    if not dest.exists():
        return hashes

    for pdf in iter_pdfs(dest):
        try:
            pdf_hash = _compute_pdf_hash(pdf)
        except OSError as exc:
            LOGGER.warning("Failed to hash %s: %s", pdf, exc)
            continue
        hashes.setdefault(pdf_hash, pdf)

    if hashes:
        LOGGER.info("Loaded %d existing PDF hashes from %s", len(hashes), dest)
    return hashes


def copy_pdfs(
    src_roots: list[Path],
    dest: Path,
    dry_run: bool = False,
    deduplicate: bool = False,
    overwrite: bool = False,
    max_pdfs: int | None = None,
) -> int:
    if max_pdfs is not None and max_pdfs <= 0:
        raise ValueError("max_pdfs=%s is not positive; ignoring limit.", max_pdfs)

    allowed_paths: list[tuple[Path, Path]] = []
    for root in src_roots:
        base = root.resolve()
        for pdf in iter_pdfs(base):
            if allowed(pdf):
                allowed_paths.append((base, pdf))

    seen_hashes = _load_existing_hashes(dest) if deduplicate else {}
    LOGGER.info(
        "Found %d eligible PDFs across %d source roots",
        len(allowed_paths),
        len(src_roots),
    )

    copied = 0
    with tqdm(total=len(allowed_paths), desc="Copying PDFs", unit="pdf") as pbar:
        for base, pdf in allowed_paths:
            dst = dest / pdf.relative_to(base)
            if deduplicate:
                try:
                    pdf_hash = _compute_pdf_hash(pdf)
                except OSError as exc:
                    LOGGER.warning("Failed to hash %s: %s", pdf, exc)
                    pbar.update(1)
                    continue

                existing_path = seen_hashes.get(pdf_hash)
                if existing_path:
                    LOGGER.info(
                        "Skip duplicate (hash=%s): %s matches %s",
                        pdf_hash,
                        pdf,
                        existing_path,
                    )
                elif dry_run:
                    LOGGER.info("DRY-RUN: would copy %s -> %s", pdf, dst)
                    seen_hashes[pdf_hash] = dst
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pdf, dst)
                    copied += 1
                    seen_hashes[pdf_hash] = dst
                if max_pdfs is not None and copied >= max_pdfs:
                    LOGGER.info("Reached max_pdfs=%d; stopping early.", max_pdfs)
                    pbar.update(1)
                    break
            else:
                if dst.exists() and not overwrite:
                    LOGGER.info("Skip (exists): %s", dst)
                elif dry_run:
                    LOGGER.info(
                        "DRY-RUN: would copy %s -> %s%s",
                        pdf,
                        dst,
                        " (overwrite)" if dst.exists() else "",
                    )
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pdf, dst)
                    copied += 1
                if max_pdfs is not None and copied >= max_pdfs:
                    LOGGER.info("Reached max_pdfs=%d; stopping early.", max_pdfs)
                    pbar.update(1)
                    break
            pbar.update(1)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy filtered PDFs from directories, preserving structure."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="One or more source directories to scan recursively.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        type=Path,
        help="Destination directory (created if missing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions without copying files.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Skip PDFs whose content hash already exists in the destination.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When not deduplicating, allow copying over existing destination PDFs.",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Stop after copying at most this many PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dest = args.dest
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    sources = [src for src in args.sources if src.exists() and src.is_dir()]
    if not sources:
        print("No valid source directories provided.", file=sys.stderr)
        return 1

    total = copy_pdfs(
        sources,
        dest,
        dry_run=args.dry_run,
        deduplicate=args.deduplicate,
        overwrite=args.overwrite,
        max_pdfs=args.max_pdfs,
    )
    if args.dry_run:
        print(f"DRY-RUN: would copy {total} PDFs to {dest}")
    else:
        print(f"Copied {total} PDFs to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
