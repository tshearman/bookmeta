import argparse
import logging
import os
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from bookmeta.cli.pipeline import (
    _expand_paths,
    _read_secrets,
    build_pipeline_config,
    process_pdf,
    run_bookinfo_only_pipeline,
)
from bookmeta.cli.utils import discover_pdfs
from bookmeta.config.settings import DEFAULT_DB_PATH
from bookmeta.data.sqlite import _compute_pdf_hash

warnings.filterwarnings(
    "ignore",
    message="Persisting input arguments took",
    category=UserWarning,
    module="joblib.memory",
)


def _count_pdfs_with_ripgrep(pdf_directory: Path) -> int:
    """Count PDFs using a small bash loop to avoid an in-Python directory walk."""
    if not pdf_directory.exists():
        raise RuntimeError(f"PDF directory does not exist: {pdf_directory}")
    if not pdf_directory.is_dir():
        raise RuntimeError(f"PDF directory is not a directory: {pdf_directory}")

    cmd = (
        "count=0; "
        "while IFS= read -r _; do count=$((count+1)); done < <(rg --files -g '*.pdf' \"$1\"); "
        "printf '%s' \"$count\""
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd, "_", str(pdf_directory)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ripgrep (rg) is required to count PDFs but was not found on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Failed to count PDFs in {pdf_directory}: {error or exc}"
        ) from exc

    output = result.stdout.strip()
    return int(output) if output else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process all PDFs within a directory and produce refined book metadata."
    )
    parser.add_argument(
        "pdf_directories",
        nargs="+",
        type=Path,
        help="Root directory/directories to scan recursively for PDFs (globs allowed).",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.json"),
        help="Path to secrets JSON with API keys.",
    )
    parser.add_argument(
        "--num-front-ocr-pages",
        type=int,
        default=5,
        help="Number of front pages to OCR.",
    )
    parser.add_argument(
        "--num-back-ocr-pages",
        type=int,
        default=3,
        help="Number of ending pages to OCR.",
    )
    parser.add_argument(
        "--ocr-ollama-model",
        type=str,
        default=None,
        help="Optional Ollama model for OCR post-processing.",
    )
    parser.add_argument(
        "--extraction-provider",
        choices=("openai", "ollama"),
        default="openai",
        help="Provider for the initial BookInfo extraction stage.",
    )
    parser.add_argument(
        "--extraction-model",
        default=None,
        help="Optional model override for the BookInfo extraction stage.",
    )
    parser.add_argument(
        "--selection-provider",
        choices=("openai", "ollama"),
        default="openai",
        help="Provider for the BookInfo selection stage.",
    )
    parser.add_argument(
        "--selection-model",
        default=None,
        help="Optional model override for the BookInfo selection stage.",
    )
    parser.add_argument(
        "--search-max-results",
        type=int,
        default=5,
        help="Maximum Books search results to fetch during book search per provider.",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite DB used to log pipeline runs.",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        help="Logging level (DEBUG, INFO, WARN, ERROR).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Number of pipeline executions to run in parallel.",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Optional cap on the number of PDFs to process after discovery.",
    )
    parser.add_argument(
        "--context-first-images",
        type=int,
        default=None,
        help="Number of first page images to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--context-last-images",
        type=int,
        default=None,
        help="Number of last page images to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--context-first-ocr-pages",
        type=int,
        default=None,
        help="Number of first OCR pages to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--context-last-ocr-pages",
        type=int,
        default=None,
        help="Number of last OCR pages to include in BookInfo/selection context (default: all).",
    )
    parser.add_argument(
        "--book-prompt",
        choices=("default", "ttrpg"),
        default="default",
        help="BookInfo prompt variant to use (default or ttrpg-focused).",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("full", "bookinfo-only"),
        default="full",
        help="Choose full pipeline (with search/selection) or bookinfo-only.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Size of each batch"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=16,
        help="Max items buffered per pipeline stage (bookinfo-only mode).",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Worker threads for OCR stage (bookinfo-only mode).",
    )
    parser.add_argument(
        "--bookinfo-workers",
        type=int,
        default=None,
        help="Worker threads for BookInfo stage (bookinfo-only mode). Defaults to ocr-workers.",
    )
    parser.add_argument(
        "--result-workers",
        type=int,
        default=None,
        help="Worker threads for result stage (bookinfo-only mode). Defaults to ocr-workers.",
    )
    args = parser.parse_args()

    for name in (
        "context_first_images",
        "context_last_images",
        "context_first_ocr_pages",
        "context_last_ocr_pages",
        "max_pdfs",
        "queue_size",
        "ocr_workers",
        "bookinfo_workers",
        "result_workers",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    return args


def chunked_iter(iterable, size):
    while True:
        chunk = list(islice(iterable, size))
        if not chunk:
            break
        yield chunk


def main() -> dict[str, Any]:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    config = build_pipeline_config(args, secrets)

    pdf_dirs = _expand_paths(args.pdf_directories)
    if not pdf_dirs:
        raise FileNotFoundError("No directories matched the provided arguments.")

    total_discovered = sum(_count_pdfs_with_ripgrep(pdf_dir) for pdf_dir in pdf_dirs)
    pdf_iter = (pdf for pdf_dir in pdf_dirs for pdf in discover_pdfs(pdf_dir))
    limited_pdf_iter = (
        islice(pdf_iter, args.max_pdfs) if args.max_pdfs is not None else pdf_iter
    )
    progress_total = (
        min(total_discovered, args.max_pdfs) if args.max_pdfs is not None else total_discovered
    )

    workers = max(1, args.workers)
    processed: dict[str, Any] = {}
    failed: dict[str, str] = {}

    if config.mode == "bookinfo_only":
        logging.info("Running staged bookinfo-only pipeline")
        with tqdm(total=progress_total, desc="Processing PDFs", unit="pdf") as progress:

            def _progress_callback(done: int, total: int) -> None:
                progress.n = done
                progress.refresh()

            results = run_bookinfo_only_pipeline(
                limited_pdf_iter,
                config,
                args.results_db,
                progress_callback=_progress_callback,
                total_expected=progress_total,
                enqueue_limit=args.max_pdfs,
                dedupe=True,
            )
        for result in results:
            pdf_hash = _compute_pdf_hash(result.pdf_path)
            if result.failure:
                failed[pdf_hash] = result.failure.error
                continue
            if result.detailed:
                processed[pdf_hash] = {
                    "input": str(result.pdf_path),
                    "output": result.detailed.model_dump(),
                }
        summary = {"processed": processed, "failed": failed}
        return summary

    chunk_size = args.chunk_size
    logging.info(f"Processing with {workers} workers in batches of size {chunk_size}.")
    chunks = chunked_iter(pdf_list, chunk_size)
    with tqdm(total=progress_total, desc="Processing PDFs", unit="pdf") as progress:
        for n, chunk in enumerate(chunks):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_pdf, pdf, config, args.results_db): pdf
                    for pdf in chunk
                }
                for future in as_completed(futures):
                    pdf = futures[future]
                    pdf_hash = _compute_pdf_hash(pdf)
                    try:
                        result = future.result()
                        if result is None:
                            logging.info(
                                f"No metadata produced for {pdf}; nothing persisted."
                            )
                            continue
                        processed[pdf_hash] = {
                            "input": str(pdf),
                            "output": result.model_dump(),
                        }
                    except Exception as exc:
                        logging.exception(f"Pipeline failed for {pdf}")
                        failed[pdf_hash] = str(exc)
                    finally:
                        progress.update(1)
            print(f"Processed chunk {n}")

    summary = {"processed": processed, "failed": failed}
    return summary


if __name__ == "__main__":
    main()
