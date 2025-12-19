import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bookmeta.cli.pipeline import _read_secrets, build_pipeline_config, process_pdf
from bookmeta.config.settings import DEFAULT_DB_PATH
from bookmeta.data.sqlite import _compute_pdf_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process all PDFs within a directory and produce refined book metadata."
    )
    parser.add_argument(
        "pdf_directory",
        type=Path,
        help="Root directory to scan recursively for PDFs.",
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
        "--google-max-results",
        type=int,
        default=3,
        help="Maximum Google Books results to fetch during book search.",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite DB used to log pipeline runs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARN, ERROR).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Number of pipeline executions to run in parallel.",
    )
    return parser.parse_args()


def _discover_pdfs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"PDF directory is not a directory: {root}")
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def _dedupe_pdfs(paths: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    duplicates = 0
    unique: list[Path] = []
    for pdf in paths:
        pdf_hash = _compute_pdf_hash(pdf)
        if pdf_hash in seen:
            duplicates += 1
            logging.info(
                f"Skipping duplicate PDF {pdf} (same hash as {seen[pdf_hash]})"
            )
        else:
            seen[pdf_hash] = pdf
            unique.append(pdf)
    if duplicates:
        logging.info(f"Deduplicated {duplicates} PDFs with identical hashes.")
    return unique


def main() -> dict[str, Any]:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    config = build_pipeline_config(args, secrets)

    pdfs = _discover_pdfs(args.pdf_directory)
    if not pdfs:
        logging.warning(f"No PDFs found inside {args.pdf_directory}")
        return {"processed": {}, "failed": {}}
    pdfs = _dedupe_pdfs(pdfs)
    if not pdfs:
        logging.warning(
            f"All discovered PDFs under {args.pdf_directory} were duplicates; nothing to process."
        )
        return {"processed": {}, "failed": {}}

    workers = max(1, args.workers)
    logging.info(
        f"Discovered {len(pdfs)} PDFs under {args.pdf_directory}. Processing with {workers} workers."
    )

    processed: dict[str, Any] = {}
    failed: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_pdf, pdf, config, args.results_db): pdf
            for pdf in pdfs
        }
        for future in as_completed(futures):
            pdf = futures[future]
            pdf_hash = _compute_pdf_hash(pdf)
            try:
                result = future.result()
                if result is None:
                    logging.info(f"No metadata produced for {pdf}; nothing persisted.")
                    continue
                processed[pdf_hash] = {"input": str(pdf), "output": result.model_dump()}
            except Exception as exc:
                logging.exception(f"Pipeline failed for {pdf}")
                failed[pdf_hash] = str(exc)

    summary = {"processed": processed, "failed": failed}
    return summary


if __name__ == "__main__":
    output = main()
