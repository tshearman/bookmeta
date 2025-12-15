import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline import (
    PipelineConfig,
    _read_secrets,
    build_pipeline_config,
    execute_pipeline,
)
from storage import (
    DEFAULT_DB_PATH,
    _compute_pdf_hash,
    persist_run,
    serialize_pipeline_config,
)


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
        default=3,
        help="Number of front pages to OCR.",
    )
    parser.add_argument(
        "--num-back-ocr-pages",
        type=int,
        default=2,
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


def _process_pdf(
    pdf: Path, config: PipelineConfig, config_signature: str, results_db: Path
) -> Any:
    logging.info("Running pipeline on %s", pdf)
    final_info = execute_pipeline(pdf, config, config_signature)
    logging.info("Final BookInfo for %s:\n%s", pdf, final_info)
    try:
        persist_run(results_db, pdf, config, final_info)
        logging.info("Persisted pipeline run to %s for %s", results_db, pdf)
    except Exception:
        logging.exception("Failed to persist pipeline run for %s", pdf)
    return final_info


def main() -> dict[str, Any]:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    config = build_pipeline_config(args, secrets)
    config_signature = json.dumps(serialize_pipeline_config(config), sort_keys=True)

    pdfs = _discover_pdfs(args.pdf_directory)
    if not pdfs:
        logging.warning("No PDFs found inside %s", args.pdf_directory)
        return {"processed": {}, "failed": {}}

    workers = max(1, args.workers)
    logging.info(
        "Discovered %d PDFs under %s. Processing with %d workers.",
        len(pdfs),
        args.pdf_directory,
        workers,
    )

    processed: dict[str, Any] = {}
    failed: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_pdf, pdf, config, config_signature, args.results_db
            ): pdf
            for pdf in pdfs
        }
        for future in as_completed(futures):
            pdf = futures[future]
            pdf_hash = _compute_pdf_hash(pdf)
            try:
                result = future.result()
                processed[pdf_hash] = {"input": str(pdf), "output": result.model_dump()}
            except Exception as exc:
                logging.exception("Pipeline failed for %s", pdf)
                failed[pdf_hash] = str(exc)

    summary = {"processed": processed, "failed": failed}
    return summary


if __name__ == "__main__":
    output = main()
