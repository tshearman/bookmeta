import argparse
import json
import logging
from pathlib import Path

from openai import OpenAI

from data_store import persist_pipeline_result
from pipeline import (
    run_pipeline,
    PipelineResult,
    invalidate_pipeline_cache_entry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process a PDF, extract key metadata with OpenAI, "
            "and query the Google Books API."
        )
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use for book metadata extraction.",
    )
    parser.add_argument(
        "--secrets-path",
        type=Path,
        default=Path("secrets.json"),
        help="Path to the secrets JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, etc.)",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path(".cache/book_store.json"),
        help="Path to the book store JSON file.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory used to compute relative PDF paths for model context.",
    )
    parser.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="Clear cached pipeline results before running.",
    )
    return parser.parse_args()


def load_credentials(secrets_path: Path) -> tuple[OpenAI, str]:

    with open(secrets_path, "r") as f:
        data = json.load(f)

    openai_api_key = data.get("OPENAI_API_KEY")
    openai_project_id = data.get("OPENAI_PROJECT_ID")
    google_book_api_key = data.get("GOOGLE_BOOKS_API_KEY")

    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in secrets file.")
    if not google_book_api_key:
        raise ValueError("Missing GOOGLE_BOOKS_API_KEY in secrets file.")
    logging.info("Loaded credentials from %s", secrets_path)
    return OpenAI(api_key=openai_api_key, project=openai_project_id), google_book_api_key


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    logging.info("Starting app")

    client, google_key = load_credentials(args.secrets_path)
    if args.invalidate_cache:
        logging.info(
            "Invalidating cached result for pdf=%s model=%s",
            args.pdf_path,
            args.model,
        )
        removed = invalidate_pipeline_cache_entry(
            pdf_path=args.pdf_path,
            model=args.model,
            client=client,
            google_books_api_key=google_key,
            base_dir=args.base_dir,
        )
        if removed:
            logging.info("Cache entry removed; fresh pipeline run will occur")
        else:
            logging.info("No existing cache entry found for this combination")

    result: PipelineResult = run_pipeline(
        pdf_path=args.pdf_path,
        model=args.model,
        client=client,
        google_books_api_key=google_key,
        base_dir=args.base_dir,
    )
    record_id = persist_pipeline_result(result, store_path=args.store_path)
    if record_id:
        logging.info("Persisted pipeline result as %s", record_id)
    else:
        logging.info("Pipeline did not produce BookInfo; nothing persisted")


if __name__ == "__main__":
    main()
