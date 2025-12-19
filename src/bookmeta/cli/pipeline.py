import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import ollama

from bookmeta.config.settings import DEFAULT_DB_PATH, PIPELINE_CACHE_DIR
from bookmeta.data.sqlite import persist_run, serialize_pipeline_config
from bookmeta.services.bookinfo.pipeline import (
    BookInfoPipelineConfig,
    generate_pipeline as generate_bookinfo_pipeline,
)
from bookmeta.services.bookinfo.book_info import DetailedBookInfo
from bookmeta.services.booksearch import BookSearchMethod
from bookmeta.services.booksearch.pipeline import (
    BookSearchPipelineConfig,
    generate_pipeline as generate_booksearch_pipeline,
)
from bookmeta.services.booksearch.providers.googlebooks import (
    GoogleBooksClientConfig,
    googlebooks_search,
)
from bookmeta.services.booksearch.providers.hardcover import (
    HardcoverClientConfig,
    hardcover_search,
)
from bookmeta.services.ocr.ocr import (
    native_ocr_method,
    ollama_ocr_method,
    tesseract_ocr_method,
)
from bookmeta.services.ocr.pdf_ocr_results import PdfOcrResults
from bookmeta.services.ocr.pipeline import (
    OcrPipelineConfig,
    generate_pipeline as generate_ocr_pipeline,
)
from bookmeta.services.rank.pipeline import (
    BookInfoSelectionPipelineConfig,
    generate_selection_pipeline,
)


@dataclass
class PipelineConfig:
    ocr_config: OcrPipelineConfig
    extraction_config: BookInfoPipelineConfig
    selection_config: BookInfoSelectionPipelineConfig
    booksearch_config: BookSearchPipelineConfig


class NoOcrTextError(RuntimeError):
    """Raised when no OCR method extracted any text from a PDF."""


LOGGER = logging.getLogger(__name__)
BookMetaPipeline = Callable[[Path], DetailedBookInfo]

PDF_MEMORY = joblib.Memory(PIPELINE_CACHE_DIR, verbose=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a PDF and produce refined book metadata."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF.")
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
        help="Path to secrets JSON with API keys.",
    )
    parser.add_argument(
        "--num-back-ocr-pages",
        type=int,
        default=2,
        help="Path to secrets JSON with API keys.",
    )
    parser.add_argument(
        "--ocr-ollama-model",
        type=str,
        default=None,
        help="Provider for the initial BookInfo extraction stage.",
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
        "--hardcover-per-page",
        type=int,
        default=5,
        help="Maximum Hardcover results per query.",
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
    return parser.parse_args()


def _read_secrets(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def _client_config_for(provider: str, secrets: dict[str, Any]) -> dict[str, Any]:
    if provider == "openai":
        api_key = secrets.get("OPENAI_API_KEY")
        project = secrets.get("OPENAI_PROJECT_ID")
        return {"api_key": api_key, "project": project}
    if provider == "ollama":
        host = secrets.get("OLLAMA_HOST")
        return {"host": host}
    raise ValueError(f"Unsupported provider: {provider}")


def _has_ocr_text(ocr_results: PdfOcrResults) -> bool:
    combined = (ocr_results.combined_text or "").strip()
    if combined:
        return True
    pages = ocr_results.ocr_results
    for page in pages:
        for result in page.ocr_results:
            text = result.text
            if text and text.strip():
                return True
    return False


def _default_booksearch_methods(
    args: argparse.Namespace, secrets: dict[str, Any]
) -> list[BookSearchMethod]:

    methods: list[BookSearchMethod] = []
    if "GOOGLE_BOOKS_API_KEY" in secrets:
        api_key = secrets["GOOGLE_BOOKS_API_KEY"]
        methods.append(
            googlebooks_search(
                GoogleBooksClientConfig(
                    api_key=api_key, max_results=args.google_max_results
                )
            )
        )

    if "HARDCOVER_API_KEY" in secrets:
        api_key = secrets["HARDCOVER_API_KEY"]
        methods.append(
            hardcover_search(
                HardcoverClientConfig(api_key=api_key, per_page=args.hardcover_per_page)
            )
        )

    return methods


def build_pipeline_config(
    args: argparse.Namespace, secrets: dict[str, Any]
) -> PipelineConfig:

    ocr_methods = [native_ocr_method, tesseract_ocr_method]
    if args.ocr_ollama_model is not None:
        llm_ocr = ollama_ocr_method(
            ollama.Client(secrets["OLLAMA_HOST"]), args.ocr_ollama_model
        )
        ocr_methods.append(llm_ocr)  # type: ignore

    ocr_config = OcrPipelineConfig(ocr_methods=ocr_methods)
    extraction_config = BookInfoPipelineConfig(
        provider=args.extraction_provider,
        client_config=_client_config_for(args.extraction_provider, secrets),
        model=args.extraction_model,
    )
    selection_config = BookInfoSelectionPipelineConfig(
        provider=args.selection_provider,
        client_config=_client_config_for(args.selection_provider, secrets),
        model=args.selection_model,
    )
    methods = _default_booksearch_methods(args, secrets)
    booksearch_config = BookSearchPipelineConfig(search_methods=methods)

    return PipelineConfig(
        ocr_config=ocr_config,
        extraction_config=extraction_config,
        selection_config=selection_config,
        booksearch_config=booksearch_config,
    )


def pipeline(config: PipelineConfig) -> BookMetaPipeline:
    ocr_pipeline = generate_ocr_pipeline(config.ocr_config)
    info_pipeline = generate_bookinfo_pipeline(config.extraction_config)
    search_pipeline = generate_booksearch_pipeline(config.booksearch_config)
    selection_pipeline = generate_selection_pipeline(config.selection_config)

    def _inner_(pdf_path: Path) -> DetailedBookInfo:
        ocr_results = ocr_pipeline(pdf_path)
        if not _has_ocr_text(ocr_results):
            LOGGER.warning("Skipping %s because OCR produced no text.", pdf_path)
            raise NoOcrTextError(f"No OCR text extracted for {pdf_path}")
        search_results = search_pipeline(info_pipeline(ocr_results))
        return selection_pipeline(ocr_results, search_results)

    return _inner_


@PDF_MEMORY.cache(ignore=["config"])
def execute_pipeline(
    pdf: Path, config: PipelineConfig, config_signature: str
) -> DetailedBookInfo:
    return pipeline(config)(pdf)


def process_pdf(
    pdf: Path,
    config: PipelineConfig,
    results_db: Path,
) -> DetailedBookInfo | None:
    LOGGER.info("Running pipeline on %s", pdf)
    config_signature = json.dumps(serialize_pipeline_config(config), sort_keys=True)
    try:
        final_info = execute_pipeline(pdf, config, config_signature)
    except NoOcrTextError as exc:
        LOGGER.warning(f"Skipping {pdf}: {exc}")
        return

    LOGGER.info(f"Final BookInfo for {pdf}:\n{final_info}")
    try:
        persist_run(results_db, pdf, config, final_info)
        LOGGER.info(f"Persisted pipeline run to {results_db} for {pdf}")
    except Exception:
        LOGGER.exception(f"Failed to persist pipeline run for {pdf}")
    return final_info


def main() -> DetailedBookInfo | None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    config = build_pipeline_config(args, secrets)
    return process_pdf(args.pdf_path, config, args.results_db)


if __name__ == "__main__":
    out = main()
    if out is not None:
        print(json.dumps(out.model_dump(), indent=2))
