import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import joblib
import ollama

from bookmeta.config.settings import DEFAULT_DB_PATH, PIPELINE_CACHE_DIR
from bookmeta.data.sqlite import persist_run, serialize_pipeline_config
from bookmeta.services.bookinfo import BOOK_PROMPT, BOOK_PROMPT_TTRPG
from bookmeta.services.bookinfo.blocks import ContextLimits
from bookmeta.services.bookinfo.book_info import DetailedBookInfo
from bookmeta.services.bookinfo.pipeline import (
    BookInfoPipelineConfig,
)
from bookmeta.services.bookinfo.pipeline import (
    generate_pipeline as generate_bookinfo_pipeline,
)
from bookmeta.services.booksearch import BookSearchMethod
from bookmeta.services.booksearch.pipeline import (
    BookSearchPipelineConfig,
)
from bookmeta.services.booksearch.pipeline import (
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
from bookmeta.services.ocr.pipeline import OcrPipelineConfig
from bookmeta.services.ocr.pipeline import generate_pipeline as generate_ocr_pipeline
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
    mode: Literal["full", "bookinfo_only"] = "full"


class NoOcrTextError(RuntimeError):
    """Raised when no OCR method extracted any text from a PDF."""


LOGGER = logging.getLogger(__name__)
BookMetaPipeline = Callable[[Path], DetailedBookInfo]

PDF_MEMORY = joblib.Memory(PIPELINE_CACHE_DIR, verbose=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a PDF and produce refined book metadata."
    )
    parser.add_argument(
        "pdf_path",
        nargs="+",
        type=Path,
        help="Path(s) to the input PDF(s). Supports globs like 'files/books_*'.",
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
        help="Number of pages from the front of the document to process.",
    )
    parser.add_argument(
        "--num-back-ocr-pages",
        type=int,
        default=3,
        help="Number of pages from the back of the document to process.",
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
        "--search-max-results",
        type=int,
        default=3,
        help="Maximum Book Search results to fetch during book search.",
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
    args = parser.parse_args()
    for name in (
        "context_first_images",
        "context_last_images",
        "context_first_ocr_pages",
        "context_last_ocr_pages",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    return args


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
    for page in ocr_results.pages:
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
                    api_key=api_key, max_results=args.search_max_results
                )
            )
        )

    # if "HARDCOVER_API_KEY" in secrets:
    #     api_key = secrets["HARDCOVER_API_KEY"]
    #     methods.append(
    #         hardcover_search(
    #             HardcoverClientConfig(api_key=api_key, per_page=args.search_max_results)
    #         )
    #     )

    return methods


def build_pipeline_config(
    args: argparse.Namespace, secrets: dict[str, Any]
) -> PipelineConfig:

    prompt = BOOK_PROMPT_TTRPG if args.book_prompt == "ttrpg" else BOOK_PROMPT
    context_limits = ContextLimits(
        num_first_images=args.context_first_images,
        num_last_images=args.context_last_images,
        num_first_ocr_pages=args.context_first_ocr_pages,
        num_last_ocr_pages=args.context_last_ocr_pages,
    )

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
        context_limits=context_limits,
        prompt=prompt,
    )
    selection_config = BookInfoSelectionPipelineConfig(
        provider=args.selection_provider,
        client_config=_client_config_for(args.selection_provider, secrets),
        model=args.selection_model,
        context_limits=context_limits,
    )
    methods = _default_booksearch_methods(args, secrets)
    booksearch_config = BookSearchPipelineConfig(
        search_methods=methods, num_responses=args.search_max_results
    )
    mode = "bookinfo_only" if args.pipeline_mode == "bookinfo-only" else "full"

    return PipelineConfig(
        ocr_config=ocr_config,
        extraction_config=extraction_config,
        selection_config=selection_config,
        booksearch_config=booksearch_config,
        mode=mode,
    )


def _full_pipeline(config: PipelineConfig) -> BookMetaPipeline:
    ocr_pipeline = generate_ocr_pipeline(config.ocr_config)
    info_pipeline = generate_bookinfo_pipeline(config.extraction_config)
    search_pipeline = generate_booksearch_pipeline(config.booksearch_config)
    selection_pipeline = generate_selection_pipeline(config.selection_config)

    def _inner_(pdf_path: Path) -> DetailedBookInfo:
        ocr_results = ocr_pipeline(pdf_path)
        if not _has_ocr_text(ocr_results):
            LOGGER.warning(f"Skipping {pdf_path} because OCR produced no text.")
            raise NoOcrTextError(f"No OCR text extracted for {pdf_path}")
        search_results = search_pipeline(info_pipeline(ocr_results))
        return selection_pipeline(ocr_results, search_results)

    return _inner_


def bookinfo_only_pipeline(config: PipelineConfig) -> BookMetaPipeline:
    """Simplified pipeline that stops after OCR + BookInfo extraction."""

    ocr_pipeline = generate_ocr_pipeline(config.ocr_config)
    info_pipeline = generate_bookinfo_pipeline(config.extraction_config)

    def _inner_(pdf_path: Path) -> DetailedBookInfo:
        ocr_results = ocr_pipeline(pdf_path)
        if not _has_ocr_text(ocr_results):
            LOGGER.warning(f"Skipping {pdf_path} because OCR produced no text.")
            raise NoOcrTextError(f"No OCR text extracted for {pdf_path}")
        info_response = info_pipeline(ocr_results)
        if info_response is None:
            raise RuntimeError(f"BookInfo extraction returned no result for {pdf_path}")
        # Promote BookInfoResponse to DetailedBookInfo shape for signature parity.
        info = info_response.info
        return DetailedBookInfo(
            author=info.author,
            title=info.title,
            subtitle=None,
            publisher=None,
            subject=None,
            keywords=info.keywords,
            isbn_identifiers=None,
            description=info.description,
        )

    return _inner_


def pipeline(config: PipelineConfig) -> BookMetaPipeline:
    """Return the configured pipeline (full or bookinfo-only)."""
    if config.mode == "bookinfo_only":
        return bookinfo_only_pipeline(config)
    return _full_pipeline(config)


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
    LOGGER.info(f"Running pipeline on {pdf}")
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


def _expand_paths(patterns: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for pattern in patterns:
        pattern = pattern.expanduser()
        text = str(pattern)
        if any(ch in text for ch in "*?[]"):
            matches = list(Path().glob(text))
            if not matches:
                LOGGER.warning(f"No files matched pattern: {pattern}")
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    return expanded


def main() -> list[DetailedBookInfo]:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    secrets = _read_secrets(args.secrets)
    config = build_pipeline_config(args, secrets)
    pdf_paths = _expand_paths(args.pdf_path)
    if not pdf_paths:
        raise FileNotFoundError("No PDF paths matched the provided arguments.")

    results: list[DetailedBookInfo] = []
    for pdf_path in pdf_paths:
        result = process_pdf(pdf_path, config, args.results_db)
        if result is not None:
            print(json.dumps(result.model_dump(), indent=2))
            results.append(result)
    return results


if __name__ == "__main__":
    out = main()
