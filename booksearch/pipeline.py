from __future__ import annotations

import logging
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from datamodel.book_info_response import BookInfoResponse
from booksearch import BookSearchMethod
from datamodel.pdf_ocr_results import PdfOcrResults


LOGGER = logging.getLogger("booksearch.pipeline")


@dataclass
class BookSearchResult:
    method: str
    payload: str


@dataclass
class BookSearchResults:
    source: BookInfoResponse
    results: list[BookSearchResult] = field(default_factory=list)


BookSearchPipeline = Callable[[BookInfoResponse], BookSearchResults]


@dataclass
class BookSearchPipelineConfig:
    search_methods: Sequence[BookSearchMethod]


def _method_name(method: BookSearchMethod) -> str:
    if hasattr(method, "__name__"):
        return method.__name__  # type: ignore[attr-defined]
    return method.__class__.__name__


def generate_pipeline(config: BookSearchPipelineConfig) -> BookSearchPipeline:
    if not config.search_methods:
        raise ValueError("At least one BookSearchMethod must be provided.")

    methods = tuple(config.search_methods)

    def run(resp: BookInfoResponse) -> BookSearchResults:
        results: list[BookSearchResult] = []
        for method in methods:
            method_name = _method_name(method)
            try:
                payload = method(resp)
            except Exception:
                LOGGER.exception("Book search method %s failed; skipping", method_name)
                continue
            if not payload:
                LOGGER.info("Book search method %s produced no payload", method_name)
                continue
            results.append(BookSearchResult(method=method_name, payload=payload))
        return BookSearchResults(source=resp, results=results)

    return run


if __name__ == "__main__":
    from bookinfo.pipeline import (
        BookInfoPipelineConfig,
        generate_pipeline as generate_bookinfo_pipeline,
    )
    from booksearch.providers.googlebooks import (
        GoogleBooksClientConfig,
        googlebooks_search,
    )
    from ocr.pipeline import (
        OcrPipelineConfig,
        generate_pipeline as generate_ocr_pipeline,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    sample_pdf = project_root / "resources" / "test_pdfs" / "bladesinthedark_v8_2.pdf"
    secrets_path = project_root / "secrets.json"

    if not sample_pdf.exists():
        raise FileNotFoundError(f"Sample PDF not found: {sample_pdf}")
    if not secrets_path.exists():
        raise FileNotFoundError("secrets.json is required to run the pipeline demo.")

    with secrets_path.open("r") as fh:
        secrets = json.load(fh)

    openai_api_key = secrets.get("OPENAI_API_KEY")
    google_api_key = secrets.get("GOOGLE_BOOKS_API_KEY")
    if not openai_api_key or not google_api_key:
        raise ValueError("Both OPENAI_API_KEY and GOOGLE_BOOKS_API_KEY are required.")

    logging.info("Running OCR pipeline...")
    ocr_pipeline = generate_ocr_pipeline(OcrPipelineConfig())
    ocr_results = ocr_pipeline(sample_pdf)
    assert isinstance(ocr_results, PdfOcrResults)
    logging.info("OCR produced %d page samples", len(ocr_results.ocr_results))

    logging.info("Running BookInfo pipeline (OpenAI)...")
    bookinfo_pipeline = generate_bookinfo_pipeline(
        BookInfoPipelineConfig(
            provider="openai",
            client_config={
                "api_key": openai_api_key,
                "project": secrets.get("OPENAI_PROJECT_ID"),
            },
        )
    )
    bookinfo_response = bookinfo_pipeline(ocr_results)
    assert isinstance(bookinfo_response, BookInfoResponse)
    if not bookinfo_response:
        raise RuntimeError("BookInfo pipeline returned no response")
    logging.info("Initial BookInfo result:\n%s", bookinfo_response)

    logging.info("Running BookSearch pipeline...")
    search_methods = [
        googlebooks_search(
            GoogleBooksClientConfig(api_key=google_api_key, max_results=3)
        )
    ]
    booksearch_pipeline = generate_pipeline(
        BookSearchPipelineConfig(search_methods=search_methods)
    )
    results = booksearch_pipeline(bookinfo_response)
    logging.info("BookSearch produced %d payload(s)", len(results.results))
    for result in results.results:
        logging.info("Method: %s\nPayload:\n%s\n", result.method, result.payload)
