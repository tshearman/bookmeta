import logging
import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from bookinfo.book_info_response import BookInfoResponse
from booksearch import BookSearchMethod


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
