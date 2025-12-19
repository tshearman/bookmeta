import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.booksearch import BookSearchMethod
from bookmeta.services.booksearch.google_books_query_params import (
    GoogleBooksQueryParams,
)
from bookmeta.services.booksearch.providers.retry import (
    is_retryable_httpx_error,
    retryable_request,
)

LOGGER = logging.getLogger("booksearch.googlebooks")


@dataclass
class GoogleBooksClientConfig:
    api_key: str
    max_results: int = 5


def _build_query(resp: BookInfoResponse) -> GoogleBooksQueryParams:
    return GoogleBooksQueryParams(
        intitle=resp.info.title,
        inauthor=resp.info.author,
    )


def _simplify_item(item: dict[str, Any]) -> dict[str, Any]:
    info = item.get("volumeInfo", {}) or {}
    identifiers = info.get("industryIdentifiers") or []
    return {
        "title": info.get("title"),
        "subtitle": info.get("subtitle"),
        "authors": info.get("authors"),
        "publisher": info.get("publisher"),
        "publishedDate": info.get("publishedDate"),
        "description": info.get("description"),
        "categories": info.get("categories"),
        "industryIdentifiers": identifiers,
        "pageCount": info.get("pageCount"),
        "language": info.get("language"),
        "previewLink": info.get("previewLink"),
    }


def googlebooks_search(config: GoogleBooksClientConfig) -> BookSearchMethod:
    base_url = "https://www.googleapis.com/books/v1/volumes"

    @retryable_request(LOGGER)
    def _fetch(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
        response = client.get(base_url, params=params)
        response.raise_for_status()
        return response.json()

    def run(resp: BookInfoResponse) -> str | None:
        params = _build_query(resp).query_params
        if not params.get("q"):
            return None
        params["maxResults"] = config.max_results
        params["key"] = config.api_key

        with httpx.Client(timeout=10) as client:
            try:
                payload = _fetch(client, params)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if is_retryable_httpx_error(exc):
                    LOGGER.warning(
                        "Google Books request failed after retries with HTTP %s; skipping this search.",
                        status,
                    )
                else:
                    LOGGER.exception(
                        "Google Books HTTP error %s; skipping this search.", status
                    )
                return None
            except httpx.RequestError:
                LOGGER.warning(
                    "Google Books request failed after retries due to connection error; skipping this search."
                )
                return None
        raw_items = payload.get("items", []) or []
        if not raw_items:
            return None
        simplified = [_simplify_item(item) for item in raw_items]
        output = {
            "source": "google_books",
            "query": params["q"],
            "result_count": len(simplified),
            "items": simplified,
        }
        return json.dumps(output)

    return run
