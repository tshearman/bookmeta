from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from booksearch import BookSearchMethod
from datamodel.book_info_response import BookInfoResponse
from datamodel.google_books_query_params import GoogleBooksQueryParams


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

    def run(resp: BookInfoResponse) -> str | None:
        params = _build_query(resp).query_params
        if not params.get("q"):
            return None
        params["maxResults"] = config.max_results
        params["key"] = config.api_key

        with httpx.Client(timeout=10) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()
            payload = response.json()
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
