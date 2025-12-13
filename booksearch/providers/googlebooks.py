from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from booksearch import BookSearchPipeline
from datamodel.book_info import BookInfo
from datamodel.book_info_response import BookInfoResponse
from datamodel.google_books_query_params import GoogleBooksQueryParams
from google_books_volume import GoogleBooksVolume


@dataclass
class GoogleBooksClientConfig:
    api_key: str
    max_results: int = 5


def _build_query(resp: BookInfoResponse) -> GoogleBooksQueryParams:
    return GoogleBooksQueryParams(
        intitle=resp.info.title,
        inauthor=resp.info.author,
    )


def _volume_to_bookinfo(volume: GoogleBooksVolume) -> BookInfo | None:
    if volume.volume_info:
        info = volume.volume_info
        isbns = info.industry_identifiers
        return BookInfo(
            author=",".join(info.authors),
            title=info.title,
            keywords=info.categories,
            isbn_identifiers=[i.identifier for i in isbns if i.identifier],
            description=info.description,
        )


def _volume_dict_to_bookinfo(d: dict) -> BookInfo | None:
    return _volume_to_bookinfo(GoogleBooksVolume.from_dict(d))


def googlebooks_search(config: GoogleBooksClientConfig) -> BookSearchPipeline:
    base_url = "https://www.googleapis.com/books/v1/volumes"

    def run(resp: BookInfoResponse) -> list[BookInfo]:
        params = _build_query(resp).query_params
        params["maxResults"] = config.max_results
        params["key"] = config.api_key

        with httpx.Client(timeout=10) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()
            payload = response.json()
        items = payload.get("items", []) or []
        volumes = [_volume_dict_to_bookinfo(item) for item in items]
        return [v for v in volumes if v]

    return run
