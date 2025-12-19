import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.booksearch import BookSearchMethod

LOGGER = logging.getLogger("booksearch.hardcover")

BOOK_WITH_AUTHOR_QUERY = """
query HardcoverBooksByTitleAndAuthor($title: String!, $author: String!, $limit: Int!, $offset: Int!) {
  books(
    where: {
      title: { _eq: $title }
      contributions: { author: { name: { _eq: $author } } }
    }
    limit: $limit
    offset: $offset
    order_by: { users_count: desc }
  ) {
    title
    description
    pages
    release_date
    cached_tags
    contributions {
      author {
        name
      }
    }
  }
}
""".strip()

BOOK_BY_TITLE_QUERY = """
query HardcoverBooksByTitle($title: String!, $limit: Int!, $offset: Int!) {
  books(
    where: { title: { _eq: $title } }
    limit: $limit
    offset: $offset
    order_by: { users_count: desc }
  ) {
    title
    description
    pages
    release_date
    cached_tags
    contributions {
      author {
        name
      }
    }
  }
}
""".strip()


@dataclass
class HardcoverClientConfig:
    api_key: str
    per_page: int = 5
    page: int = 1
    base_url: str = "https://api.hardcover.app/v1/graphql"


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned or None


def _headers(config: HardcoverClientConfig) -> dict[str, str]:
    return {
        "authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def _build_query_inputs(resp: BookInfoResponse) -> dict[str, str | None] | None:
    title = _clean(resp.info.title)
    if not title:
        keywords = " ".join(
            keyword.strip() for keyword in (resp.info.keywords or []) if keyword
        )
        keywords = keywords.strip() or None
        if keywords:
            title = keywords
        elif resp.info.description:
            snippet = resp.info.description.strip()
            if len(snippet) > 200:
                snippet = snippet[:200]
            title = snippet or None
    if not title:
        return None

    author = _clean(resp.info.author)
    return {
        "title": title,
        "author": author,
    }


def _execute_books_query(
    client: httpx.Client,
    config: HardcoverClientConfig,
    title_pattern: str | None,
    author_pattern: str | None,
) -> list[Any]:
    query = BOOK_WITH_AUTHOR_QUERY if author_pattern else BOOK_BY_TITLE_QUERY
    offset = max((config.page - 1) * config.per_page, 0)
    variables: dict[str, Any] = {
        "title": title_pattern,
        "limit": config.per_page,
        "offset": offset,
    }
    if author_pattern:
        variables["author"] = author_pattern
    payload = {"query": query, "variables": variables}
    response = client.post(config.base_url, json=payload, headers=_headers(config))
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        LOGGER.warning(f"Hardcover GraphQL errors: {data['errors']}")
        return []
    books = (data.get("data") or {}).get("books") or []
    return books


def hardcover_search(config: HardcoverClientConfig) -> BookSearchMethod:
    def run(resp: BookInfoResponse) -> str | None:
        inputs = _build_query_inputs(resp)
        if not inputs:
            LOGGER.debug(
                "No Hardcover query inputs could be derived from BookInfoResponse"
            )
            return None

        title = inputs["title"]
        author = inputs["author"]
        results: list[Any] = []
        with httpx.Client(timeout=10) as client:
            try:
                results = _execute_books_query(
                    client,
                    config,
                    title_pattern=title,
                    author_pattern=author,
                )
                if not results and author:
                    LOGGER.debug(
                        "Hardcover combined query returned no results; retrying with title only."
                    )
                    results = _execute_books_query(
                        client,
                        config,
                        title_pattern=title,
                        author_pattern=None,
                    )
            except Exception:
                LOGGER.exception(
                    f"Hardcover query failed for title pattern '{title}' author pattern '{author}'"
                )
                return None

        if not results:
            LOGGER.info(
                f"Hardcover query returned no results for title pattern '{title}' and author pattern '{author}'"
            )
            return None

        payload = {
            "source": "hardcover",
            "query": {
                "title_pattern": title,
                "author_pattern": author,
            },
            "result_count": len(results),
            "items": results,
        }
        return json.dumps(payload)

    return run


def main() -> None:
    import argparse
    import logging

    from bookmeta.services.bookinfo.book_info import BookInfo
    from bookmeta.services.bookinfo.book_info_confidence import BookInfoConfidence

    parser = argparse.ArgumentParser(description="Test the Hardcover search provider.")
    parser.add_argument(
        "--title",
        default=None,
        help="Book title (required unless keywords/description provided).",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Optional author paired with the title.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Optional keywords to fall back on when title missing.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional description snippet to fall back on.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=5,
        help="Number of Hardcover results per query.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to fetch from Hardcover search.",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.json"),
        help="Path to secrets JSON containing HARDCOVER_API_KEY.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    try:
        with args.secrets.open("r", encoding="utf-8") as fh:
            secrets = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Secrets file not found at {args.secrets}.") from exc

    api_key = secrets.get("HARDCOVER_API_KEY")
    if not api_key:
        raise ValueError(
            "HARDCOVER_API_KEY missing from secrets. Provide it via the secrets file."
        )

    resp = BookInfoResponse(
        info=BookInfo(
            title=args.title,
            author=args.author,
            keywords=args.keywords,
            description=args.description,
        ),
        confidence=BookInfoConfidence(author_confidence=1.0, title_confidence=1.0),
    )
    method = hardcover_search(
        HardcoverClientConfig(
            api_key=api_key,
            per_page=args.per_page,
            page=args.page,
        )
    )
    payload = method(resp)
    if payload is None:
        print("Hardcover provider returned no results.")
    else:
        print(payload)


if __name__ == "__main__":
    main()
