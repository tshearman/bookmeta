from typing import Any
from joblib import Memory
from pydantic import BaseModel, Field
import httpx
import logging

from google_books_volume import GoogleBooksVolume


GOOGLE_BOOKS_MEMORY = Memory(location=".cache/openai_google_books", verbose=0)


class GoogleBooksQuery(BaseModel):
    """Fields to populate a Google Books `q=` query string."""

    generalquery: str = Field(description="General fuzzy query information")
    inauthor: str = Field(description="Primary Author Name")
    author_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence (0-1) that the extracted primary author is correct",
    )
    intitle: str | None = Field(
        default=None, description="Book title for the intitle: search operator."
    )
    title_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence (0-1) that the extracted title is correct",
    )
    inpublisher: str | None = Field(default=None, description="Publisher name")
    subject: str | None = Field(default=None, description="Subject/category")
    isbn: str | None = Field(default=None, description="10- or 13-digit ISBN")
    isbn_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence (0-1) that the extracted ISBN is correct",
    )
    lccn: str | None = None
    oclc: str | None = None
    tags: list[str] | None = None

    @property
    def query_params(self) -> str:
        query_str = ""
        if not self.intitle:
            query_str += f"{self.generalquery}"
        if self.intitle:
            query_str += f"+intitle:{self.intitle}"
        if self.inauthor:
            query_str += f"+inauthor:{self.inauthor}"
        return query_str


def construct_query_params(search: GoogleBooksQuery, key: str, max_results: int = 5):
    return {"q": search.query_params, "key": key, "maxResults": max_results}


def redact_url(url_):
    query = dict(url_.params)
    if "key" in query:
        query["key"] = "***REDACTED***"
    redacted = url_.copy_with(params=query)
    return redacted


@GOOGLE_BOOKS_MEMORY.cache(ignore=["api_key"])
def google_books_api_request(params: dict[str, Any], api_key: str):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params["key"] = api_key
    with httpx.Client() as client:
        resp = client.get(base_url, params=params)
        logging.info("Google Books request URL: %s", redact_url(resp.request.url))
        if resp.status_code != 200:
            raise RuntimeError(
                f"Google Books request failed with status {resp.status_code}"
            )
        resp.raise_for_status()
        return resp.json()


def fetch_google_books(search: GoogleBooksQuery, key: str) -> list[GoogleBooksVolume]:
    params = construct_query_params(search, key)
    volumes = google_books_api_request(params, key)
    items = volumes.get("items", []) or []
    return [GoogleBooksVolume.from_dict(d) for d in items]


if __name__ == "__main__":
    import json

    with open("./secrets.json", "r") as f:
        google_key = json.load(f)["GOOGLE_BOOKS_API_KEY"]
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    params = GoogleBooksQuery(
        generalquery="", inauthor="Michael Sands", intitle="Monster of the Week"
    )
    logging.debug(params)
    book_responses = fetch_google_books(params, google_key)
    logging.debug(book_responses)
    volumes = book_responses
    for v in volumes:
        logging.debug(v)
