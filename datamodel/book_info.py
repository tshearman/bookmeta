from pydantic.dataclasses import dataclass


@dataclass
class BookInfo:
    author: str | None
    title: str | None
    # subtitle: str | None = None
    # publisher: str | None = None
    # subject: str | None = None
    keywords: list[str] | None = None
    # isbn_identifiers: list[str] | None = None
    description: str | None = None
    # series: str | None = None
