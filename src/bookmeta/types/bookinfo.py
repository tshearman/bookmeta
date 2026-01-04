from dataclasses import dataclass

from pydantic import BaseModel

from bookmeta.types import Pdf
from bookmeta.utils import split_authors


class BookInfo(BaseModel):
    author: str | None
    title: str | None
    subtitle: str | None = None
    publisher: str | None = None
    keywords: list[str] | None = None
    description: str | None = None
    nsfw: bool = False

    def as_detailed_book_info(self) -> "DetailedBookInfo":
        if self.author is None:
            authors = None
        else:
            split = split_authors(self.author)
            if len(split) == 0:
                authors = None
            elif len(split) == 1:
                authors = split[0]
            else:
                authors = split

        return DetailedBookInfo(
            author=authors,
            title=self.title,
            subtitle=self.subtitle,
            publisher=self.publisher,
            subject=None,
            keywords=self.keywords,
            isbn_identifiers=None,
            description=self.description,
            nsfw=self.nsfw,
        )


class DetailedBookInfo(BaseModel):
    author: str | list[str] | None
    title: str | None
    subtitle: str | None = None
    publisher: str | None = None
    subject: str | None = None
    keywords: list[str] | None = None
    isbn_identifiers: list[str] | None = None
    description: str | None = None
    nsfw: bool = False


class BookInfoConfidence(BaseModel):
    author_confidence: float
    title_confidence: float


class BookInfoResponse(BaseModel):
    info: BookInfo
    confidence: BookInfoConfidence


@dataclass(frozen=True)
class BookInfoResult:
    pdf: Pdf
    bookinfo: BookInfoResponse


@dataclass(frozen=True)
class DetailedBookInfoResult:
    pdf: Pdf
    detailed: DetailedBookInfo
