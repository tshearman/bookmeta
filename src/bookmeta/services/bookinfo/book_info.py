import re

from pydantic import BaseModel


def split_authors(author_text: str) -> list[str]:
    """Split a free-form author string on commas and the word 'and'."""
    parts = re.split(r",\s*|\band\b", author_text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


class BookInfo(BaseModel):
    author: str | None
    title: str | None
    subtitle: str | None = None
    publisher: str | None = None
    keywords: list[str] | None = None
    description: str | None = None

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
