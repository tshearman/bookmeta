from pydantic import BaseModel


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
            authors = [a.strip() for a in self.author.split(",") if a.strip()]
            if len(authors) == 1:
                authors = authors[0]

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
