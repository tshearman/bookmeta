from typing import Any
from pydantic.dataclasses import dataclass


@dataclass
class GoogleBooksQueryParams:
    intitle: str | None = None
    inauthor: str | None = None
    inpublisher: str | None = None
    subject: str | None = None
    isbn: str | None = None
    lccn: str | None = None
    oclc: str | None = None

    @staticmethod
    def _clause(keyword: str, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.strip()
        return f"+{keyword}:{cleaned}" if cleaned else None

    @property
    def query_params(self) -> dict[str, Any]:
        query = "+".join(
            clause
            for clause in (
                self._clause("intitle", self.intitle),
                self._clause("inauthor", self.inauthor),
                self._clause("inpublisher", self.inpublisher),
                self._clause("subject", self.subject),
                self._clause("isbn", self.isbn),
                self._clause("lccn", self.lccn),
                self._clause("oclc", self.oclc),
            )
            if clause
        )
        return {"q": query}
