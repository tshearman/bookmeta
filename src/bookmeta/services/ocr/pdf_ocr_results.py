from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from PIL import Image

from .img_ocr_result import OcrResult


class OCRedPage(NamedTuple):
    page_number: int
    image: Image.Image
    ocr_results: list[OcrResult]

    def __repr__(self) -> str:
        lines = []
        for ocr_result in self.ocr_results:
            if ocr_result.text:
                cleaned = ocr_result.text.strip()
                if cleaned:
                    lines.append(
                        f"=== Page {self.page_number} - {ocr_result.method} ==="
                    )
                    lines.append(cleaned)
                    lines.append("")
        return "\n".join(lines).strip()


def build_combined_ocr_text(page_results: list[OCRedPage]) -> str:
    return "\n".join([str(page_result) for page_result in page_results])


class PageOcrResult(NamedTuple):
    page_number: int
    ocr_result: OcrResult


@dataclass
class PdfOcrResults:
    pdf_path: str | Path
    combined_text: str | None
    metadata: dict[str, str | None] = field(default_factory=dict)
    pages: list[OCRedPage] = field(default_factory=list)

    def _select_pages(
        self, n_first: int | None = None, n_end: int | None = None
    ) -> list[OCRedPage]:
        """Return a deduped list of pages keeping order by page_number."""
        ordered_pages = sorted(self.pages, key=lambda page: page.page_number)
        if n_first is None and n_end is None:
            return ordered_pages

        first = ordered_pages[: n_first or 0]
        last = ordered_pages[-(n_end or 0) :] if n_end else []

        seen: set[int] = set()
        selected: list[OCRedPage] = []
        for page in first + last:
            if page.page_number in seen:
                continue
            selected.append(page)
            seen.add(page.page_number)
        return selected

    def images(
        self, n_first: int | None = None, n_end: int | None = None
    ) -> list[Image.Image]:
        """Return page images from the first/last slices requested."""
        return [page.image for page in self._select_pages(n_first, n_end)]

    def ocr_results(
        self, n_first: int | None = None, n_end: int | None = None
    ) -> list[PageOcrResult]:
        """Return OCR results (with page numbers) from the requested slices."""
        results: list[PageOcrResult] = []
        for page in self._select_pages(n_first, n_end):
            for ocr_result in page.ocr_results:
                results.append(PageOcrResult(page.page_number, ocr_result))
        return results
