from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image
from .img_ocr_result import OcrResult
from typing import NamedTuple


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


@dataclass
class PdfOcrResults:
    pdf_path: str | Path
    combined_text: str | None
    metadata: dict[str, str | None] = field(default_factory=dict)
    ocr_results: list[OCRedPage] = field(default_factory=list)
