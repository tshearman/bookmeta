from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz
from PIL import Image, ImageOps
from pydantic import BaseModel
import pytesseract


class PdfPageEntry(BaseModel):
    page_number: int
    text: dict[str, str | None]
    image_path: str


class PdfProcessingResult(BaseModel):
    pdf_path: str
    metadata: Dict[str, str | None]
    pages: list[PdfPageEntry]
    combined_text: str

    def combined_text_for(self, source: str) -> str:
        """
        Aggregate OCR text for the requested source (``tesseract_ocr`` or ``native_ocr``)
        across all processed pages.
        """
        source = source.strip().lower()
        if source not in {"tesseract_ocr", "native_ocr"}:
            raise ValueError("source must be 'tesseract' or 'native'")

        lines: list[str] = []
        for page in self.pages:
            page_text = page.text or {}
            value = page_text.get(source, None)
            if value:
                lines.append(f"=== Page {page.page_number} - {source} ===")
                lines.append(value)
                lines.append("")
        return "\n".join(lines).strip()


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Convert to grayscale and apply light contrast enhancement for better OCR."""
    gray = image.convert("L")
    return ImageOps.autocontrast(gray)


def _page_is_blank(
    page: fitz.Page,
    white_threshold: int = 250,
    max_nonwhite_ratio: float = 0.005,
) -> bool:
    """
    Heuristic to decide if a page is effectively blank.

    Render at low resolution, count non-white pixels, and compare the ratio to
    ``max_nonwhite_ratio``.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
    samples = pix.samples
    n_channels = pix.n
    total_pixels = pix.width * pix.height

    if total_pixels == 0:
        return True

    nonwhite = 0
    for i in range(0, len(samples), n_channels):
        if any(c < white_threshold for c in samples[i : i + n_channels]):
            nonwhite += 1

    return (nonwhite / total_pixels) <= max_nonwhite_ratio


def _nonblank_sample_indices(
    doc: fitz.Document,
    num_first_pages: int = 3,
    num_last_pages: int = 1,
) -> List[int]:
    """
    Return indices for the first ``num_first_pages`` nonblank pages plus the last
    ``num_last_pages`` nonblank pages.
    """
    n_pages = len(doc)
    if n_pages == 0:
        return []

    selected: set[int] = set()
    first_indices: List[int] = []
    last_indices: List[int] = []

    for idx in range(n_pages):
        page = doc.load_page(idx)
        if not _page_is_blank(page):
            if idx not in selected:
                selected.add(idx)
                first_indices.append(idx)
            if len(first_indices) >= num_first_pages:
                break

    for idx in range(n_pages - 1, -1, -1):
        if len(last_indices) >= num_last_pages:
            break
        if idx in selected:
            continue

        page = doc.load_page(idx)
        if not _page_is_blank(page):
            selected.add(idx)
            last_indices.append(idx)

    return first_indices + sorted(last_indices)


def load_pdf_metadata(pdf_path: str | Path) -> Dict[str, str | None]:
    """Load PDF metadata such as title, author, subject, etc."""
    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}

    return {
        "title": meta.get("title"),
        "author": meta.get("author"),
        "subject": meta.get("subject"),
        "keywords": meta.get("keywords"),
    }


def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """Convert a PyMuPDF Pixmap to a PIL Image."""
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img.convert("RGB") if mode == "RGBA" else img


def resize_for_vision(img: Image.Image, max_long_edge: int = 1200) -> Image.Image:
    """Resize image so the longest edge is <= ``max_long_edge`` while preserving aspect."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_long_edge:
        return img
    scale = max_long_edge / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def extract_page_text_with_fallback(
    pdf_path: str | Path,
    page_index: int,
    image_path: str | Path,
    lang: str = "eng",
) -> Dict[str, str]:
    """
    Extract text for a single page.

    Prefer native PDF text but fall back to running Tesseract on the rendered page image.
    """
    pdf_path = Path(pdf_path)
    image_path = Path(image_path)
    result: Dict[str, str] = {}

    with fitz.open(pdf_path) as doc:
        if page_index < 0 or page_index >= len(doc):
            return result
        page = doc.load_page(page_index)
        native_text = (page.get_text("text") or "").strip()
        if native_text:
            result["existing_ocr"] = native_text

    img = preprocess_for_ocr(Image.open(image_path))
    ocr_text = (pytesseract.image_to_string(img, lang=lang) or "").strip()
    if ocr_text:
        result["tesseract_ocr"] = ocr_text
    return result


def process_single_page_to_entry(
    doc: fitz.Document,
    pdf_path: Path,
    page_index: int,
    output_dir: Path,
    max_long_edge: int = 1200,
    lang: str = "eng",
) -> PdfPageEntry:
    """Render a single page, OCR it, and return a ``PdfPageEntry``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = fitz.Matrix(2.0, 2.0)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    pil_img = resize_for_vision(_pixmap_to_pil(pix), max_long_edge=max_long_edge)

    img_name = f"{pdf_path.stem}_page_{page_index + 1}.png"
    img_path = output_dir / img_name
    pil_img.save(img_path, format="PNG")

    text = extract_page_text_with_fallback(
        pdf_path=pdf_path,
        page_index=page_index,
        image_path=img_path,
        lang=lang,
    )

    return PdfPageEntry(
        page_number=page_index + 1,
        text=text,
        image_path=str(img_path),
    )


def build_combined_ocr_text(
    page_text_map: Dict[int, Dict[str, str]],
) -> str:
    """
    Build a combined OCR text blob for all pages and OCR approaches.

    The map should use zero-based page indices.
    """
    lines: list[str] = []

    for page_index in sorted(page_text_map.keys()):
        info = page_text_map[page_index]

        existing_key = None
        if "existing_ocr" in info:
            existing_key = "existing_ocr"
        elif "native_ocr" in info:
            existing_key = "native_ocr"

        page_number = page_index + 1

        if existing_key and info[existing_key].strip():
            lines.append(f"=== Page {page_number} - {existing_key} ===")
            lines.append(info[existing_key].strip())
            lines.append("")

        if "tesseract_ocr" in info and info["tesseract_ocr"].strip():
            lines.append(f"=== Page {page_number} - tesseract_ocr ===")
            lines.append(info["tesseract_ocr"].strip())
            lines.append("")

    return "\n".join(lines).strip()


def process_pdf_for_openai_inputs(
    pdf_path: str | Path,
    output_dir: str | Path,
    max_long_edge: int = 1200,
    lang: str = "eng",
) -> PdfProcessingResult:
    """
    High-level pipeline:

    1. Load metadata.
    2. Find first/last nonblank pages.
    3. Render, OCR, and package each page into ``PdfPageEntry``.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    metadata = load_pdf_metadata(pdf_path)
    pages: List[PdfPageEntry] = []

    with fitz.open(pdf_path) as doc:
        page_indices = _nonblank_sample_indices(
            doc, num_first_pages=3, num_last_pages=1
        )

        for idx in page_indices:
            pages.append(
                process_single_page_to_entry(
                    doc=doc,
                    pdf_path=pdf_path,
                    page_index=idx,
                    output_dir=output_dir,
                    max_long_edge=max_long_edge,
                    lang=lang,
                )
            )

    combined_text = build_combined_ocr_text(
        {idx: page.text for idx, page in zip(page_indices, pages)}
    )

    return PdfProcessingResult(
        pdf_path=str(pdf_path),
        metadata=metadata,
        pages=pages,
        combined_text=combined_text,
    )


if __name__ == "__main__":
    import tempfile

    pdf = "test_pdfs/bladesinthedark_v8_2.pdf"

    with tempfile.TemporaryDirectory(prefix="pdf_pages_") as tmpdir:
        pdf_result = process_pdf_for_openai_inputs(
            pdf_path=pdf,
            output_dir=tmpdir,
            max_long_edge=1200,
        )

        print("Metadata:", pdf_result.metadata)
        for p in pdf_result.pages:
            print(f"\nPage {p.page_number}")
            print("Image:", p.image_path)
            print("Text snippet:", p.text)
