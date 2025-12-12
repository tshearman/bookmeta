import logging
from pathlib import Path
from typing import Sequence

from joblib import Memory
from openai import OpenAI
from pydantic.dataclasses import dataclass

from google_books_volume import GoogleBooksVolume
from pdf_processor import PdfProcessingResult
from book_info_extractor import image_path_to_data_url


CACHE_DIR = Path(".cache/openai_google_books")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)


@dataclass
class Rank:
    rank: int
    confidence: float


RANK_PROMPT = """
You are given images of a book (cover + sample interior pages), two OCR transcripts
("NATIVE OCR" and "TESSERACT OCR"), and up to five Google Books candidate entries.

Determine which candidate (if any) best matches the book shown in the images. Return:
  - rank: 1-based index of the best matching candidate (1-5). If none match, return -1.
  - confidence: a float between 0 and 1 indicating how certain you are in this match.

Use only the provided evidence. Prefer candidates whose title, author, publisher,
subject, ISBN, and description align with the visible text. Be conservative and choose
confidence near 0 for weak matches.
"""


def format_candidates(volumes: Sequence[GoogleBooksVolume]) -> str:
    lines: list[str] = ["CANDIDATE RESPONSES:"]
    for idx, volume in enumerate(volumes[:5], start=1):
        info = volume.volume_info
        title = (info.title if info and info.title else "UNKNOWN")
        authors = ", ".join(info.authors) if (info and info.authors) else "UNKNOWN"
        publisher = getattr(info, "publisher", None) or "UNKNOWN"
        categories = ", ".join(info.categories) if (info and info.categories) else "NONE"
        description = info.description if info and info.description else "NONE"
        identifiers = []
        if info:
            for ident in info.industry_identifiers:
                if ident.identifier:
                    label = f"{ident.type}:{ident.identifier}" if ident.type else ident.identifier
                    identifiers.append(label)
        lines.append(f"--- Candidate {idx} ---")
        lines.append(f"Title: {title}")
        lines.append(f"Authors: {authors}")
        lines.append(f"Publisher: {publisher}")
        lines.append(f"Categories: {categories}")
        lines.append(f"Description: {description}")
        lines.append(f"Identifiers: {', '.join(identifiers) if identifiers else 'NONE'}")
    return "\n".join(lines)


def construct_rank_content_blocks(
    pdf_result: PdfProcessingResult,
    volumes: Sequence[GoogleBooksVolume],
    context_path: str | None = None,
) -> list[dict[str, str]]:
    content_blocks: list[dict[str, str]] = [
        {"type": "input_text", "text": RANK_PROMPT},
        {"type": "input_text", "text": format_candidates(volumes)},
    ]

    if context_path:
        content_blocks.append(
            {
                "type": "input_text",
                "text": f"PDF CONTEXT\nRelative Path: {context_path}",
            }
        )

    for page in pdf_result.pages:
        content_blocks.append(
            {"type": "input_image", "image_url": image_path_to_data_url(page.image_path)}
        )

    native_text = pdf_result.combined_text_for("native_ocr")
    if native_text:
        content_blocks.append({"type": "input_text", "text": f"NATIVE OCR\n{native_text}"})

    tesseract_text = pdf_result.combined_text_for("tesseract_ocr")
    if tesseract_text:
        content_blocks.append(
            {"type": "input_text", "text": f"TESSERACT OCR\n{tesseract_text}"}
        )

    return content_blocks


def rank_google_books_candidates(
    pdf_result: PdfProcessingResult,
    volumes: Sequence[GoogleBooksVolume],
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    context_path: str | None = None,
) -> Rank:
    """
    Determine which Google Books candidate best matches the PDF evidence.
    """
    if not volumes:
        return Rank(rank=-1, confidence=0.0)

    content_blocks = construct_rank_content_blocks(
        pdf_result,
        volumes[:5],
        context_path=context_path,
    )
    logging.info("[Rank] Evaluating %d candidates with model '%s'", len(volumes), model)
    response = _cached_rank_request(content_blocks, model=model, client=client)
    logging.info("[Rank] Model selected rank=%s confidence=%.3f", response.rank, response.confidence)
    return response


@memory.cache(ignore=["client"])
def _cached_rank_request(content_blocks, model, client: OpenAI) -> Rank:
    logging.info("[Rank] Cache miss, calling OpenAI model=%s", model)
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
        text_format=Rank,
    )
    return response.output_parsed
