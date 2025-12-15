import json
from typing import Any, Callable, Sequence, Union

from booksearch.pipeline import BookSearchResults
from datamodel.book_info import BookInfo, DetailedBookInfo
from datamodel.pdf_ocr_results import PdfOcrResults
from bookinfo.blocks import (
    construct_blocks,
    get_text_blocks as bookinfo_text_blocks,
)

BOOK_SELECTION_PROMPT = """
You are a meticulous bibliographic assistant. You are given:
  • Images of a book (cover + sample interior pages)
  • OCR text transcripts derived from those pages
  • Metadata snippets from book-search services (may include partial titles, authors, ISBN, etc.)

Using ONLY this evidence, produce the most accurate, self-consistent `BookInfo` JSON object you can.
Include the following fields when confidently supported by the data:
  - title
  - author (primary author)
  - subtitle (if explicitly visible or clearly implied)
  - publisher (as printed on the book or in metadata corroborated by the images/OCR)
  - subject: a single word or short phrase indicating genre/category
  - keywords: concise descriptive tags drawn from the visible content
  - isbn_identifiers: list of 10- or 13-digit ISBNs only if clearly present
  - description: a paragraph summarizing the book's content or blurb based on the visible text

If any field is uncertain or unsubstantiated, leave it null or empty; never hallucinate details.
Ensure the returned JSON strictly matches the `BookInfo` schema.
"""

BookSearchCandidate = Union[BookInfo, dict[str, Any], str]
BookInfoSelectionPipeline = Callable[
    [PdfOcrResults, BookSearchResults], DetailedBookInfo
]


def serialize_candidate(candidate: BookSearchCandidate) -> str:
    if isinstance(candidate, BookInfo):
        return candidate.model_dump_json(indent=2)
    if isinstance(candidate, dict):
        return json.dumps(candidate, indent=2, sort_keys=True)
    if isinstance(candidate, str):
        return candidate
    return repr(candidate)


def format_candidates(candidates: Sequence[BookSearchCandidate]) -> str:
    if not candidates:
        return "No book search candidates were provided."
    entries = []
    for idx, candidate in enumerate(candidates, start=1):
        serialized = serialize_candidate(candidate)
        entries.append(f"CANDIDATE {idx}:\n{serialized}")
    return "\n\n".join(entries)


def build_blocks_with_candidates(
    pdf_results: PdfOcrResults,
    candidates: Sequence[BookSearchCandidate],
):
    blocks = construct_blocks(pdf_results, BOOK_SELECTION_PROMPT)
    text_blocks = bookinfo_text_blocks(blocks)
    text_blocks.append(
        {
            "type": "input_text",
            "text": "BOOK SEARCH CANDIDATES\n" + format_candidates(candidates),
        }
    )
    return blocks, text_blocks
