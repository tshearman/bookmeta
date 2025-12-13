from typing import Any, Callable
from datamodel.book_info import BookInfo
from datamodel.pdf_ocr_results import PdfOcrResults
from pydantic import BaseModel


class ScoredBookInfo(BaseModel):
    bookinfo: BookInfo
    score: float


BookInfoRankPipeline = Callable[[PdfOcrResults, list[BookInfo]], ScoredBookInfo | None]


RANK_PROMPT = """
You are given images of a book (cover + sample interior pages), multiple OCR transcripts, 
and up to five candidate book entries.

Determine which candidate (if any) best matches the book shown in the images. Return:
  - rank: 1-based index of the best matching candidate (1-5). If none match, return -1.
  - confidence: a float between 0 and 1 indicating how certain you are in this match.

Use only the provided evidence. Prefer candidates whose title, author, publisher,
subject, ISBN, and description align with the visible text. Be conservative and choose
confidence near 0 for weak matches.
"""


def get_text_blocks(blocks: dict[str, Any], candidates: list[BookInfo]):
    content_blocks = [blocks["prompt"]]
    content_blocks.append(blocks["path"])
    content_blocks.extend(
        {"type": "input_text", "text": f"CANDIDATE BOOK\n {c.model_dump_json()}"} for c in candidates  # type: ignore
    )
    content_blocks.extend(blocks["ocr"])
    return content_blocks
