import logging
import re
from typing import Any

import ollama
from pydantic import BaseModel

from bookmeta.services.bookinfo import BookInfoRequestPipeline
from bookmeta.services.bookinfo.blocks import (
    ContextLimits,
    construct_blocks,
    get_img_blocks,
    get_text_blocks,
)
from bookmeta.services.bookinfo.book_info import BookInfo
from bookmeta.services.bookinfo.book_info_confidence import BookInfoConfidence
from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.llm import cached_ollama_chat
from bookmeta.services.ocr.pdf_ocr_results import PdfOcrResults
from bookmeta.services.ocr.rendering import img_to_b64


class SimplifiedBookInfoSummary(BaseModel):
    title: str | None = None
    author: str | None = None
    keywords: list[str] | None = None
    description: str | None = None
    title_confidence: float | None = None
    author_confidence: float | None = None


OLLAMA_TITLE_AUTHOR_PROMPT = """
You are analyzing book cover/interior images and OCR excerpts.
Use only the visible/provided text. Never invent or infer facts that are not explicitly present.

Return ONLY a JSON object matching this schema:
{
    "title": "<book title or null>",
    "author": "<primary author or null>",
    "keywords": ["keyword1", "keyword2"] or null,
    "description": "<brief summary using visible text>" or null,
    "title_confidence": <0-1 confidence in the title>,
    "author_confidence": <0-1 confidence in the author>
}

Guidance:
- "keywords" Populate the keywords field with a list of strings, up to but no more
  than 8. These should be keywords that describe information about the book
  like genre, subject, if its a game, and other high-level metadata
- "description" a concise transcription/summary of the readable text you
  can infer from the images and OCR excerpts (do not just echo the provided OCR
  but use the provided OCR as context).
- If a field cannot be confirmed, set it to null and set the corresponding confidence near 0.
- Do not add any text outside the JSON. Do not wrap the JSON in code fences.
"""


def blocks_to_content(blocks: dict[str, Any]) -> str:
    return "\n\n".join(b["text"] for b in get_text_blocks(blocks))


def blocks_to_imgs(blocks: dict[str, Any]) -> list[ollama.Image]:
    return [ollama.Image(value=img_to_b64(b["image"])) for b in get_img_blocks(blocks)]


def blocks_to_ollama_messages(blocks: dict[str, Any]) -> list[ollama.Message]:
    return [
        ollama.Message(
            role="user",
            content=blocks_to_content(blocks),
            images=blocks_to_imgs(blocks),  # type: ignore
        )
    ]


def strip_code_fences(payload: str) -> str:
    """Remove Markdown code fences so JSON parsing succeeds."""
    text = payload.strip()
    if not text.startswith("```"):
        return text
    # Remove leading ```json and trailing ``` blocks.
    text = re.sub(
        r"^```(?:\w+)?\s*", "", text, count=1, flags=re.IGNORECASE | re.MULTILINE
    )
    text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def ollama_bookinfo_request(
    client: ollama.Client, model: str, prompt: str, context_limits: ContextLimits | None
) -> BookInfoRequestPipeline:

    def summary_to_response(summary: SimplifiedBookInfoSummary) -> BookInfoResponse:
        info = BookInfo(
            author=summary.author,
            title=summary.title,
            keywords=summary.keywords,
            description=summary.description,
        )
        confidence = BookInfoConfidence(
            author_confidence=summary.author_confidence or 0.0,
            title_confidence=summary.title_confidence or 0.0,
        )
        return BookInfoResponse(info=info, confidence=confidence)

    def run(input: PdfOcrResults) -> BookInfoResponse | None:
        blocks = construct_blocks(input, prompt=prompt, limits=context_limits)
        messages = blocks_to_ollama_messages(blocks)
        response = cached_ollama_chat(model, messages, client)
        response_content = response["message"]["content"]
        logging.info(f"\n\nOLLAMA RESPONSE:\n{response_content}\n\n")
        summary = SimplifiedBookInfoSummary.model_validate_json(
            strip_code_fences(response_content)
        )
        return summary_to_response(summary)

    return run  # type: ignore
