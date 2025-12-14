import logging
import ollama
from typing import Any
from pydantic import BaseModel
from bookinfo import BookInfoRequestPipeline
from bookinfo.blocks import construct_blocks, get_img_blocks, get_text_blocks
from datamodel.book_info import BookInfo
from datamodel.book_info_confidence import BookInfoConfidence
from datamodel.book_info_response import BookInfoResponse
from datamodel.pdf_ocr_results import PdfOcrResults
from llm import cached_ollama_chat
from ocr.rendering import img_to_b64


class SimplifiedBookInfoSummary(BaseModel):
    title: str | None = None
    author: str | None = None
    keywords: list[str] | None = None
    description: str | None = None
    title_confidence: float | None = None
    author_confidence: float | None = None


OLLAMA_TITLE_AUTHOR_PROMPT = """
You are analyzing images of a book's cover/interior plus OCR text excerpts.

Return ONLY a JSON object matching this schema:
{
    "title": "<book title or null>",
    "author": "<primary author or null>",
    "keywords": ["keyword1", "keyword2"] or null,
    "description": "<brief summary using visible text>" or null,
    "title_confidence": <0-1 confidence in the title>,
    "author_confidence": <0-1 confidence in the author>
}

If you cannot see a field, set it to null with confidence near 0. Do not add text outside the JSON.
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


def ollama_bookinfo_request(
    client: ollama.Client, model: str
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
        blocks = construct_blocks(input, prompt=OLLAMA_TITLE_AUTHOR_PROMPT)
        messages = blocks_to_ollama_messages(blocks)
        response = cached_ollama_chat(model, messages, client)
        response_content = response["message"]["content"]
        logging.info(f"\n\nOLLAMA RESPONSE:\n{response_content}\n\n")
        summary = SimplifiedBookInfoSummary.model_validate_json(response_content)
        return summary_to_response(summary)

    return run
