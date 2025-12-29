from typing import Any

from openai import OpenAI

from bookmeta.services.bookinfo import BOOK_PROMPT, BookInfoRequestPipeline
from bookmeta.services.bookinfo.blocks import (
    ContextLimits,
    construct_blocks,
    get_img_blocks,
    get_text_blocks,
)
from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.llm import cached_openapi_response_parsed
from bookmeta.services.ocr.pdf_ocr_results import PdfOcrResults
from bookmeta.services.ocr.rendering import img_to_url


def parse_img_block(block: dict[str, Any]) -> dict[str, Any]:
    return {"type": block["type"], "image_url": img_to_url(block["image"])}


def blocks_to_openai_context(blocks: dict[str, Any]):
    text_blocks = get_text_blocks(blocks)
    img_blocks = [parse_img_block(b) for b in get_img_blocks(blocks)]
    all_blocks = text_blocks + img_blocks
    return [{"role": "user", "content": all_blocks}]


def openai_bookinfo_request(
    client: OpenAI, model: str, prompt: str, context_limits: ContextLimits | None
) -> BookInfoRequestPipeline:

    def run(input: PdfOcrResults) -> BookInfoResponse | None:
        blocks = construct_blocks(input, prompt, limits=context_limits)
        context = blocks_to_openai_context(blocks)
        return cached_openapi_response_parsed(
            model, context, client, text_format=BookInfoResponse
        )

    return run  # type: ignore
