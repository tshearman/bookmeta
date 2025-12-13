import ollama
from typing import Any
from bookinfo import BOOK_PROMPT, BookInfoRequestPipeline
from bookinfo.blocks import construct_blocks, get_img_blocks, get_text_blocks
from datamodel.book_info_response import BookInfoResponse
from datamodel.pdf_ocr_results import PdfOcrResults
from ocr.rendering import img_to_b64


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

    def run(input: PdfOcrResults) -> BookInfoResponse | None:
        blocks = construct_blocks(input, prompt=BOOK_PROMPT)
        messages = blocks_to_ollama_messages(blocks)
        response = client.chat(
            model=model,
            messages=messages,
            format=BookInfoResponse.model_json_schema(),
        )
        return BookInfoResponse.model_validate_json(response["message"]["content"])

    return run
