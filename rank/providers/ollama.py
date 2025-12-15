import logging
from typing import Sequence

import ollama

from bookinfo.providers.ollama import blocks_to_imgs
from bookinfo.book_info import DetailedBookInfo
from ocr.pdf_ocr_results import PdfOcrResults
from llm import cached_ollama_chat
from rank import (
    BookInfoSelectionPipeline,
    BookSearchCandidate,
    build_blocks_with_candidates,
)


def ollama_selection_runner(
    client: ollama.Client,
    model: str,
) -> BookInfoSelectionPipeline:
    def run(
        pdf_results: PdfOcrResults,
        candidates: Sequence[BookSearchCandidate],
    ) -> DetailedBookInfo | None:
        blocks, text_blocks = build_blocks_with_candidates(pdf_results, candidates)
        content = "\n\n".join(block["text"] for block in text_blocks)
        messages = [
            ollama.Message(
                role="user",
                content=content,
                images=blocks_to_imgs(blocks),  # type: ignore
            )
        ]
        response = cached_ollama_chat(
            model,
            messages,
            client,
            format=DetailedBookInfo.model_json_schema(),  # type: ignore[arg-type]
        )
        logging.info("\n\nOUTPUT FROM OLLAMA============================")
        logging.info(response["message"]["content"])
        logging.info("==================================================\n\n")
        return DetailedBookInfo.model_validate_json(response["message"]["content"])

    return run  # type: ignore
