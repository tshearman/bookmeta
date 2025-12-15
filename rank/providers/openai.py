from typing import Sequence
import openai
from bookinfo.blocks import get_img_blocks
from bookinfo.providers.openai import parse_img_block
from datamodel.book_info import DetailedBookInfo
from datamodel.pdf_ocr_results import PdfOcrResults
from llm import cached_openapi_response_parsed
from rank import (
    BookInfoSelectionPipeline,
    BookSearchCandidate,
    build_blocks_with_candidates,
)


def openai_selection_runner(
    client: openai.OpenAI, model: str
) -> BookInfoSelectionPipeline:
    def run(
        pdf_results: PdfOcrResults,
        candidates: Sequence[BookSearchCandidate],
    ) -> DetailedBookInfo | None:
        blocks, text_blocks = build_blocks_with_candidates(pdf_results, candidates)
        img_blocks = [parse_img_block(b) for b in get_img_blocks(blocks)]
        context = [{"role": "user", "content": text_blocks + img_blocks}]
        return cached_openapi_response_parsed(
            model,
            context,
            client,
            text_format=DetailedBookInfo,
        )

    return run  # type: ignore
