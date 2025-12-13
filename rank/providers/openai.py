from pathlib import Path
import openai
from bookinfo.blocks import construct_blocks, get_img_blocks
from bookinfo.providers.openai import parse_img_block
from datamodel.book_info import BookInfo
from datamodel.pdf_ocr_results import PdfOcrResults
from rank import RANK_PROMPT, BookInfoRankPipeline, ScoredBookInfo
from typing import List


def openai_bookinfo_rank(client: openai.OpenAI, model: str) -> BookInfoRankPipeline:

    def construct_rank_content_blocks(
        ocr_results: PdfOcrResults,
        candidates: list[BookInfo],
        context_path: Path | None,
    ) -> list[dict]:
        blocks = construct_blocks(ocr_results, RANK_PROMPT, context_path)
        content_blocks = [blocks["prompt"]]
        content_blocks.append(blocks["path"])
        content_blocks.extend(
            {"type": "input_text", "text": f"CANDIDATE BOOK\n {c.model_dump_json()}"} for c in candidates  # type: ignore
        )
        content_blocks.extend([parse_img_block(b) for b in get_img_blocks(blocks)])
        content_blocks.extend(blocks["ocr"])
        return content_blocks

    def run(
        ocr_results: PdfOcrResults,
        candidates: list[BookInfo],
        context_path: Path | None = None,
    ) -> ScoredBookInfo | None:
        context = construct_rank_content_blocks(ocr_results, candidates, context_path)
        return client.responses.parse(
            model=model, input=context, text_format=ScoredBookInfo  # type: ignore
        ).output_parsed

    return run
