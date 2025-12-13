from pathlib import Path
import ollama
from bookinfo.blocks import construct_blocks
from bookinfo.providers.ollama import blocks_to_imgs
from datamodel.book_info import BookInfo
from datamodel.pdf_ocr_results import PdfOcrResults
from rank import RANK_PROMPT, BookInfoRankPipeline, ScoredBookInfo, get_text_blocks


def ollama_bookinfo_rank(client: ollama.Client, model: str) -> BookInfoRankPipeline:

    def construct_rank_content_blocks(
        ocr_results: PdfOcrResults,
        candidates: list[BookInfo],
        context_path: Path | None,
    ) -> list[ollama.Message]:

        blocks = construct_blocks(ocr_results, RANK_PROMPT, context_path)
        content = "\n\n".join(b["text"] for b in get_text_blocks(blocks, candidates))
        return [
            ollama.Message(
                role="user",
                content=content,
                images=blocks_to_imgs(blocks),  # type: ignore
            )
        ]

    def run(
        ocr_results: PdfOcrResults,
        candidates: list[BookInfo],
        context_path: Path | None = None,
    ) -> ScoredBookInfo | None:
        messages = construct_rank_content_blocks(ocr_results, candidates, context_path)
        response = client.chat(
            model=model,
            messages=messages,
            format=ScoredBookInfo.model_json_schema(),  # type: ignore
        )
        return ScoredBookInfo.model_validate_json(response["message"]["content"])

    return run
