from pathlib import Path
from typing import Any
from ocr.pdf_ocr_results import PdfOcrResults


def construct_blocks(
    pdf_result: PdfOcrResults,
    prompt: str,
    context_path: Path | None = None,
) -> dict[str, Any]:

    if context_path is None:
        context_path = Path(pdf_result.pdf_path)

    blocks = dict()
    blocks["prompt"] = {
        "type": "input_text",
        "text": prompt,
    }
    blocks["path"] = {
        "type": "input_text",
        "text": f"PDF FILE:\nRelative Path: {str(context_path)}\nFile Name: {context_path.stem}",
    }
    blocks["images"] = []
    blocks["ocr"] = []
    for page in pdf_result.ocr_results:
        blocks["images"].append(
            {
                "type": "input_image",
                "image": page.image,
            }
        )

        for ocr in page.ocr_results:
            blocks["ocr"].append(
                {
                    "type": "input_text",
                    "text": f"PAGE: {page.page_number} OCR USING METHOD: {ocr.method}\n{ocr.text}",
                }
            )

    return blocks


def get_text_blocks(blocks: dict[str, Any]) -> list[dict[str, Any]]:
    return [blocks["prompt"], blocks["path"]] + blocks["ocr"]


def get_img_blocks(blocks: dict[str, Any]) -> list[dict[str, Any]]:
    return blocks["images"]
