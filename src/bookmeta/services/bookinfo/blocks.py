from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookmeta.services.ocr.pdf_ocr_results import PdfOcrResults


@dataclass
class ContextLimits:
    num_first_images: int | None = None
    num_last_images: int | None = None
    num_first_ocr_pages: int | None = None
    num_last_ocr_pages: int | None = None


def construct_blocks(
    pdf_result: PdfOcrResults,
    prompt: str,
    context_path: Path | None = None,
    limits: ContextLimits | None = None,
) -> dict[str, Any]:

    limits = limits or ContextLimits()
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

    metadata_values = pdf_result.metadata
    metadata_lines = [
        f"{key.title()}: {value}" for key, value in metadata_values.items() if value
    ]
    if metadata_lines:
        blocks["metadata"] = {
            "type": "input_text",
            "text": "PDF METADATA\n" + "\n".join(metadata_lines),
        }

    for img in pdf_result.images(limits.num_first_images, limits.num_last_images):
        blocks["images"].append(
            {
                "type": "input_image",
                "image": img,
            }
        )

    for ocr in pdf_result.ocr_results(
        limits.num_first_ocr_pages, limits.num_last_ocr_pages
    ):
        text = ocr.ocr_result.text or ""
        blocks["ocr"].append(
            {
                "type": "input_text",
                "text": f"PAGE: {ocr.page_number} OCR USING METHOD: {ocr.ocr_result.method}\n{text}",
            }
        )

    return blocks


def get_text_blocks(blocks: dict[str, Any]) -> list[dict[str, Any]]:
    text_blocks = [blocks["prompt"], blocks["path"]]
    if "metadata" in blocks:
        text_blocks.append(blocks["metadata"])
    text_blocks.extend(blocks["ocr"])
    return text_blocks


def get_img_blocks(blocks: dict[str, Any]) -> list[dict[str, Any]]:
    return blocks["images"]
