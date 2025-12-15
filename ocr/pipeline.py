import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, cast
import ollama
from openai import OpenAI
from pydantic.dataclasses import dataclass
import fitz

from bookinfo import DEFAULT_OPENAI_MODEL
from datamodel.img_ocr_result import OcrResult
from datamodel.pdf_ocr_results import OCRedPage, PdfOcrResults, build_combined_ocr_text
from ocr.metadata import load_pdf_metadata
from ocr.ocr import (
    OcrMethod,
    native_ocr_method,
    ollama_ocr_method,
    openai_ocr_method,
    tesseract_ocr_method,
)
from ocr.rendering import page_to_image
from ocr.sampling import sample_page_indices


LOGGER = logging.getLogger("ocr.pipeline")


def _log_info(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info("[OCRPipeline %s] %s", timestamp, message)


OcrPipeline = Callable[[str | Path], PdfOcrResults]
DEFAULT_OCR_METHODS: tuple[OcrMethod, ...] = (
    native_ocr_method,
    tesseract_ocr_method,
)


@dataclass
class OcrPipelineConfig:
    num_first_pages: int = 3
    num_last_pages: int = 1
    ocr_methods: Sequence[OcrMethod] = DEFAULT_OCR_METHODS


def process_page(
    doc: fitz.Document, ocr_methods: Sequence[OcrMethod]
) -> Callable[[int], OCRedPage]:
    def __inner__(idx: int) -> OCRedPage:
        page_number = idx + 1
        page = doc.load_page(idx)
        img = page_to_image(page, grayscale=True, max_long_edge=1200)
        ocr_results: list[OcrResult] = []
        for method in ocr_methods:
            result = method(page)
            if result:
                ocr_results.append(result)
        return OCRedPage(page_number, img, ocr_results)

    return __inner__


def pdf_ocr_pipeline(
    pdf_path: str | Path,
    num_first_pages: int = 3,
    num_last_pages: int = 1,
    ocr_methods: Sequence[OcrMethod] | None = None,
) -> PdfOcrResults:
    source_path = Path(pdf_path)
    metadata = load_pdf_metadata(source_path)
    ocr_results: list[OCRedPage] = []

    with fitz.open(source_path) as doc:
        page_indices = sample_page_indices(doc, num_first_pages, num_last_pages)
        methods = tuple(ocr_methods) if ocr_methods else DEFAULT_OCR_METHODS
        ocr_results = list(map(process_page(doc, methods), page_indices))
    combined_text = build_combined_ocr_text(ocr_results)

    return PdfOcrResults(pdf_path, combined_text, metadata, ocr_results)  # type: ignore


def generate_pipeline(config: OcrPipelineConfig) -> OcrPipeline:
    def run(pdf_path: str | Path) -> PdfOcrResults:
        return pdf_ocr_pipeline(
            pdf_path=pdf_path,
            num_first_pages=config.num_first_pages,
            num_last_pages=config.num_last_pages,
            ocr_methods=config.ocr_methods,
        )

    return run
