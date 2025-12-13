import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, cast
import ollama
from openai import OpenAI
from pydantic.dataclasses import dataclass
import fitz

from bookinfo import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL
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


if __name__ == "__main__":
    sample_pdf = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "test_pdfs"
        / "bladesinthedark_v8_2.pdf"
    )

    if not sample_pdf.exists():
        raise FileNotFoundError(f"Sample PDF not found: {sample_pdf}")

    ollama_client = ollama.Client(host=DEFAULT_OLLAMA_HOST)
    with open("secrets.json", "r") as f:
        secrets = json.load(f)
    OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
    OPENAI_PROJECT_ID = secrets["OPENAI_PROJECT_ID"]
    openai_client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)

    methods = (
        native_ocr_method,
        tesseract_ocr_method,
        ollama_ocr_method(client=ollama_client, model=DEFAULT_OLLAMA_MODEL),
        ollama_ocr_method(client=ollama_client, model="qwen3-vl:32b"),
        # openai_ocr_method(client=openai_client, model=DEFAULT_OPENAI_MODEL),
    )
    config = OcrPipelineConfig(ocr_methods=methods)
    pipeline = generate_pipeline(config)
    results = pipeline(sample_pdf)

    combined_len = len(results.combined_text or "")
    sampled_pages = len(results.ocr_results)
    print(
        f"OCR complete for {sample_pdf.name}: "
        f"{sampled_pages} sampled pages, {combined_len} characters of text."
    )

    print("\nMetadata")
    if results.metadata:
        for key, value in results.metadata.items():
            print(f"  {key}: {value or '<empty>'}")
    else:
        print("  <none>")

    print("\nSampled page summaries")
    for page in results.ocr_results:
        print(f"  Page {page.page_number}")
        if not page.ocr_results:
            print("    <no OCR results>")
            continue

        for ocr_result in page.ocr_results:
            snippet = (ocr_result.text or "").strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            snippet = snippet or "<no text>"
            print(f"    {ocr_result.method}: {snippet}")
