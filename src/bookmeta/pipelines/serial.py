from pathlib import Path

from bookmeta.extraction import execute_extraction_task, execute_llm_task
from bookmeta.ocr import OcrConfig, execute_task, ocr_tasks
from bookmeta.pipelines.metadata import write_metadata_with_cli
from bookmeta.types import Pdf
from bookmeta.types.bookinfo import DetailedBookInfoResult
from bookmeta.types.extraction import ExtractionConfig, ExtractionTask


def run_serial_pipeline(
    pdf_path: Path,
    *,
    ocr_config: OcrConfig,
    extraction_config: ExtractionConfig,
    writer_bin: Path | None = None,
    destination_pdf: Path | None = None,
) -> DetailedBookInfoResult | None:
    """
    Simple single-threaded pipeline:
    PDF -> OCR -> Extraction -> LLM -> Detailed -> return result
    """
    pdf = Pdf(pdf_path)
    ocr_results = [execute_task(task) for task in ocr_tasks(pdf, ocr_config)]
    extraction_task = ExtractionTask(pdf, ocr_results, extraction_config)
    llm_task = execute_extraction_task(extraction_task)
    bookinfo_result = execute_llm_task(llm_task)
    if bookinfo_result is None:
        return None
    detailed = bookinfo_result.bookinfo.info.as_detailed_book_info()
    detailed_result = DetailedBookInfoResult(pdf=pdf, detailed=detailed)

    if writer_bin and destination_pdf:
        write_metadata_with_cli(pdf.path, destination_pdf, detailed, writer_bin)

    return detailed_result
