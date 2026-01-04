from dataclasses import dataclass
from typing import Iterable, Iterator

import fitz

from bookmeta.config import DEFAULT_NUM_FIRST_PAGES, DEFAULT_NUM_LAST_PAGES
from bookmeta.ocr.methods import DEFAULT_OCR_METHODS
from bookmeta.types import Pdf
from bookmeta.types.ocr import OcrMethod, OcrResult, OcrResults, OcrTask
from bookmeta.utils.page import sample_page_indices


@dataclass(frozen=True)
class OcrConfig:
    num_first_pages: int = DEFAULT_NUM_FIRST_PAGES
    num_last_pages: int = DEFAULT_NUM_LAST_PAGES
    methods: Iterable[OcrMethod] = DEFAULT_OCR_METHODS


def execute_task(task: OcrTask) -> OcrResult:
    return OcrResult(task, task.method.process(task.page))


def execute_ocr(pdf: Pdf, config: OcrConfig) -> OcrResults:
    return sorted(
        [execute_task(task) for task in ocr_tasks(pdf, config)],
        key=lambda t: t.task.page_number,
    )


def ocr_tasks(pdf: Pdf, config: OcrConfig) -> Iterator[OcrTask]:
    doc = fitz.open(pdf.path)
    to_process = sample_page_indices(
        doc,
        config.num_first_pages,
        config.num_last_pages,
    )

    for idx in to_process:
        page_number = idx + 1  # human-friendly 1-based index
        for method in config.methods:
            page = doc.load_page(idx)
            yield OcrTask(page=page, page_number=page_number, method=method)


__all__ = [
    "DEFAULT_NUM_FIRST_PAGES",
    "DEFAULT_NUM_LAST_PAGES",
    "DEFAULT_OCR_METHODS",
    "OcrConfig",
    "OcrResult",
    "OcrTask",
    "OcrMethod",
    "OcrResults",
    "execute_task",
    "ocr_tasks",
]
