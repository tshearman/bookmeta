from typing import Iterator
import fitz
from bookmetarefactor.config import DEFAULT_NUM_FIRST_PAGES, DEFAULT_NUM_LAST_PAGES
from bookmetarefactor.ocr.methods import DEFAULT_OCR_METHODS
from bookmetarefactor.types import Pdf
from bookmetarefactor.types.ocr import *
from bookmetarefactor.utils.page import sample_page_indices


@dataclass(frozen=True)
class OcrConfig:
    num_first_pages: int = DEFAULT_NUM_FIRST_PAGES
    num_last_pages: int = DEFAULT_NUM_LAST_PAGES
    methods: Iterable[OcrMethod] = DEFAULT_OCR_METHODS


def execute_task(task: OcrTask) -> OcrResult:
    return OcrResult(task, task.method.process(task.page))


def ocr_tasks(pdf: Pdf, config: OcrConfig) -> Iterator[OcrTask]:

    with fitz.open(pdf.path) as doc:
        to_process = sample_page_indices(
            doc,
            config.num_first_pages,
            config.num_last_pages,
        )

    for idx in to_process:
        page_number = idx + 1
        for method in config.methods:
            with fitz.open(pdf.path) as doc:
                page = doc.load_page(page_number)
                yield OcrTask(page=page, page_number=page_number, method=method)
