from typing import Iterator
import fitz
from bookmetarefactor.ocr import OcrConfig
from bookmetarefactor.types import Pdf
from bookmetarefactor.types.ocr import *
from bookmetarefactor.utils.page import sample_page_indices


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
            yield OcrTask(pdf=pdf, page_number=page_number, method=method)
