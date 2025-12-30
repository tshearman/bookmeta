from dataclasses import dataclass
from typing import Callable, Iterable
from bookmetarefactor.types import Page, PageNumber, Pdf


type OcrMethodName = str
type OcrOutput = str | None


@dataclass(frozen=True)
class OcrMethod:
    name: OcrMethodName
    process: Callable[[Page], OcrOutput]


@dataclass(frozen=True)
class OcrResult:
    page: Page
    method_name: OcrMethodName
    output: OcrOutput


@dataclass(frozen=True)
class OcrTask:
    pdf: Pdf
    page_number: PageNumber
    method: OcrMethod


type OcrResults = list[OcrResult]
type OcrTasks = Iterable[OcrTask]
