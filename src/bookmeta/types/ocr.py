from dataclasses import dataclass
from typing import Callable, Iterable

from bookmeta.types import Page, PageNumber

type OcrMethodName = str
type OcrOutput = str | None


@dataclass(frozen=True)
class OcrMethod:
    name: OcrMethodName
    process: Callable[[Page], OcrOutput]


@dataclass(frozen=True)
class OcrTask:
    page: Page
    page_number: PageNumber
    method: OcrMethod


@dataclass(frozen=True)
class OcrResult:
    task: OcrTask
    output: OcrOutput


type OcrResults = list[OcrResult]
type OcrTasks = Iterable[OcrTask]
