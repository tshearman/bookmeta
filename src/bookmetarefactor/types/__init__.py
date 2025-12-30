from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
from bookmetarefactor.utils import hash_file

type PdfHash = str
type PageNumber = int
type Page = fitz.Page
type JsonDict = dict[str, Any]


@dataclass(frozen=True)
class Pdf:
    path: Path
    hash: PdfHash = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hash", hash_file(self.path))


@dataclass(frozen=True)
class RequestBatch:
    """Files that map PDFs to OpenAI batch request JSONL payloads."""

    request_path: Path
    mapping_path: Path
    count: int
    hashes: list[str]
    request_hash: str
    mapping_hash: str


@dataclass(frozen=True)
class SubmittedBatch:
    """OpenAI batch submission metadata."""

    request_path: Path
    mapping_path: Path
    response_path: Path
    batch_obj: JsonDict


@dataclass(frozen=True)
class CompletedBatch:
    """Completed batch ready for parsing and persistence."""

    batch_id: str
    mapping_path: Path
    output_path: Path
    batch_obj: JsonDict


@dataclass(frozen=True)
class ParsedResult:
    """Structured extraction from a completed batch entry."""

    pdf_path: Path
    pdf_hash: str
    detailed: JsonDict
