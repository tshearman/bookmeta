import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from bookmeta.types import Pdf
from bookmeta.types.ocr import OcrResults

type Provider = Literal["openai", "ollama"]
type LLMPayload = dict[str, Any]
type ClientConfig = dict[str, Any]


@dataclass(frozen=True)
class ContextLimits:
    num_first_images: int | None = 1
    num_last_images: int | None = 1
    num_first_ocr_pages: int | None = None
    num_last_ocr_pages: int | None = None


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    model: str
    client_config: ClientConfig


@dataclass(frozen=True)
class ExtractionConfig:
    prompt: str
    provider_config: ProviderConfig
    context_limits: ContextLimits = field(default_factory=ContextLimits)
    context_path: Path | None = None

    @property
    def hash(self) -> str:
        """Stable hash of OCR content and source page."""
        serialized = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractionTask:
    pdf: Pdf
    ocr_results: OcrResults
    config: ExtractionConfig


type ExtractionTasks = Iterable[ExtractionTask]


@dataclass(frozen=True)
class LLMTask:
    pdf: Pdf
    extraction: "ExtractionTask"
    payload: LLMPayload
    config: ProviderConfig


type LLMTaskBatch = list[LLMTask]
type LLMTaskBatches = Iterable[LLMTaskBatch]


@dataclass(frozen=True)
class PersistedBatch:
    path: Path
    digest: str
    pdfs: list[Pdf]
    pipeline_hash: str
    count: int
    submission: dict[str, Any] | None = None
