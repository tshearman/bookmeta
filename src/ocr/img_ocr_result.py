from pydantic.dataclasses import dataclass


@dataclass
class OcrResult:
    method: str
    text: str | None = None
