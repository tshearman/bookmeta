from pydantic.dataclasses import dataclass


@dataclass
class BookInfoConfidence:
    author_confidence: float
    title_confidence: float
