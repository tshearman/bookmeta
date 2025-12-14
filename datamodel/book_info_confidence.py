from pydantic.dataclasses import dataclass


@dataclass
class BookInfoConfidence:
    author_confidence: float
    title_confidence: float
    # publisher_confidence: float
    # isbn_confidence: float = 0.0
