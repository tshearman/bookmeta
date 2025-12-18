from .book_info import BookInfo
from .book_info_confidence import BookInfoConfidence

from pydantic import BaseModel


class BookInfoResponse(BaseModel):
    info: BookInfo
    confidence: BookInfoConfidence
