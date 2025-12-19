from pydantic import BaseModel

from .book_info import BookInfo
from .book_info_confidence import BookInfoConfidence


class BookInfoResponse(BaseModel):
    info: BookInfo
    confidence: BookInfoConfidence
