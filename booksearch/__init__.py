from typing import Any, Callable
from datamodel.book_info import BookInfo
from datamodel.book_info_response import BookInfoResponse


BookSearchPipeline = Callable[[BookInfoResponse], list[BookInfo]]
