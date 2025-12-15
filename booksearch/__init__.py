from typing import Callable
from bookinfo.book_info_response import BookInfoResponse


BookSearchMethod = Callable[[BookInfoResponse], str | None]
