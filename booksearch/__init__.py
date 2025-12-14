from typing import Callable
from datamodel.book_info_response import BookInfoResponse


BookSearchMethod = Callable[[BookInfoResponse], str | None]
