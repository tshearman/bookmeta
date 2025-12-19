from typing import Callable

from bookmeta.services.bookinfo.book_info_response import BookInfoResponse

BookSearchMethod = Callable[[BookInfoResponse], str | None]
