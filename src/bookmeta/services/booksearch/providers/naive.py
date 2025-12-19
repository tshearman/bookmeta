import json

from bookmeta.services.bookinfo.book_info_response import BookInfoResponse
from bookmeta.services.booksearch import BookSearchMethod


def book_search(resp: BookInfoResponse) -> str:
    return json.dumps(resp.info)


def naive_search() -> BookSearchMethod:
    return book_search
