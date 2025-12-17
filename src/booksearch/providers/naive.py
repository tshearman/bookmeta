import json
from booksearch import BookSearchMethod
from bookinfo.book_info_response import BookInfoResponse


def book_search(resp: BookInfoResponse) -> str:
    return json.dumps(resp.info)


def naive_search() -> BookSearchMethod:
    return book_search
