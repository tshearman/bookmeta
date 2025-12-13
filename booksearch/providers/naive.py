from datamodel.book_info import BookInfo
from datamodel.book_info_response import BookInfoResponse


def book_search(resp: BookInfoResponse) -> BookInfo:
    return resp.info
