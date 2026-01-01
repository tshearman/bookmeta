from bookmeta.services.bookinfo.book_info import BookInfo, split_authors


def test_split_authors_handles_commas_and_and():
    text = "a, b, c and d"
    assert split_authors(text) == ["a", "b", "c", "d"]


def test_split_authors_handles_oxford_comma():
    text = "a, b, c, and d"
    assert split_authors(text) == ["a", "b", "c", "d"]


def test_as_detailed_book_info_returns_list_for_multiple_authors():
    info = BookInfo(author="a, b and c", title=None)
    assert info.as_detailed_book_info().author == ["a", "b", "c"]


def test_as_detailed_book_info_returns_string_for_single_author():
    info = BookInfo(author="Jane Doe", title=None)
    assert info.as_detailed_book_info().author == "Jane Doe"
