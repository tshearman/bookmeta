from itertools import islice
from typing import Iterable, List

import fitz


def page_is_blank(
    page: fitz.Page,
    white_threshold: int = 250,
    max_nonwhite_ratio: float = 0.005,
) -> bool:
    pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
    samples = pix.samples
    n_channels = pix.n
    total_pixels = pix.width * pix.height
    if total_pixels == 0:
        return True

    nonwhite = 0
    for idx in range(0, len(samples), n_channels):
        if any(
            channel < white_threshold for channel in samples[idx : idx + n_channels]
        ):
            nonwhite += 1
    return (nonwhite / total_pixels) <= max_nonwhite_ratio


def sample_page_indices(
    doc: fitz.Document,
    num_first_pages: int = 3,
    num_last_pages: int = 1,
) -> List[int]:
    n_pages = len(doc)
    if n_pages <= 0:
        return []

    def _non_blank_indices(indices: Iterable[int]) -> Iterable[int]:
        for idx in indices:
            if not page_is_blank(doc.load_page(idx)):
                yield idx

    first = set(islice(_non_blank_indices(range(n_pages)), num_first_pages))
    last = set(islice(_non_blank_indices(range(n_pages - 1, -1, -1)), num_last_pages))
    return sorted(first.union(last))
