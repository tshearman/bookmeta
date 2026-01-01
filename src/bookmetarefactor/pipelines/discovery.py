from pathlib import Path
from queue import Queue
from typing import Iterable, Iterator

from bookmetarefactor.types import Pdf


def _iter_roots(roots: Path | str | Iterable[Path | str]) -> Iterator[Path]:
    """Normalize a single root or iterable of roots into Path instances."""
    if isinstance(roots, (str, Path)):
        yield Path(roots)
        return
    for root in roots:
        yield Path(root)


def discover_pdfs(root: Path) -> Iterator[Path]:
    yield from (path for path in root.rglob("*.pdf") if path.is_file())


def produce_pdfs(
    roots: Path | str | Iterable[Path | str],
    queue: Queue[Pdf],
    *,
    dedupe: bool = True,
    limit: int | None = None,
) -> int:

    seen_hashes: set[str] = set()
    enqueued = 0

    for root in _iter_roots(roots):
        for path in discover_pdfs(root):
            if limit is not None and enqueued >= limit:
                return enqueued

            pdf = Pdf(path)
            if dedupe:
                if pdf.hash in seen_hashes:
                    continue
                seen_hashes.add(pdf.hash)

            queue.put(pdf)
            enqueued += 1

    return enqueued
