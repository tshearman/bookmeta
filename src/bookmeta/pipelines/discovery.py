import logging
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Callable, Iterable, Iterator

from bookmeta.monitoring import TimedItem
from bookmeta.types import Pdf


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
    queue: Queue[TimedItem[Pdf]],
    *,
    dedupe: bool = True,
    limit: int | None = None,
    skip_processed: Callable[[Pdf], bool] | None = None,
) -> int:

    buckets: dict[str, list[Pdf]] = {}
    enqueued = 0

    for root in _iter_roots(roots):
        for path in discover_pdfs(root):
            if limit is not None and enqueued >= limit:
                return enqueued

            pdf = Pdf(path)

            if skip_processed and skip_processed(pdf):
                continue

            if dedupe:
                bucket = buckets.setdefault(pdf.fast_hash, [])

                if bucket:
                    if any(existing.hash == pdf.hash for existing in bucket):
                        continue
                bucket.append(pdf)

            queue.put(TimedItem(pdf, time.perf_counter()))
            enqueued += 1

    return enqueued


LOGGER = logging.getLogger(__name__)


def start_discovery(
    roots: Path | str | Iterable[Path | str],
    queue: Queue[TimedItem[Pdf]],
    *,
    dedupe: bool = True,
    limit: int | None = None,
    done_event: Event | None = None,
    skip_processed: Callable[[Pdf], bool] | None = None,
) -> Thread:
    """
    Launch PDF discovery in a background thread and signal completion on done_event.
    """

    def _produce() -> None:
        try:
            LOGGER.debug(
                "Starting PDF discovery: roots=%s dedupe=%s limit=%s",
                tuple(roots) if isinstance(roots, (list, tuple)) else roots,
                dedupe,
                limit,
            )
            count = produce_pdfs(
                roots,
                queue,
                dedupe=dedupe,
                limit=limit,
                skip_processed=skip_processed,
            )
            LOGGER.debug("PDF discovery enqueued %s PDFs", count)
        except Exception:
            LOGGER.exception("PDF discovery failed")
        finally:
            LOGGER.debug("PDF discovery stage signaling done")
            if done_event:
                done_event.set()

    thread = Thread(target=_produce, daemon=True)
    thread.start()
    return thread
