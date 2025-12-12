import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, List

from joblib import Memory
from openai import OpenAI

from data_store import compute_pdf_hash
from google_books import fetch_google_books, GoogleBooksQuery
from google_books_volume import GoogleBooksVolume
from openai_rank_request import rank_google_books_candidates, Rank
from book_info_extractor import (
    BookInfo,
    bookinfo_to_google_books_query,
    extract_bookinfo_via_model,
)
from pdf_processor import process_pdf_for_openai_inputs


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_CACHE_DIR = BASE_DIR / ".cache/bookmeta"
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_MEMORY = Memory(location=str(PIPELINE_CACHE_DIR), verbose=0)


@dataclass
class PipelineResult:
    pdf_path: Path
    pdf_hash: str
    model: str
    provider: str
    book_info: Optional[BookInfo]
    query: Optional[GoogleBooksQuery]
    volumes: List[GoogleBooksVolume]
    ranking: Optional[Rank]
    selected_volume: Optional[GoogleBooksVolume]

    def serialize(self) -> dict:
        return {
            "pdf_path": str(self.pdf_path),
            "pdf_hash": self.pdf_hash,
            "model": self.model,
            "provider": self.provider,
            "book_info": _bookinfo_dict(self.book_info),
            "query": self.query.model_dump() if self.query else None,
            "volumes": [vol.raw if vol.raw else {} for vol in self.volumes],
            "ranking": (
                {
                    "rank": self.ranking.rank,
                    "confidence": self.ranking.confidence,
                }
                if self.ranking
                else None
            ),
            "selected_volume": (
                self.selected_volume.raw
                if (self.selected_volume and self.selected_volume.raw)
                else None
            ),
        }


def _bookinfo_dict(book: Optional[BookInfo]) -> Optional[dict]:
    if book is None:
        return None
    return {field.name: getattr(book, field.name, None) for field in fields(BookInfo)}


@PIPELINE_MEMORY.cache(ignore=["client", "google_books_api_key"])
def run_pipeline(
    pdf_path: Path,
    model: str,
    client: OpenAI | None,
    google_books_api_key: str,
    base_dir: Path | None = None,
    provider: str = "openai",
) -> PipelineResult:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logging.info(
        "Starting pipeline for %s using model %s via %s", pdf_path, model, provider
    )

    pdf_hash = compute_pdf_hash(pdf_path)
    book_info: Optional[BookInfo] = None
    query: Optional[GoogleBooksQuery] = None
    volumes: List[GoogleBooksVolume] = []
    ranking: Optional[Rank] = None
    selected_volume: Optional[GoogleBooksVolume] = None

    with tempfile.TemporaryDirectory(prefix="pdf_pages_") as tmpdir:
        logging.info("Processing PDF with temporary output dir %s", tmpdir)
        pdf_result = process_pdf_for_openai_inputs(
            pdf_path=pdf_path,
            output_dir=tmpdir,
            max_long_edge=1200,
        )

        logging.info("Calling OpenAI to extract BookInfo")
        relative_context = None
        if base_dir:
            try:
                relative_context = str(
                    Path(pdf_path).resolve().relative_to(base_dir.resolve())
                )
            except ValueError:
                relative_context = str(Path(pdf_path).resolve())
        else:
            relative_context = str(Path(pdf_path).resolve())

        book_info = extract_bookinfo_via_model(
            pdf_result=pdf_result,
            client=client,
            model=model,
            context_path=relative_context,
            provider=provider,
        )
        query = bookinfo_to_google_books_query(book_info)

        logging.info("Fetching Google Books candidates")
        volumes = fetch_google_books(query, key=google_books_api_key)
        logging.info("Fetched %d Google Books candidates", len(volumes))

        ranking = rank_google_books_candidates(
            pdf_result=pdf_result,
            volumes=volumes,
            client=client,
            model=model,
            context_path=relative_context,
            provider=provider,
        )

    logging.info("Pipeline complete")

    if book_info:
        logging.info("Extracted BookInfo: %s", book_info)
    if query:
        logging.info("Derived GoogleBooksQuery: %s", query.model_dump())

    serialized_volumes = [vol.raw if vol.raw else {} for vol in volumes]
    logging.debug(
        "Raw Google Books responses: %s", json.dumps(serialized_volumes, indent=2)
    )

    if ranking is not None and ranking.rank > 0:
        ranking_payload = {"rank": ranking.rank, "confidence": ranking.confidence}
        logging.info("Ranking result: %s", ranking_payload)
        if 0 < ranking.rank <= len(volumes):
            selected_volume = volumes[ranking.rank - 1]
            logging.info(
                "Top candidate summary: title=%s authors=%s",
                (
                    selected_volume.volume_info.title
                    if selected_volume.volume_info
                    else "UNKNOWN"
                ),
                (
                    ", ".join(selected_volume.volume_info.authors)
                    if selected_volume.volume_info
                    and selected_volume.volume_info.authors
                    else "UNKNOWN"
                ),
            )
    else:
        logging.info("No ranking result available")

    if book_info and query:
        logging.info("Returning pipeline result for hash %s", pdf_hash)

    return PipelineResult(
        pdf_path=pdf_path,
        pdf_hash=pdf_hash,
        model=model,
        provider=provider,
        book_info=book_info,
        query=query,
        volumes=volumes,
        ranking=ranking,
        selected_volume=selected_volume,
    )


def invalidate_pipeline_cache_entry(
    pdf_path: Path,
    model: str,
    client: OpenAI | None,
    google_books_api_key: str,
    base_dir: Path | None = None,
    provider: str = "openai",
) -> bool:
    """
    Remove the cached result for a specific pipeline call.
    Returns True if a cache entry was removed.
    """
    args_id = run_pipeline._get_args_id(
        pdf_path, model, client, google_books_api_key, base_dir, provider
    )
    func_path = Path(PIPELINE_MEMORY.location) / "joblib" / Path(run_pipeline.func_id)
    target = func_path / args_id
    if target.exists():
        shutil.rmtree(target)
        return True
    return False
