from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Iterable

import joblib
import pandas as pd
from tqdm.auto import tqdm

from bookmeta.config.settings import DEFAULT_DB_PATH, NORMALIZE_CACHE_DIR
from bookmeta.data.sqlite import _compute_pdf_hash
from bookmeta.metadata.writer import MetadataHelper, embed_metadata, ensure_copy
from bookmeta.pipelines.normalization.keywords import (
    agglomerative_cluster_keywords,
    assign_canonical_keywords_per_cluster,
    generate_keyword_embeddings,
    get_keywords_by_pdf_hash,
    normalize_keyword,
)
from bookmeta.pipelines.normalization.publisher import normalize_publisher

LOGGER = logging.getLogger(__name__)
FAILURE_LOG_LOCK = Lock()
NORMALIZE_CACHE = joblib.Memory(NORMALIZE_CACHE_DIR, verbose=0)


def _flatten_keywords(keyword_sets: Iterable[Iterable[str]]) -> list[str]:
    unique: dict[str, None] = {}
    for group in keyword_sets:
        for keyword in group:
            if keyword not in unique:
                unique[keyword] = None
    return list(unique.keys())


@NORMALIZE_CACHE.cache
def _normalize_keywords_pipeline_cached(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    workers: int = 4,
    embedding_host: str = "http://192.168.1.31:11434",
    embedding_model: str = "snowflake-arctic-embed:335m",
    cluster_count: int | None = None,
    canonical_host: str = "http://192.168.1.31:11434",
    canonical_model: str = "qwen2.5vl:32b",
    canonical_temperature: float = 0.01,
) -> pd.DataFrame:
    """Load keywords from the DB, cluster them, and assign canonical labels."""

    LOGGER.info("Loading keyword metadata from %s", db_path)
    df = get_keywords_by_pdf_hash(db_path)
    LOGGER.info("Loaded %d rows", len(df))
    if df.empty:
        df["canonical_keywords"] = [[] for _ in range(len(df))]
        return df

    all_keywords = _flatten_keywords(df["keywords"])
    LOGGER.info("Discovered %d raw keywords", len(all_keywords))
    keyword_to_clean = {keyword: normalize_keyword(keyword) for keyword in all_keywords}
    cleaned_keywords = [kw for kw in dict.fromkeys(keyword_to_clean.values()) if kw]
    LOGGER.info("Normalized to %d unique keywords", len(cleaned_keywords))

    if not cleaned_keywords:
        LOGGER.warning("No keywords remained after normalization.")
        df["canonical_keywords"] = [
            sorted({kw for kw in keywords if kw}) for keywords in df["keywords"]
        ]
        return df

    embeddings = generate_keyword_embeddings(
        cleaned_keywords,
        workers=workers,
        host=embedding_host,
        model=embedding_model,
    )
    LOGGER.info("Generated %d embeddings", len(embeddings))

    clusters = agglomerative_cluster_keywords(
        embeddings,
        n_clusters=cluster_count,
    )
    LOGGER.info("Computed %d keyword clusters", clusters["cluster_id"].nunique())

    canonical_clusters = assign_canonical_keywords_per_cluster(
        clusters,
        ollama_host=canonical_host,
        ollama_model=canonical_model,
        temperature=canonical_temperature,
        workers=workers,
    )
    canonical_clusters["canonical_keyword"] = canonical_clusters[
        "canonical_keyword"
    ].fillna(canonical_clusters["keyword"])
    clean_to_canonical = dict(
        canonical_clusters[["keyword", "canonical_keyword"]].values.tolist()
    )
    LOGGER.info("Mapped %d keywords to canonical labels", len(clean_to_canonical))

    def _map_keywords(keywords: Iterable[str]) -> list[str]:
        canonical: list[str] = []
        for keyword in keywords:
            clean_kw = keyword_to_clean.get(keyword, keyword)
            canonical_kw = clean_to_canonical.get(clean_kw, clean_kw).lower()
            canonical.append(canonical_kw)
        return sorted(set(filter(None, canonical)))

    df = df.copy()
    df["canonical_keywords"] = df["keywords"].apply(_map_keywords)
    df["keywords"] = df["keywords"].apply(
        lambda keywords: sorted(keywords) if isinstance(keywords, set) else keywords
    )
    df["keywords"] = df["keywords"].apply(json.dumps)
    df["canonical_keywords"] = df["canonical_keywords"].apply(json.dumps)
    LOGGER.info("Assigned canonical keywords for all rows")

    return df


def normalize_keywords_pipeline(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    workers: int = 4,
    embedding_host: str = "http://192.168.1.31:11434",
    embedding_model: str = "snowflake-arctic-embed:335m",
    cluster_count: int | None = None,
    canonical_host: str = "http://192.168.1.31:11434",
    canonical_model: str = "qwen2.5vl:32b",
    canonical_temperature: float = 0.01,
    write_to_disk: bool = False,
) -> pd.DataFrame:
    df = _normalize_keywords_pipeline_cached(
        db_path=db_path,
        workers=workers,
        embedding_host=embedding_host,
        embedding_model=embedding_model,
        cluster_count=cluster_count,
        canonical_host=canonical_host,
        canonical_model=canonical_model,
        canonical_temperature=canonical_temperature,
    )

    if write_to_disk:
        path = Path(db_path)
        LOGGER.info("Writing normalized metadata to %s", path)
        with sqlite3.connect(path) as conn:
            df.to_sql("normalize_metadata", conn, if_exists="replace", index=False)

    return df


def _load_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [str(item) for item in data if item]
        except json.JSONDecodeError:
            return [value] if value else []
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


@dataclass
class BookMetadataEntity:
    authors: list[str]
    title: str | None
    subtitle: str | None
    publisher: str | None
    categories: list[str]
    tags: list[str]
    description: str | None
    isbn13: str | None
    isbn10: str | None


def row_to_book_metadata(row: pd.Series) -> BookMetadataEntity:
    keywords = _load_list(row.get("keywords"))
    canonical = _load_list(row.get("canonical_keywords"))
    tags = sorted({kw.lower() for kw in keywords + canonical if kw})

    author_value = row.get("author")
    authors = _load_list(author_value)
    if not authors and isinstance(author_value, str) and author_value.strip():
        authors = [author_value.strip()]
    elif authors:
        authors = [authors[0]]

    publisher_value = row.get("canonical_publisher") or row.get("publisher")
    publisher = normalize_publisher(publisher_value) if publisher_value else None

    isbn_candidates = _load_list(row.get("isbn_identifiers"))
    isbn_13 = None
    isbn_10 = None
    for candidate in isbn_candidates:
        digits = "".join(ch for ch in candidate if ch.isdigit())
        if len(digits) == 13 and isbn_13 is None:
            isbn_13 = digits
        elif len(digits) == 10 and isbn_10 is None:
            isbn_10 = digits

    out = BookMetadataEntity(
        authors=authors,
        title=row.get("title"),
        subtitle=row.get("subtitle"),
        publisher=publisher.title() if publisher else None,
        categories=[],
        tags=tags,
        description=row.get("description"),
        isbn13=isbn_13,
        isbn10=isbn_10,
    )
    return out


def attach_book_metadata_payloads(df: pd.DataFrame) -> pd.DataFrame:
    def _build(row: pd.Series) -> str:
        metadata = row_to_book_metadata(row)
        payload = {
            "authors": metadata.authors,
            "title": metadata.title,
            "subtitle": metadata.subtitle,
            "publisher": metadata.publisher,
            "categories": metadata.categories,
            "tags": metadata.tags,
            "description": metadata.description,
            "isbn_13": metadata.isbn13,
            "isbn_10": metadata.isbn10,
        }
        return json.dumps(payload, ensure_ascii=False)

    df = df.copy()
    df["book_metadata_payload"] = df.apply(_build, axis=1)
    return df


def dataframe_to_metadata_maps(
    df: pd.DataFrame,
) -> tuple[dict[str, BookMetadataEntity], dict[str, BookMetadataEntity]]:
    by_hash: dict[str, BookMetadataEntity] = {}
    by_name: dict[str, BookMetadataEntity] = {}
    for _, row in df.iterrows():
        entity = row_to_book_metadata(row)
        pdf_hash = str(row.get("pdf_hash") or "").strip()
        if pdf_hash:
            by_hash[pdf_hash] = entity
        pdf_name = row.get("pdf_name")
        if isinstance(pdf_name, str):
            name = pdf_name.strip()
            if name:
                by_name[name] = entity
    return by_hash, by_name


def _collect_pdfs(pdf_path: Path) -> list[Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF path not found: {pdf_path}")
    if pdf_path.is_file():
        return [pdf_path]
    return sorted(pdf_path.rglob("*.pdf"))


def _process_pdf(
    pdf_path: Path,
    metadata_by_hash: dict[str, BookMetadataEntity],
    metadata_by_name: dict[str, BookMetadataEntity],
    output_dir: Path,
    failure_log: Path,
    pdf_root: Path,
) -> None:
    try:
        relative = pdf_path.relative_to(pdf_root)
    except ValueError:
        relative = Path(pdf_path.name)

    target_dir = output_dir / relative.parent
    target_pdf = target_dir / relative.name
    source_resolved = pdf_path.resolve()
    target_resolved = target_pdf.resolve()
    if target_resolved == source_resolved:
        target_dir = output_dir / "__normalized__" / relative.parent
        target_pdf = target_dir / relative.name
        target_resolved = target_pdf.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_pdf.exists() and target_resolved != source_resolved:
        LOGGER.info("Skipping %s; output already exists at %s", pdf_path, target_pdf)
        return

    pdf_hash = _compute_pdf_hash(pdf_path)
    entity = metadata_by_hash.get(pdf_hash)

    LOGGER.info("Processing %s (hash=%s)", pdf_path, pdf_hash)
    LOGGER.info("Hash lookup result: %s", entity)
    if not entity:
        entity = metadata_by_name.get(pdf_path.name)
        LOGGER.info("Name lookup result: %s", entity)
    if not entity:
        LOGGER.warning("No metadata available for %s (hash=%s)", pdf_path, pdf_hash)
        return

    helper = MetadataHelper(asdict(entity))
    try:
        copied_pdf = ensure_copy(pdf_path, target_dir)
        embed_metadata(copied_pdf, helper)
        embed_metadata(copied_pdf, helper)
        LOGGER.info("Embedded metadata for %s into %s", pdf_path, copied_pdf)
    except Exception as exc:
        LOGGER.error("Failed processing %s: %s", pdf_path, exc)
        with FAILURE_LOG_LOCK:
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            with failure_log.open("a", encoding="utf-8") as fh:
                fh.write(f"{pdf_path}: {exc}\n")
        return


def run_metadata_writer(
    db_path: Path | str,
    pdf_path: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    n_clusters=750,
) -> None:
    LOGGER.info("Normalizing metadata from %s", db_path)
    df = normalize_keywords_pipeline(
        db_path=db_path, write_to_disk=False, cluster_count=n_clusters
    )
    df = attach_book_metadata_payloads(df)
    metadata_by_hash, metadata_by_name = dataframe_to_metadata_maps(df)
    LOGGER.info(
        "Built metadata maps with %d hash entries and %d name entries",
        len(metadata_by_hash),
        len(metadata_by_name),
    )

    pdfs = _collect_pdfs(pdf_path)
    LOGGER.info("Found %d PDF(s) to process", len(pdfs))
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_dir / "metadata_failures.log"
    failure_log.write_text("", encoding="utf-8")
    pdf_root = pdf_path if pdf_path.is_dir() else pdf_path.parent

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_pdf,
                pdf,
                metadata_by_hash,
                metadata_by_name,
                output_dir,
                failure_log,
                pdf_root,
            )
            for pdf in pdfs
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing PDFs",
            unit="pdf",
        ):
            future.result()


__all__ = [
    "normalize_keywords_pipeline",
    "row_to_book_metadata",
    "attach_book_metadata_payloads",
    "dataframe_to_metadata_maps",
    "run_metadata_writer",
    "BookMetadataEntity",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize PDF metadata and embed it directly into PDFs."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF file or directory.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory where metadata will be written."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the pipeline SQLite database.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for PDF processing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    logging.getLogger("httpx").setLevel(logging.INFO)

    run_metadata_writer(
        db_path=args.db_path,
        pdf_path=args.pdf_path.resolve(),
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
