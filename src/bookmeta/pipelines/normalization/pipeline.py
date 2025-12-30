import argparse
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from bookmeta.services.bookinfo import Provider
import joblib
import pandas as pd
from tqdm.auto import tqdm

from bookmeta.cli.batch import _count_pdfs_with_ripgrep
from bookmeta.cli.pipeline import _client_config_for, _read_secrets
from bookmeta.cli.utils import discover_pdfs
from bookmeta.config.settings import DEFAULT_DB_PATH, NORMALIZE_CACHE_DIR
from bookmeta.data.sqlite import _compute_pdf_hash
from bookmeta.pipelines.normalization.keywords import (
    agglomerative_cluster_keywords,
    assign_canonical_keywords_per_cluster,
    generate_keyword_embeddings,
    get_keywords_by_pdf_hash,
    normalize_keyword,
)
from bookmeta.pipelines.normalization.publisher import (
    attach_canonical_publishers,
    normalize_publisher,
)

LOGGER = logging.getLogger(__name__)
FAILURE_LOG_LOCK = Lock()
NORMALIZE_CACHE = joblib.Memory(NORMALIZE_CACHE_DIR, verbose=0)
WRITER_ENV_VAR = "BOOKMETA_WRITER_BIN"
WRITER_FALLBACK = Path(
    "tools/java/pdf-metadata-writer-cli/build/install/pdf-metadata-writer-cli/bin/pdf-metadata-writer-cli"
)


def _writer_binary(cli_path: Path | None = None) -> Path:
    if cli_path is not None:
        return cli_path
    override = os.environ.get(WRITER_ENV_VAR)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(WRITER_FALLBACK.resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Path to pdf-metadata-writer CLI not provided. "
        "Pass --writer-bin, set BOOKMETA_WRITER_BIN, or run "
        "tools/java/gradlew :pdf-metadata-writer-cli:installDist."
    )


def _flatten_keywords(keyword_sets: Iterable[Iterable[str]]) -> list[str]:
    uniques = set([])
    for group in keyword_sets:
        for keyword in group:
            if keyword not in uniques:
                uniques.add(keyword)
    return sorted(uniques)


def _collect_cleaned_keywords(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    all_keywords = _flatten_keywords(df["keywords"])
    LOGGER.debug(f"Discovered {len(all_keywords)} raw keywords")
    keyword_to_clean = {keyword: normalize_keyword(keyword) for keyword in all_keywords}
    cleaned_keywords = sorted(
        list([kw for kw in dict.fromkeys(keyword_to_clean.values()) if kw])
    )
    LOGGER.debug(f"Normalized to {len(cleaned_keywords)} unique keywords")
    for kw in cleaned_keywords[:100]:
        LOGGER.debug(f"\t{kw}")
    return keyword_to_clean, cleaned_keywords


def _build_canonical_keyword_map(
    cleaned_keywords: list[str],
    *,
    workers: int,
    embedding_host: str,
    embedding_model: str,
    cluster_count: int | None,
    canonical_model: str,
    canonical_temperature: float,
    canonical_provider: Provider = "ollama",
    canonical_client_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    embeddings = generate_keyword_embeddings(
        cleaned_keywords,
        workers=workers,
        host=embedding_host,
        model=embedding_model,
    )
    LOGGER.debug(f"Generated {len(embeddings)} embeddings")

    clusters = agglomerative_cluster_keywords(
        embeddings,
        n_clusters=cluster_count,
    )
    LOGGER.debug(f"Computed {clusters['cluster_id'].nunique()} keyword clusters")
    LOGGER.debug(clusters.head())

    client_config = canonical_client_config
    canonical_clusters = assign_canonical_keywords_per_cluster(
        clusters,
        provider=canonical_provider,
        client_config=client_config,
        model=canonical_model,
        temperature=canonical_temperature,
        workers=workers,
    )

    LOGGER.info(f"Generated canonical clusters:")
    LOGGER.info(canonical_clusters.head())

    canonical_clusters["canonical_keyword"] = canonical_clusters[
        "canonical_keyword"
    ].fillna(canonical_clusters["keyword"])

    LOGGER.info(canonical_clusters.head())

    clean_to_canonical = dict(
        canonical_clusters[["keyword", "canonical_keyword"]].values.tolist()
    )

    LOGGER.info(f"Mapped {len(clean_to_canonical)} keywords to canonical labels")

    return clean_to_canonical


def _apply_canonical_keyword_map(
    df: pd.DataFrame,
    keyword_to_clean: dict[str, str],
    clean_to_canonical: dict[str, str],
) -> pd.DataFrame:
    LOGGER.info("Applying Canonical Keyword Map to:")
    LOGGER.info(df.columns)
    LOGGER.info(df.head())

    def _map_keywords(keywords: Iterable[str]) -> list[str]:
        canonical: list[str] = []
        for keyword in keywords:
            clean_kw = keyword_to_clean.get(keyword, keyword)
            canonical_kw = clean_to_canonical.get(clean_kw, clean_kw).lower()
            canonical.append(canonical_kw)
        return sorted(set(filter(None, canonical)))

    df = df.copy()
    df["canonical_keywords"] = df["keywords"].apply(_map_keywords)
    df["canonical_keywords"] = df["canonical_keywords"].apply(json.dumps)

    LOGGER.info("Assigned canonical keywords for all rows")
    LOGGER.info(df.columns)
    LOGGER.info(df.head())
    return df


@NORMALIZE_CACHE.cache
def _normalize_keywords_pipeline_cached(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    workers: int = 8,
    embedding_host: str,
    embedding_model: str,
    cluster_count: int | None = None,
    canonical_model: str,
    canonical_temperature: float = 0.01,
    canonical_provider: Provider = "ollama",
    canonical_client_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load keywords from the DB, cluster them, and assign canonical labels."""

    LOGGER.info(f"Loading keyword metadata from {db_path}")
    df = get_keywords_by_pdf_hash(db_path)
    LOGGER.info(f"Loaded {len(df)} rows")
    if df.empty:
        df["canonical_keywords"] = [[] for _ in range(len(df))]
        return df

    keyword_to_clean_map, cleaned_keywords = _collect_cleaned_keywords(df)

    if not cleaned_keywords:
        LOGGER.warning("No keywords remained after normalization.")
        df["canonical_keywords"] = [
            sorted({kw for kw in keywords if kw}) for keywords in df["keywords"]
        ]
        return df

    clean_to_canonical = _build_canonical_keyword_map(
        cleaned_keywords,
        workers=workers,
        embedding_host=embedding_host,
        embedding_model=embedding_model,
        cluster_count=cluster_count,
        canonical_model=canonical_model,
        canonical_temperature=canonical_temperature,
        canonical_provider=canonical_provider,
        canonical_client_config=canonical_client_config,
    )

    return _apply_canonical_keyword_map(df, keyword_to_clean_map, clean_to_canonical)


def normalize_keywords_pipeline(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    workers: int = 4,
    embedding_host: str,
    embedding_model: str,
    cluster_count: int | None = None,
    canonical_model: str,
    canonical_temperature: float = 0.0,
    canonical_provider: Provider = "ollama",
    canonical_client_config: dict[str, Any] | None = None,
    write_to_disk: bool = False,
) -> pd.DataFrame:
    df = _normalize_keywords_pipeline_cached(
        db_path=db_path,
        workers=workers,
        embedding_host=embedding_host,
        embedding_model=embedding_model,
        cluster_count=cluster_count,
        canonical_model=canonical_model,
        canonical_temperature=canonical_temperature,
        canonical_provider=canonical_provider,
        canonical_client_config=canonical_client_config,
    )
    LOGGER.info("Applied Keyword Normalization")
    LOGGER.info(df.columns)
    LOGGER.info(df.head())

    def to_str(xs: Iterable):
        return json.dumps(list(xs))

    if write_to_disk:
        path = Path(db_path)
        LOGGER.info(f"Writing normalized metadata to {path}")
        df_to_write = df.copy()
        df_to_write["keywords"] = df["keywords"].map(to_str)
        df_to_write["canonical_keywords"] = df["canonical_keywords"].map(to_str)
        with sqlite3.connect(path) as conn:
            df_to_write.to_sql(
                "normalize_metadata", conn, if_exists="replace", index=False
            )

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
    keywords = _load_list(row.get("canonical_keywords") or row.get("keywords"))
    tags = sorted({kw.lower() for kw in keywords if kw})

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
        categories=tags,
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


def _writer_payload(entity: BookMetadataEntity) -> dict[str, Any]:
    def _list_or_none(values: list[str]) -> list[str] | None:
        cleaned = [value for value in values if value]
        return cleaned or None

    payload: dict[str, Any] = {
        "title": entity.title,
        "subtitle": entity.subtitle,
        "description": entity.description,
        "publisher": entity.publisher,
        "authors": _list_or_none(entity.authors),
        "categories": _list_or_none(entity.categories) or _list_or_none(entity.tags),
        "tags": _list_or_none(entity.tags),
        "isbn10": entity.isbn10,
        "isbn13": entity.isbn13,
    }
    return {key: value for key, value in payload.items() if value}


def _write_metadata_with_cli(
    source_pdf: Path,
    destination_pdf: Path,
    payload: dict[str, Any],
    writer_bin: Path,
) -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        metadata_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                str(writer_bin),
                str(source_pdf),
                str(destination_pdf),
                str(metadata_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"pdf-metadata-writer CLI failed: {exc.stderr or exc.stdout}"
        ) from exc
    finally:
        metadata_path.unlink(missing_ok=True)


def _process_pdf(
    pdf_path: Path,
    metadata_by_hash: dict[str, BookMetadataEntity],
    metadata_by_name: dict[str, BookMetadataEntity],
    output_dir: Path,
    failure_log: Path,
    pdf_root: Path,
    writer_bin: Path,
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
        LOGGER.info(f"Skipping {pdf_path}; output already exists at {target_pdf}")
        return

    pdf_hash = _compute_pdf_hash(pdf_path)
    entity = metadata_by_hash.get(pdf_hash)

    LOGGER.info(f"Processing {pdf_path} (hash={pdf_hash})")
    LOGGER.info(f"Hash lookup result: {entity}")
    if not entity:
        entity = metadata_by_name.get(pdf_path.name)
        LOGGER.info(f"Name lookup result: {entity}")
    if not entity:
        LOGGER.warning(f"No metadata available for {pdf_path} (hash={pdf_hash})")
        return

    metadata_payload = _writer_payload(entity)
    if not metadata_payload:
        LOGGER.warning(f"No metadata fields to embed for {pdf_path}")
        return
    try:
        _write_metadata_with_cli(pdf_path, target_pdf, metadata_payload, writer_bin)
        LOGGER.debug(f"Embedded metadata for {pdf_path} into {target_pdf}")
    except Exception as exc:
        LOGGER.error(f"Failed processing {pdf_path}: {exc}")
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
    writer_bin: Path | None = None,
    workers: int = 16,
    n_clusters=1000,
    embedding_model: str,
    canonical_model: str,
    embedding_host: str,
    canonical_provider: Provider = "ollama",
    canonical_client_config: dict[str, Any] | None = None,
) -> None:
    LOGGER.info(f"Normalizing metadata from {db_path}")
    df = normalize_keywords_pipeline(
        db_path=db_path,
        write_to_disk=True,
        cluster_count=n_clusters,
        workers=workers,
        embedding_model=embedding_model,
        canonical_model=canonical_model,
        embedding_host=embedding_host,
        canonical_provider=canonical_provider,
        canonical_client_config=canonical_client_config,
    )
    df = attach_book_metadata_payloads(df)
    df = attach_canonical_publishers(df)
    metadata_by_hash, metadata_by_name = dataframe_to_metadata_maps(df)
    LOGGER.info(
        f"Built metadata maps with {len(metadata_by_hash)} hash entries and "
        f"{len(metadata_by_name)} name entries"
    )

    is_file = pdf_path.is_file()
    total_pdfs = 1 if is_file else _count_pdfs_with_ripgrep(pdf_path)
    LOGGER.info(f"Found {total_pdfs} PDF(s) to process")

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_dir / "metadata_failures.log"
    failure_log.write_text("", encoding="utf-8")
    pdf_root = pdf_path if pdf_path.is_dir() else pdf_path.parent
    resolved_writer = _writer_binary(writer_bin)

    pdfs = [pdf_path] if is_file else discover_pdfs(pdf_path)
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
                resolved_writer,
            )
            for pdf in pdfs
        ]
        with tqdm(total=total_pdfs, desc="Processing PDFs", unit="pdf") as progress:
            for future in as_completed(futures):
                try:
                    future.result()
                finally:
                    progress.update(1)


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
        default="ERROR",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.json"),
        help="Path to secrets JSON containing OLLAMA_HOST.",
    )
    parser.add_argument(
        "--embedding-model",
        default="snowflake-arctic-embed:335m",
        help="Model ID for keyword embedding generation.",
    )
    parser.add_argument(
        "--canonical-model",
        default="qwen2.5vl:32b",
        help="Model ID used to choose canonical keywords.",
    )
    parser.add_argument(
        "--canonical-provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider used to choose canonical keywords.",
    )
    parser.add_argument(
        "--writer-bin",
        type=Path,
        default=None,
        help="Path to the pdf-metadata-writer CLI binary (or set BOOKMETA_WRITER_BIN).",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    secrets = _read_secrets(args.secrets)
    ollama_host = secrets["OLLAMA_HOST"]
    canonical_client_config = _client_config_for(args.canonical_provider, secrets)

    run_metadata_writer(
        db_path=args.db_path,
        pdf_path=args.pdf_path.resolve(),
        output_dir=args.output_dir.resolve(),
        writer_bin=args.writer_bin,
        workers=args.workers,
        embedding_model=args.embedding_model,
        canonical_model=args.canonical_model,
        embedding_host=ollama_host,
        canonical_provider=args.canonical_provider,
        canonical_client_config=canonical_client_config,
    )


if __name__ == "__main__":
    main()
