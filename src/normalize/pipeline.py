from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from storage import DEFAULT_DB_PATH, _compute_pdf_hash

from src.normalize.keywords import (
    agglomerative_cluster_keywords,
    assign_canonical_keywords_per_cluster,
    generate_keyword_embeddings,
    get_keywords_by_pdf_hash,
    normalize_keyword,
)
from src.normalize.publisher import normalize_publisher

LOGGER = logging.getLogger(__name__)
FAILURE_LOG_LOCK = Lock()


def _flatten_keywords(keyword_sets: Iterable[Iterable[str]]) -> list[str]:
    unique: dict[str, None] = {}
    for group in keyword_sets:
        for keyword in group:
            if keyword not in unique:
                unique[keyword] = None
    return list(unique.keys())


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
    description: str | None
    isbn13: str | None
    isbn10: str | None


def row_to_book_metadata(row: pd.Series) -> BookMetadataEntity:
    keywords = _load_list(row.get("keywords"))
    canonical = _load_list(row.get("canonical_keywords"))
    categories = sorted({kw.lower() for kw in keywords + canonical if kw})

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

    return BookMetadataEntity(
        authors=authors,
        title=row.get("title"),
        subtitle=row.get("subtitle"),
        publisher=publisher.title() if publisher else None,
        categories=categories,
        description=row.get("description"),
        isbn13=isbn_13,
        isbn10=isbn_10,
    )


def attach_book_metadata_payloads(df: pd.DataFrame) -> pd.DataFrame:
    def _build(row: pd.Series) -> str:
        metadata = row_to_book_metadata(row)
        payload = {
            "authors": metadata.authors,
            "title": metadata.title,
            "subtitle": metadata.subtitle,
            "publisher": metadata.publisher,
            "categories": metadata.categories,
            "description": metadata.description,
            "isbn_13": metadata.isbn13,
            "isbn_10": metadata.isbn10,
        }
        return json.dumps(payload, ensure_ascii=False)

    df = df.copy()
    df["book_metadata_payload"] = df.apply(_build, axis=1)
    return df


def dataframe_to_metadata_map(df: pd.DataFrame) -> dict[str, BookMetadataEntity]:
    records: dict[str, BookMetadataEntity] = {}
    for _, row in df.iterrows():
        pdf_hash = str(row.get("pdf_hash"))
        if not pdf_hash:
            continue
        records[pdf_hash] = row_to_book_metadata(row)
    return records


def _collect_pdfs(pdf_path: Path) -> list[Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF path not found: {pdf_path}")
    if pdf_path.is_file():
        return [pdf_path]
    return sorted(pdf_path.rglob("*.pdf"))


def _compute_java_classpath(java_root: Path) -> str:
    build_dir = java_root / "build"
    default_cp = ":".join(
        [
            str(build_dir / "classes" / "java" / "main"),
            str(build_dir / "resources" / "main"),
        ]
    )
    gradle_exec = java_root / "gradlew"
    if gradle_exec.exists():
        try:
            result = subprocess.run(
                [str(gradle_exec), "-q", "printClasspath"],
                cwd=java_root,
                capture_output=True,
                text=True,
                check=True,
            )
            extra = result.stdout.strip()
            if extra:
                return f"{default_cp}:{extra}"
        except subprocess.CalledProcessError as exc:
            LOGGER.warning(
                "Failed to compute Gradle classpath via printClasspath: %s", exc
            )
    else:
        LOGGER.warning("gradlew not found at %s; using default classpath", gradle_exec)
    return default_cp


def _write_metadata_file(entity: BookMetadataEntity) -> Path:
    temp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    try:
        json.dump(asdict(entity), temp, ensure_ascii=False, indent=2)
    finally:
        temp.close()
    return Path(temp.name)


def _run_java_writer(
    pdf: Path,
    metadata_file: Path,
    output_dir: Path,
    classpath: str,
    java_root: Path,
    java_binary: str,
) -> None:
    cmd = [
        java_binary,
        "-cp",
        classpath,
        "com.adityachandel.booklore.service.metadata.writer.PdfMetadataWriter",
        "--pdf",
        str(pdf),
        "--out",
        str(output_dir),
        "--metadata",
        str(metadata_file),
    ]
    LOGGER.info(
        "Executing Java metadata writer: %s", " ".join(str(part) for part in cmd)
    )
    try:
        subprocess.run(cmd, check=True, cwd=java_root, capture_output=True, text=True)
    except FileNotFoundError as exc:
        LOGGER.error(
            "Java executable not found while processing %s. Ensure Java is installed.",
            pdf,
        )
        raise RuntimeError("Java runtime not found") from exc
    except subprocess.CalledProcessError as exc:
        LOGGER.error(
            "Java process failed for %s with return code %s", pdf, exc.returncode
        )
        if exc.stdout:
            LOGGER.error("STDOUT:\n%s", exc.stdout)
        if exc.stderr:
            LOGGER.error("STDERR:\n%s", exc.stderr)
        raise RuntimeError(
            "Java command failed. Ensure a JVM is installed and available."
        ) from exc


def _process_pdf(
    pdf_path: Path,
    metadata_map: dict[str, BookMetadataEntity],
    output_dir: Path,
    classpath: str,
    java_root: Path,
    java_binary: str,
    failure_log: Path,
    pdf_root: Path,
) -> None:
    try:
        relative = pdf_path.relative_to(pdf_root)
    except ValueError:
        relative = Path(pdf_path.name)

    target_dir = output_dir / relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = target_dir / relative.name
    if target_pdf.exists():
        LOGGER.info("Skipping %s; output already exists at %s", pdf_path, target_pdf)
        return

    pdf_hash = _compute_pdf_hash(pdf_path)
    entity = metadata_map.get(pdf_hash)
    if not entity:
        LOGGER.warning("No metadata available for %s (hash=%s)", pdf_path, pdf_hash)
        return
    metadata_file = _write_metadata_file(entity)
    try:
        _run_java_writer(
            pdf_path, metadata_file, target_dir, classpath, java_root, java_binary
        )
        LOGGER.info("Wrote metadata for %s to %s", pdf_path, target_dir)
    except Exception as exc:
        LOGGER.error("Failed processing %s: %s", pdf_path, exc)
        with FAILURE_LOG_LOCK:
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            with failure_log.open("a", encoding="utf-8") as fh:
                fh.write(f"{pdf_path}: {exc}\n")
        return
    finally:
        metadata_file.unlink(missing_ok=True)


def run_metadata_writer(
    db_path: Path | str,
    pdf_path: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    n_clusters=750,
    java_root: Path | None = None,
    java_binary: str = "java",
) -> None:
    LOGGER.info("Normalizing metadata from %s", db_path)
    df = normalize_keywords_pipeline(
        db_path=db_path, write_to_disk=False, cluster_count=n_clusters
    )
    df = attach_book_metadata_payloads(df)
    metadata_map = dataframe_to_metadata_map(df)
    LOGGER.info("Built metadata map with %d entries", len(metadata_map))

    pdfs = _collect_pdfs(pdf_path)
    LOGGER.info("Found %d PDF(s) to process", len(pdfs))
    output_dir.mkdir(parents=True, exist_ok=True)
    java_root = java_root or PROJECT_ROOT
    classpath = _compute_java_classpath(java_root)
    failure_log = output_dir / "metadata_failures.log"
    failure_log.write_text("", encoding="utf-8")
    pdf_root = pdf_path if pdf_path.is_dir() else pdf_path.parent

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_pdf,
                pdf,
                metadata_map,
                output_dir,
                classpath,
                java_root,
                java_binary,
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
    "build_metadata_payload",
    "write_metadata_json",
    "row_to_book_metadata",
    "attach_book_metadata_payloads",
    "dataframe_to_metadata_map",
    "run_metadata_writer",
    "BookMetadataEntity",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize PDF metadata and invoke the Java PdfMetadataWriter."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF file or directory.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory where metadata will be written."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default="resources/bookmeta.db",
        help="Path to the pipeline SQLite database.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for PDF processing.",
    )
    parser.add_argument(
        "--java-root",
        type=Path,
        default="/Users/toby/projects/booklore/booklore-api",
        help="Path to the Java project root containing gradlew. Defaults to script root.",
    )
    parser.add_argument(
        "--java-binary",
        type=str,
        default="/opt/homebrew/opt/openjdk@21/bin/java",
        help="Path to the java executable. Defaults to 'java' in PATH.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    run_metadata_writer(
        db_path=args.db_path,
        pdf_path=args.pdf_path.resolve(),
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
        java_root=args.java_root.resolve() if args.java_root else None,
        java_binary=args.java_binary,
    )


if __name__ == "__main__":
    main()
