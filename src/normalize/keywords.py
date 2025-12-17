import inspect
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import ollama
import pandas as pd
from cleantext import clean
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import validation as sk_validation
from tqdm.auto import tqdm

from storage import DEFAULT_DB_PATH, _ensure_db

LOGGER = logging.getLogger(__name__)
EMBEDDING_CACHE = joblib.Memory(Path("../.cache") / "embedding", verbose=0)


def _ensure_check_array_support() -> None:
    signature = inspect.signature(sk_validation.check_array)
    if "ensure_all_finite" in signature.parameters:
        return
    original = sk_validation.check_array

    def _patched_check_array(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("ensure_all_finite", None)
        return original(*args, **kwargs)

    sk_validation.check_array = _patched_check_array


_ensure_check_array_support()


def get_keywords_by_pdf_hash(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    path = Path(db_path)
    _ensure_db(path)
    LOGGER.info("Loading keywords from %s", path)
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT pdf_hash, result FROM pipeline_runs", conn)
    finally:
        conn.close()
    LOGGER.info("Read %d rows in dataframe from %s", len(df), path)
    if df.empty:
        LOGGER.warning("No pipeline runs found in %s", path)
        return pd.DataFrame(
            columns=[
                "pdf_hash",
                "title",
                "author",
                "publisher",
                "keywords",
                "description",
                "isbn_identifiers",
            ]
        )

    parsed = (
        df["result"]
        .apply(json.loads)
        .apply(pd.Series)[["title", "author", "publisher", "keywords", "description"]]
    )
    out = pd.concat([df, parsed], axis=1)
    out = out[out["keywords"].notna()]
    out["keywords"] = out["keywords"].apply(set)
    return out.reset_index(drop=True)


@EMBEDDING_CACHE.cache
def get_keyword_embedding(host: str, model: str, keyword: str) -> list[float]:
    client = ollama.Client(host=host)
    response = client.embed(model=model, input=keyword)
    return response.get("embeddings")


def agglomerative_cluster_keywords(
    keyword_embeddings: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    linkage: Literal["ward", "complete", "average", "single"] = "average",
    metric: str = "cosine",
) -> pd.DataFrame:

    keywords = keyword_embeddings["keyword"].tolist()
    matrix = np.vstack(
        [
            np.asarray(vec, dtype=float).ravel()
            for vec in keyword_embeddings["embedding"]
        ]
    )

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric=metric,
    )
    labels = model.fit_predict(matrix)
    return pd.DataFrame({"keyword": keywords, "cluster_id": labels})


def generate_keyword_embeddings(
    keywords: list[str],
    *,
    workers: int = 4,
    host: str = "http://192.168.1.31:11434",
    model: str = "snowflake-arctic-embed:335m",
) -> pd.DataFrame:
    unique_keywords = [keyword for keyword in dict.fromkeys(keywords) if keyword]
    workers = max(1, workers)
    results: list[dict[str, Any]] = []

    def embed(keyword):
        return get_keyword_embedding(host, model, keyword)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(embed, keyword): keyword for keyword in unique_keywords
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Embedding keywords",
            unit="keyword",
        ):
            keyword = futures[future]
            try:
                embedding = future.result()
            except Exception as exc:
                LOGGER.exception("Failed to embed keyword '%s'", keyword)
                raise RuntimeError(
                    f"Failed to embed keyword '{keyword}': {exc}"
                ) from exc
            results.append({"keyword": keyword, "embedding": embedding})

    return pd.DataFrame(results)


@EMBEDDING_CACHE.cache
def canonicalize_keyword_group(
    keywords: list[str],
    *,
    ollama_host: str = "http://192.168.1.31:11434",
    ollama_model: str = "qwen2.5vl:32b",
    temperature: float = 0.0,
) -> str:
    if not keywords:
        raise ValueError("keywords must be a non-empty collection of strings.")
    cleaned = [keyword.strip() for keyword in keywords if keyword and keyword.strip()]
    if not cleaned:
        raise ValueError("No usable keywords provided.")

    keyword_list = "\n".join(f"- {keyword}" for keyword in cleaned)

    canonical_keyword_system_prompt = (
        "You are a meticulous metadata librarian. You will receive a noisy cluster "
        "of related keywords or tags; some may be redundant, imprecise, or outliers. Choose a "
        "single concise English noun phrase that best represents the dominant shared "
        "concept across the cluster. It can be a new keyword if that yields a clearer "
        "label. Respond with only the chosen keyword."
    )

    user_prompt = (
        "Choose a single concise English keyword that best represents the "
        "shared concept. Respond with only the keyword.\n"
        f"KEYWORDS:\n{keyword_list}"
    )

    client = ollama.Client(host=ollama_host)
    response = client.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": canonical_keyword_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )
    result = response["message"]["content"].strip()
    return result.splitlines()[0].strip().strip('"')


def assign_canonical_keywords_per_cluster(
    clusters: pd.DataFrame,
    *,
    ollama_host: str = "http://192.168.1.31:11434",
    ollama_model: str = "qwen2.5vl:32b",
    temperature: float = 0.0,
    workers: int = 4,
) -> pd.DataFrame:

    grouped = list(clusters.groupby("cluster_id", sort=True))
    workers = max(1, workers)
    mapping: dict[int, str | None] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Any, int] = {}
        progress = tqdm(
            total=len(grouped),
            desc="Canonicalizing clusters",
            unit="cluster",
        )
        try:
            for cluster_id, df in grouped:
                keywords = df["keyword"].dropna().tolist()
                if not keywords:
                    mapping[cluster_id] = None  # type: ignore
                    progress.update(1)
                    continue
                future = executor.submit(
                    canonicalize_keyword_group,
                    keywords,
                    ollama_host=ollama_host,
                    ollama_model=ollama_model,
                    temperature=temperature,
                )
                futures[future] = cluster_id  # type: ignore

            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    mapping[cluster_id] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to canonicalize cluster {cluster_id}: {exc}"
                    ) from exc
                finally:
                    progress.update(1)
        finally:
            progress.close()

    result = clusters.copy()
    result["canonical_keyword"] = result["cluster_id"].map(mapping)
    return result


def normalize_keyword(keyword: str) -> str:
    if not keyword:
        return ""
    text = clean(
        keyword,
        fix_unicode=True,
        lower=True,
        to_ascii=False,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
    )
    text = text.replace("&", " and ")
    for delim in ("_", "-", "/"):
        text = text.replace(delim, " ")
    tokens = [token for token in text.split()]
    normalized = " ".join(tokens)
    return " ".join(normalized.split())
