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
from openai import OpenAI
from cleantext import clean
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import validation as sk_validation
from tqdm.auto import tqdm

from bookmeta.config.settings import CACHE_ROOT, DEFAULT_DB_PATH
from bookmeta.data.sqlite import _ensure_db
from bookmeta.services.bookinfo import Provider
from bookmeta.services.llm import (
    cached_ollama_chat,
    cached_openapi_response_text,
)

LOGGER = logging.getLogger(__name__)
EMBEDDING_CACHE = joblib.Memory(CACHE_ROOT / "embedding", verbose=0)


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
    LOGGER.info(f"Loading keywords from {path}")
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT pdf_hash, pdf_name, result FROM pipeline_runs", conn
        )
    finally:
        conn.close()
    LOGGER.info(f"Read {len(df)} rows in dataframe from {path}")
    if df.empty:
        LOGGER.warning(f"No pipeline runs found in {path}")
        return pd.DataFrame(
            columns=[
                "pdf_hash",
                "pdf_name",
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
    out["keywords"] = out["keywords"].apply(
        lambda kws: sorted(set(kws)) if kws else set([])
    )
    return out.reset_index(drop=True)


@EMBEDDING_CACHE.cache
def get_keyword_embedding(host: str, model: str, keyword: str) -> list[float]:
    client = ollama.Client(host=host)
    response = client.embed(model=model, input=keyword)
    return response.get("embeddings")


@EMBEDDING_CACHE.cache
def agglomerative_cluster_keywords(
    keyword_embeddings: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    linkage: Literal["ward", "complete", "average", "single"] = "average",
    metric: str = "cosine",
) -> pd.DataFrame:

    LOGGER.info(f"Building Clusters from Embeddings:")
    LOGGER.info(keyword_embeddings.head())
    keywords = keyword_embeddings["keyword"].tolist()
    matrix = np.vstack(
        [
            np.asarray(vec, dtype=float).ravel()
            for vec in keyword_embeddings["embedding"]
        ]
    )
    LOGGER.info(f"Stacked Embeddings, shape: {matrix.shape}")
    LOGGER.info(matrix[:10])

    LOGGER.info("Generating Clusters Model")
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
            unit="keywords",
        ):
            keyword = futures[future]
            try:
                embedding = future.result()
            except Exception as exc:
                LOGGER.exception(f"Failed to embed keyword '{keyword}'")
                raise RuntimeError(
                    f"Failed to embed keyword '{keyword}': {exc}"
                ) from exc
            results.append({"keyword": keyword, "embedding": embedding})

    return pd.DataFrame(results)


@EMBEDDING_CACHE.cache
def canonicalize_keyword_group(
    keywords: list[str],
    *,
    provider: Provider = "ollama",
    model: str = "qwen2.5vl:32b",
    temperature: float = 0.0,
    client_config: dict[str, Any] | None = None,
) -> str | None:
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

    config = client_config or {}
    messages = [
        {"role": "system", "content": canonical_keyword_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if provider == "ollama":
        config = config or {"host": "http://192.168.1.31:11434"}
        client = ollama.Client(**config)
        response = cached_ollama_chat(
            model=model,
            messages=messages,
            client=client,
            options={"temperature": temperature},
        )
        result = response["message"]["content"].strip()
    elif provider == "openai":
        client = OpenAI(**config)
        result = cached_openapi_response_text(
            model=model,
            input=messages,
            client=client,
            temperature=temperature,
        ).strip()
    else:
        raise ValueError(f"Unsupported provider for canonical keywords: {provider}")

    try:
        return result.splitlines()[0].strip().strip('"')
    except:
        return None


def assign_canonical_keywords_per_cluster(
    clusters: pd.DataFrame,
    *,
    provider: Provider = "ollama",
    client_config: dict[str, Any] | None = None,
    model: str = "qwen2.5vl:32b",
    temperature: float = 0.0,
    workers: int = 4,
) -> pd.DataFrame:

    grouped = list(clusters.groupby("cluster_id", sort=True))
    workers = max(1, workers)
    mapping: dict[int, str | None] = {}
    effective_client_config = client_config
    if provider == "ollama" and effective_client_config is None:
        effective_client_config = {"host": "http://192.168.1.31:11434"}

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
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    client_config=effective_client_config,
                )
                futures[future] = cluster_id  # type: ignore

            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    mapping[cluster_id] = normalize_keyword(future.result())
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


def normalize_keyword(keyword: str | None) -> str | None:
    if keyword is None:
        return None
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
        no_punct=True,
    )
    text = text.replace("&", " and ")
    for delim in ("_", "-", "/", ","):
        text = text.replace(delim, " ")
    tokens = [token for token in text.split()]
    normalized = " ".join(tokens)
    return " ".join(normalized.split())
