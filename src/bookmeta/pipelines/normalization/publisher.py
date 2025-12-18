from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz

from bookmeta.pipelines.normalization import (
    CORP_SUFFIX_PATTERN,
    LEADING_ARTICLE_PATTERN,
    LICENSE_SPLIT_PATTERN,
)
from bookmeta.pipelines.normalization.keywords import normalize_keyword


def normalize_publisher(name: str) -> str:
    cleaned = normalize_keyword(name)
    cleaned = LICENSE_SPLIT_PATTERN.sub("", cleaned).strip()
    cleaned = LEADING_ARTICLE_PATTERN.sub(r"\1", cleaned)
    cleaned = cleaned.split("(")[0].strip()
    cleaned = CORP_SUFFIX_PATTERN.split(cleaned)[0].strip()
    tokens = cleaned.split()
    normed = " ".join(tokens)
    return normed or normalize_keyword(name)


def fuzzy_group_publishers(
    publishers: Iterable[str], threshold: int = 75
) -> list[list[str]]:
    normalized = {pub: normalize_publisher(pub) for pub in publishers}
    groups: list[list[str]] = []
    seen: set[str] = set()

    def _matches(norm: str, other: str) -> bool:
        if fuzz.ratio(norm, other) >= threshold:
            return True
        if norm and other.startswith(norm):
            return True
        if other and norm.startswith(other):
            return True
        return False

    for original, norm in normalized.items():
        if original in seen:
            continue
        matches: list[str] = []
        for candidate, other in normalized.items():
            if candidate in seen:
                continue
            if _matches(norm, other):
                matches.append(candidate)
        if matches:
            groups.append(matches)
            seen.update(matches)
    return groups


def assign_canonical_publishers(
    df: pd.DataFrame, clusters: list[list[str]]
) -> pd.DataFrame:
    if "publisher" not in df.columns:
        raise ValueError("DataFrame must contain a 'publisher' column.")
    mapping: dict[str, str] = {}
    for group in clusters:
        if not group:
            continue
        canonical = sorted(group, key=lambda s: (len(s), s))[0]
        for name in group:
            mapping[name] = canonical
    result = df.copy()
    result["canonical_publisher"] = result["publisher"].map(mapping)
    result["canonical_publisher"].fillna(result["publisher"], inplace=True)
    return result


def attach_canonical_publishers(df: pd.DataFrame) -> pd.DataFrame:
    publishers = list(df["publisher"].unique())
    pub_groups = fuzzy_group_publishers(publishers)
    return assign_canonical_publishers(df, pub_groups)
