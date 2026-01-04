from __future__ import annotations

import logging
import warnings
from typing import Any

from joblib import Memory
from openai import OpenAI

from bookmeta.config import CACHE_ROOT
from bookmeta.utils import sanitize

LLM_CACHE_DIR = CACHE_ROOT / "bookmeta_llm_calls"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message=r"Persisting input arguments took .*s to run",
    category=UserWarning,
)
LLM_MEMORY = Memory(LLM_CACHE_DIR, verbose=2)
LOGGER = logging.getLogger("bookmeta.monitoring")


@LLM_MEMORY.cache(ignore=["client"])
def _run_openai_parsed(model: str, input: Any, client: OpenAI, **kwargs):
    print(f"Making OPENAI Api call")
    return client.responses.parse(
        model=model,
        input=sanitize(input),
        **kwargs,
    ).output_parsed


def openai_response_parsed(model: str, input: Any, client: OpenAI, **kwargs: Any):
    """
    Cached wrapper around OpenAI responses.parse.

    Caches by model + deterministic hash of the sanitized payload to avoid
    joblib hashing large inputs and to improve hit rate.
    """
    return _run_openai_parsed(model, input, client, **kwargs)
