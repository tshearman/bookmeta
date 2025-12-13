from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Any
import ollama
from openai import OpenAI
from bookinfo import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL, Provider
from rank import BookInfoRankPipeline
from .providers.openai import openai_bookinfo_rank
from .providers.ollama import ollama_bookinfo_rank


@dataclass
class BookInfoPipelineConfig:
    provider: Provider
    client_config: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


def generate_pipeline(config: BookInfoPipelineConfig) -> BookInfoRankPipeline:
    if config.provider == "openai":
        client = OpenAI(**config.client_config)
        model = config.model or DEFAULT_OPENAI_MODEL
        return openai_bookinfo_rank(client, model)

    if config.provider == "ollama":
        client = ollama.Client(**config.client_config)
        model = config.model or DEFAULT_OLLAMA_MODEL
        return ollama_bookinfo_rank(client, model)

    raise ValueError(f"Unsupported provider: {config.provider}")
