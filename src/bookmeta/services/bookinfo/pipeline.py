from typing import Any

import ollama
from openai import OpenAI
from pydantic import Field
from pydantic.dataclasses import dataclass

from bookmeta.services.bookinfo import (
    BOOK_PROMPT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    BookInfoRequestPipeline,
    Provider,
)
from bookmeta.services.bookinfo.blocks import ContextLimits

from .providers.ollama import ollama_bookinfo_request
from .providers.openai import openai_bookinfo_request


@dataclass
class BookInfoPipelineConfig:
    provider: Provider
    client_config: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    context_limits: ContextLimits | None = None
    prompt: str | None = None


def generate_pipeline(config: BookInfoPipelineConfig) -> BookInfoRequestPipeline:
    """Return a callable that executes the configured book-info request."""

    prompt = config.prompt or BOOK_PROMPT

    if config.provider == "openai":
        client = OpenAI(**config.client_config)
        model = config.model or DEFAULT_OPENAI_MODEL
        return openai_bookinfo_request(client, model, prompt, config.context_limits)

    if config.provider == "ollama":
        client = ollama.Client(**config.client_config)
        model = config.model or DEFAULT_OLLAMA_MODEL
        return ollama_bookinfo_request(client, model, prompt, config.context_limits)

    raise ValueError(f"Unsupported provider: {config.provider}")
