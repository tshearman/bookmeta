import os
from functools import lru_cache

import ollama

DEFAULT_OLLAMA_HOST = "http://192.168.1.31:11434"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:32b"


@lru_cache(maxsize=1)
def get_ollama_client() -> ollama.Client:
    """Return a cached Ollama client configured for the desired host."""
    return ollama.Client(host="http://192.168.1.31:11434")


def resolve_ollama_model(requested_model: str | None) -> str:
    """Use qwen3-vl:32b unless the caller explicitly overrides it."""
    if not requested_model:
        return DEFAULT_OLLAMA_MODEL
    cleaned = requested_model.strip()
    if not cleaned or cleaned.startswith("gpt-"):
        return DEFAULT_OLLAMA_MODEL
    return cleaned


def data_url_to_base64(data_url: str) -> str:
    """Strip the data URL header if present."""
    if data_url.startswith("data:"):
        parts = data_url.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return data_url


def prepare_ollama_images(images: list[str]) -> list[str]:
    """Normalize images to base64 strings without headers."""
    return [data_url_to_base64(image).strip() for image in images if image]
