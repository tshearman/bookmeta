import ollama
import openai
import pytesseract
from joblib import Memory

from bookmeta.config.settings import CACHE_ROOT

LLM_CACHE_DIR = CACHE_ROOT / "llm_calls"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LLM_MEMORY = Memory(str(LLM_CACHE_DIR), verbose=0)


def _clean_string(value: str) -> str:
    """Remove surrogate code points while preserving readable characters."""
    replaced = value.encode("utf-8", "replace").decode("utf-8")
    return replaced.encode("utf-8", "ignore").decode("utf-8")


def _sanitize_payload(obj):
    if isinstance(obj, str):
        return _clean_string(obj)
    if isinstance(obj, list):
        return [_sanitize_payload(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_payload(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _sanitize_payload(val) for key, val in obj.items()}
    return obj


@LLM_MEMORY.cache(ignore=["client"])
def cached_ollama_chat(
    model: str, messages: list[ollama.Message], client: ollama.Client, **kwargs
):
    return client.chat(model=model, messages=messages, **kwargs)


@LLM_MEMORY.cache(ignore=["client"])
def cached_openapi_response_text(model: str, input, client: openai.OpenAI, **kwargs):
    safe_input = _sanitize_payload(input)
    response = client.responses.parse(model=model, input=safe_input, **kwargs)
    return response.output_text


@LLM_MEMORY.cache(ignore=["client"])
def cached_openapi_response_parsed(model: str, input, client: openai.OpenAI, **kwargs):
    safe_input = _sanitize_payload(input)
    response = client.responses.parse(model=model, input=safe_input, **kwargs)
    return response.output_parsed


@LLM_MEMORY.cache
def cached_pytesseract_image_to_string(img, lang):
    return pytesseract.image_to_string(img, lang=lang)
