from pathlib import Path
from joblib import Memory
import ollama
import openai
import pytesseract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm_calls"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Setting verbose>=20 enables joblib's info-level logging hooks
LLM_MEMORY = Memory(str(LLM_CACHE_DIR), verbose=20)


@LLM_MEMORY.cache(ignore=["client"])
def cached_ollama_chat(
    model: str, messages: list[ollama.Message], client: ollama.Client, **kwargs
):
    return client.chat(model=model, messages=messages, **kwargs)


@LLM_MEMORY.cache(ignore=["client"])
def cached_openapi_response_text(model: str, input, client: openai.OpenAI, **kwargs):
    response = client.responses.parse(model=model, input=input, **kwargs)
    return response.output_text


@LLM_MEMORY.cache(ignore=["client"])
def cached_openapi_response_parsed(model: str, input, client: openai.OpenAI, **kwargs):
    response = client.responses.parse(model=model, input=input, **kwargs)
    return response.output_parsed


@LLM_MEMORY.cache
def cached_pytesseract_image_to_string(img, lang):
    return pytesseract.image_to_string(img, lang=lang)
