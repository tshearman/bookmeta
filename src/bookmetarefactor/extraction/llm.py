import hashlib

from bookmetarefactor.types.blocks import ExtractionBlocks
from bookmetarefactor.types.extraction import (
    ExtractionConfig,
    LLMPayload,
)
from bookmetarefactor.utils.img import img_to_b64


def extraction_custom_id(blocks: ExtractionBlocks, config: ExtractionConfig) -> str:
    hash = hashlib.sha256()
    hash.update(blocks.hash.encode("utf-8"))
    hash.update(config.hash.encode("utf-8"))
    return hash.hexdigest()


def _build_openai_request(
    custom_id: str, blocks: ExtractionBlocks, config: ExtractionConfig
) -> LLMPayload:
    return {
        "custom_id": custom_id,
        "model": config.provider_config.model,
        "messages": [
            {
                "role": "user",
                "content": [cb.as_payload() for cb in blocks.as_context_blocks],
            }
        ],
    }


def _build_ollama_request(
    custom_id: str, blocks: ExtractionBlocks, config: ExtractionConfig
) -> LLMPayload:
    text_content = "\n\n".join(
        [
            cb.as_context_block.content["text"]
            for cb in blocks.text_blocks
            if cb.as_context_block is not None
        ]
    )

    images_b64 = [img_to_b64(blk.image) for blk in blocks.img_blocks]

    return {
        "custom_id": custom_id,
        "model": config.provider_config.model,
        "messages": [
            {
                "role": "user",
                "content": text_content,
                "images": images_b64,
            }
        ],
    }


def build_llm_request(
    custom_id: str, blocks: ExtractionBlocks, config: ExtractionConfig
) -> LLMPayload:
    """Build provider-specific LLM request payload."""
    provider = config.provider_config.provider
    if provider == "openai":
        return _build_openai_request(custom_id, blocks, config)
    if provider == "ollama":
        return _build_ollama_request(custom_id, blocks, config)
    raise ValueError(f"Unsupported provider: {provider}")
