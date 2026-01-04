import hashlib

from bookmeta.extraction.blocks import filter_blocks
from bookmeta.types.blocks import ExtractionBlocks
from bookmeta.types.extraction import ExtractionConfig, LLMPayload


def extraction_custom_id(blocks: ExtractionBlocks, config: ExtractionConfig) -> str:
    hash = hashlib.sha256()
    hash.update(blocks.hash.encode("utf-8"))
    hash.update(config.hash.encode("utf-8"))
    return hash.hexdigest()


def _build_openai_request(
    blocks: ExtractionBlocks, config: ExtractionConfig
) -> LLMPayload:
    return {
        "model": config.provider_config.model,
        "messages": [
            {
                "role": "user",
                "content": [cb.as_payload() for cb in blocks.as_context_blocks],
            }
        ],
    }


def build_llm_request(blocks: ExtractionBlocks, config: ExtractionConfig) -> LLMPayload:
    """Build provider-specific LLM request payload."""
    limited_blocks = filter_blocks(blocks, config.context_limits)
    provider = config.provider_config.provider
    if provider == "openai":
        return _build_openai_request(limited_blocks, config)
    raise ValueError(f"Unsupported provider: {provider}")
