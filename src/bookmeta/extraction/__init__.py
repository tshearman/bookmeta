from openai import OpenAI

from bookmeta.extraction.blocks import blocks_from_ocr
from bookmeta.extraction.llm import build_llm_request
from bookmeta.llm import openai_response_parsed
from bookmeta.types.bookinfo import BookInfoResponse, BookInfoResult
from bookmeta.types.extraction import ExtractionTask, LLMTask


def execute_extraction_task(task: ExtractionTask) -> LLMTask:
    blocks = blocks_from_ocr(task.pdf, task.ocr_results, task.config)
    llm_request = build_llm_request(blocks, task.config)
    return LLMTask(task.pdf, task, llm_request, task.config.provider_config)


def execute_llm_task(task: LLMTask) -> BookInfoResult | None:
    """Send the LLM request to the configured provider and parse the result."""
    provider = task.config.provider
    model = task.config.model

    if provider == "openai":
        client = OpenAI(**task.config.client_config)
        parsed = openai_response_parsed(
            model=model,
            input=task.payload["messages"],
            client=client,
            text_format=BookInfoResponse,
        )
        if parsed is not None:
            return BookInfoResult(task.pdf, parsed)

    raise ValueError(f"Unsupported provider: {provider}")
