from openai import OpenAI

from bookmetarefactor.extraction.blocks import blocks_from_ocr
from bookmetarefactor.extraction.llm import extraction_custom_id, build_llm_request
from bookmetarefactor.types.bookinfo import BookInfoResponse, BookInfoResult
from bookmetarefactor.types.extraction import ExtractionTask, LLMTask
from bookmetarefactor.utils import sanitize


def execute_extraction_task(task: ExtractionTask) -> LLMTask:
    blocks = blocks_from_ocr(task.pdf, task.ocr_results, task.config)
    id_ = extraction_custom_id(blocks, task.config)
    llm_request = build_llm_request(id_, blocks, task.config)
    return LLMTask(task.pdf, id_, task, llm_request, task.config.provider_config)


def execute_llm_task(task: LLMTask) -> BookInfoResult | None:
    """Send the LLM request to the configured provider and parse the result."""
    provider = task.config.provider
    model = task.config.model
    payload = sanitize(task.payload)

    if provider == "openai":
        client = OpenAI(**task.config.client_config)
        response = client.responses.parse(
            model=model, input=payload, text_format=BookInfoResponse
        )
        if response.output_parsed is not None:
            return BookInfoResult(task.pdf, response.output_parsed)

    raise ValueError(f"Unsupported provider: {provider}")
