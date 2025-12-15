import json
import logging
from dataclasses import dataclass
from typing import Any
import ollama
from openai import OpenAI
from pydantic import Field
from bookinfo import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL, Provider
from booksearch.pipeline import BookSearchResults
from datamodel.book_info import DetailedBookInfo
from datamodel.pdf_ocr_results import PdfOcrResults
from rank import BookInfoSelectionPipeline, BookSearchCandidate
from rank.providers.ollama import ollama_selection_runner
from rank.providers.openai import openai_selection_runner


@dataclass
class BookInfoSelectionPipelineConfig:
    provider: Provider
    client_config: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


def search_results_to_candidates(
    results: BookSearchResults,
) -> list[BookSearchCandidate]:
    if not results or not results.results:
        return []
    candidates: list[BookSearchCandidate] = []
    for result in results.results:
        try:
            candidates.append(json.loads(result.payload))
        except json.JSONDecodeError:
            logging.warning("Skipping invalid book search payload.")
    return candidates


def generate_selection_pipeline(
    config: BookInfoSelectionPipelineConfig,
) -> BookInfoSelectionPipeline:

    def forked(selection):
        def pipeline(
            ocr_results: PdfOcrResults, search_results: BookSearchResults
        ) -> DetailedBookInfo:
            return selection(ocr_results, search_results_to_candidates(search_results))  # type: ignore

        return pipeline

    if config.provider == "openai":
        client = OpenAI(**config.client_config)
        model = config.model or DEFAULT_OPENAI_MODEL
        return forked(openai_selection_runner(client, model))

    if config.provider == "ollama":
        client = ollama.Client(**config.client_config)
        model = config.model or DEFAULT_OLLAMA_MODEL
        return forked(ollama_selection_runner(client, model))

    raise ValueError(f"Unsupported provider: {config.provider}")
