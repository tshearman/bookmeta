import json
import logging
from pathlib import Path
from typing import Any

import ollama
from openai import OpenAI
from pydantic import Field
from pydantic.dataclasses import dataclass

from bookinfo import (
    BookInfoRequestPipeline,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    Provider,
)
from .providers.openai import openai_bookinfo_request
from .providers.ollama import ollama_bookinfo_request
from ocr.pipeline import OcrPipelineConfig, generate_pipeline as generate_ocr_pipeline


@dataclass
class BookInfoPipelineConfig:
    provider: Provider
    client_config: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


def generate_pipeline(config: BookInfoPipelineConfig) -> BookInfoRequestPipeline:
    """Return a callable that executes the configured book-info request."""

    if config.provider == "openai":
        client = OpenAI(**config.client_config)
        model = config.model or DEFAULT_OPENAI_MODEL
        return openai_bookinfo_request(client, model)

    if config.provider == "ollama":
        client = ollama.Client(**config.client_config)
        model = config.model or DEFAULT_OLLAMA_MODEL
        return ollama_bookinfo_request(client, model)

    raise ValueError(f"Unsupported provider: {config.provider}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_pdf = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "test_pdfs"
        / "bladesinthedark_v8_2.pdf"
    )
    if not sample_pdf.exists():
        raise FileNotFoundError(f"Sample PDF not found: {sample_pdf}")

    secrets_path = Path("secrets.json")
    if not secrets_path.exists():
        raise FileNotFoundError(f"Missing secrets file: {secrets_path}")

    with secrets_path.open("r") as f:
        secrets = json.load(f)

    openai_api_key = secrets.get("OPENAI_API_KEY")
    openai_project = secrets.get("OPENAI_PROJECT_ID")

    logging.info("Running OCR pipeline to gather inputs...")
    ocr_pipeline = generate_ocr_pipeline(OcrPipelineConfig())
    ocr_results = ocr_pipeline(sample_pdf)
    logging.info("OCR pipeline produced %d page samples", len(ocr_results.ocr_results))

    def run_and_log_pipeline(config: BookInfoPipelineConfig, label: str) -> None:
        logging.info("Running BookInfo pipeline with %s provider...", label)
        pipeline = generate_pipeline(config)
        response = pipeline(ocr_results)
        if not response:
            logging.info("%s pipeline returned no response.", label)
        else:
            logging.info("%s pipeline completed successfully.", label)
            logging.info("BookInfo result:\n%s", response)

    run_and_log_pipeline(
        BookInfoPipelineConfig(
            provider="openai",
            client_config={"api_key": openai_api_key, "project": openai_project},
        ),
        "OpenAI",
    )

    ollama_host = secrets.get("OLLAMA_HOST")
    run_and_log_pipeline(
        BookInfoPipelineConfig(
            provider="ollama",
            client_config={"host": ollama_host},
            model=secrets.get("OLLAMA_MODEL", "qwen3-vl:32b"),
        ),
        "Ollama",
    )
