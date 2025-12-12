import json
import logging
from pathlib import Path
from typing import Sequence

from joblib import Memory
from openai import OpenAI
from pydantic.dataclasses import dataclass

from google_books_volume import GoogleBooksVolume
from pdf_processor import PdfProcessingResult
from book_info_extractor import image_path_to_data_url
from ollama_client import (
    data_url_to_base64,
    get_ollama_client,
    prepare_ollama_images,
    resolve_ollama_model,
)


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache/bookmeta"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(location=str(CACHE_DIR), verbose=0)


@dataclass
class Rank:
    rank: int
    confidence: float


RANK_PROMPT = """
You are given images of a book (cover + sample interior pages), two OCR transcripts
("NATIVE OCR" and "TESSERACT OCR"), and up to five Google Books candidate entries.

Determine which candidate (if any) best matches the book shown in the images. Return:
  - rank: 1-based index of the best matching candidate (1-5). If none match, return -1.
  - confidence: a float between 0 and 1 indicating how certain you are in this match.

Use only the provided evidence. Prefer candidates whose title, author, publisher,
subject, ISBN, and description align with the visible text. Be conservative and choose
confidence near 0 for weak matches.
"""


def format_candidates(volumes: Sequence[GoogleBooksVolume]) -> str:
    lines: list[str] = ["CANDIDATE RESPONSES:"]
    for idx, volume in enumerate(volumes[:5], start=1):
        info = volume.volume_info
        title = (info.title if info and info.title else "UNKNOWN")
        authors = ", ".join(info.authors) if (info and info.authors) else "UNKNOWN"
        publisher = getattr(info, "publisher", None) or "UNKNOWN"
        categories = ", ".join(info.categories) if (info and info.categories) else "NONE"
        description = info.description if info and info.description else "NONE"
        identifiers = []
        if info:
            for ident in info.industry_identifiers:
                if ident.identifier:
                    label = f"{ident.type}:{ident.identifier}" if ident.type else ident.identifier
                    identifiers.append(label)
        lines.append(f"--- Candidate {idx} ---")
        lines.append(f"Title: {title}")
        lines.append(f"Authors: {authors}")
        lines.append(f"Publisher: {publisher}")
        lines.append(f"Categories: {categories}")
        lines.append(f"Description: {description}")
        lines.append(f"Identifiers: {', '.join(identifiers) if identifiers else 'NONE'}")
    return "\n".join(lines)


def construct_rank_content_blocks(
    pdf_result: PdfProcessingResult,
    volumes: Sequence[GoogleBooksVolume],
    context_path: str | None = None,
) -> list[dict[str, str]]:
    content_blocks: list[dict[str, str]] = [
        {"type": "input_text", "text": RANK_PROMPT},
        {"type": "input_text", "text": format_candidates(volumes)},
    ]

    if context_path:
        content_blocks.append(
            {
                "type": "input_text",
                "text": f"PDF CONTEXT\nRelative Path: {context_path}",
            }
        )

    for page in pdf_result.pages:
        content_blocks.append(
            {"type": "input_image", "image_url": image_path_to_data_url(page.image_path)}
        )

    native_text = pdf_result.combined_text_for("native_ocr")
    if native_text:
        content_blocks.append({"type": "input_text", "text": f"NATIVE OCR\n{native_text}"})

    tesseract_text = pdf_result.combined_text_for("tesseract_ocr")
    if tesseract_text:
        content_blocks.append(
            {"type": "input_text", "text": f"TESSERACT OCR\n{tesseract_text}"}
        )

    return content_blocks


def _blocks_to_prompt_and_images(content_blocks: list[dict[str, str]]):
    texts: list[str] = []
    images: list[str] = []
    for block in content_blocks:
        if block["type"] == "input_text":
            texts.append(block["text"])
        elif block["type"] == "input_image":
            images.append(data_url_to_base64(block["image_url"]))
    return "\n\n".join(texts), images


def rank_google_books_candidates(
    pdf_result: PdfProcessingResult,
    volumes: Sequence[GoogleBooksVolume],
    client: OpenAI | None,
    model: str = "gpt-4.1-mini",
    context_path: str | None = None,
    provider: str = "openai",
) -> Rank:
    """
    Determine which Google Books candidate best matches the PDF evidence.
    """
    if not volumes:
        return Rank(rank=-1, confidence=0.0)

    content_blocks = construct_rank_content_blocks(
        pdf_result,
        volumes[:5],
        context_path=context_path,
    )
    logging.info("[Rank] Evaluating %d candidates with model '%s' via %s", len(volumes), model, provider)
    if provider == "ollama":
        prompt_text, images = _blocks_to_prompt_and_images(content_blocks)
        response = _ollama_rank_request(prompt_text, images, model)
    else:
        if client is None:
            raise ValueError("OpenAI client is required when provider='openai'")
        response = _openai_rank_request(content_blocks, model=model, client=client)
    logging.info("[Rank] Model selected rank=%s confidence=%.3f", response.rank, response.confidence)
    return response


@memory.cache(ignore=["client"])
def _openai_rank_request(content_blocks, model, client: OpenAI | None) -> Rank:
    logging.info("[Rank] Cache miss, calling OpenAI model=%s", model)
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
        text_format=Rank,
    )
    status_code = getattr(getattr(response, "response", None), "status_code", None)
    if status_code is not None and status_code != 200:
        raise RuntimeError(f"OpenAI rank request failed with status {status_code}")
    return response.output_parsed


@memory.cache()
def _ollama_rank_request(prompt_text: str, images: list[str], model: str) -> Rank:
    message: dict = {
        "role": "user",
        "content": f"{prompt_text}\n\nReturn ONLY JSON with fields rank (int) and confidence (float).",
    }
    encoded_images = prepare_ollama_images(images)
    if encoded_images:
        message["images"] = encoded_images
    client = get_ollama_client()
    response = client.chat(
        model=resolve_ollama_model(model),
        messages=[message],
    )
    response_text = response.get("message", {}).get("content", "").strip()
    if not response_text:
        raise RuntimeError("Ollama rank response did not include any content")
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama rank response not valid JSON") from exc
    return Rank(rank=int(parsed.get("rank", -1)), confidence=float(parsed.get("confidence", 0.0)))
