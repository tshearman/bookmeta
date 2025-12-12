import base64
import json
import logging
import mimetypes
import tempfile
from pathlib import Path

from joblib import Memory
from openai import OpenAI
from pydantic.dataclasses import dataclass

from google_books import GoogleBooksQuery
from pdf_processor import PdfProcessingResult, process_pdf_for_openai_inputs
from ollama_client import (
    get_ollama_client,
    prepare_ollama_images,
    resolve_ollama_model,
)


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache/bookmeta"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(location=str(CACHE_DIR), verbose=0)


@dataclass
class BookInfo:
    author: str
    author_confidence: float
    title: str
    title_confidence: float
    publisher: str | None
    publisher_confidence: float
    subject: str | None
    keywords: list[str] | None
    isbn_identifiers: list[str] | None = None
    isbn_confidence: float = 0.0
    model_ocr: str | None = None


def image_path_to_data_url(path: str) -> str:
    """Read a local image and return a data: URL suitable for `input_image`."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


BOOK_PROMPT = """
You are given images of the first few pages of a book (front cover, back cover,
and early interior pages) plus two OCR transcripts labeled "NATIVE OCR" 
and "TESSERACT OCR". Using only information visible in these sources:

• Extract the book's title and primary author (highest priority).
• Provide confidence values between 0 and 1 for title_confidence and
  author_confidence, where 0 means very uncertain / likely wrong and 1 means
  certain.
• If present, also extract:
  - publisher name
  - subject/category (e.g. “Fiction”, “Role Playing Games”, etc.)
  - any ISBN identifiers (10 or 13 digits) and list them in `isbn_identifiers`
• Populate the tags field with a list of strings, up to but no more
  than 8. These should be tags that describe information about the book
  like genre, subject, if its a game, and other high-level metadata
• Provide `model_ocr`: a concise transcription/summary of the readable text you
  can infer from the images and OCR excerpts (do not just echo the provided OCR
  but use the provided OCR as context).

Return your best guess for any field you can see. If a field is not visible or
uncertain, set it to null (and confidence near 0). Do NOT invent information.
"""


def construct_content_blocks(
    pdf_result: PdfProcessingResult,
    context_path: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the multimodal content blocks (prompt + page images + OCR text snippets).
    """
    content_blocks: list[dict[str, str]] = [
        {"type": "input_text", "text": BOOK_PROMPT},
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
            {
                "type": "input_image",
                "image_url": image_path_to_data_url(page.image_path),
            }
        )

    native_text = pdf_result.combined_text_for("native_ocr")
    if native_text:
        content_blocks.append(
            {
                "type": "input_text",
                "text": f"NATIVE OCR\n{native_text}",
            }
        )

    tesseract_text = pdf_result.combined_text_for("tesseract_ocr")
    if tesseract_text:
        content_blocks.append(
            {
                "type": "input_text",
                "text": f"TESSERACT OCR\n{tesseract_text}",
            }
        )

    return content_blocks


@memory.cache(ignore=["client"])
def _openai_bookinfo_request(content_blocks, model, client):
    logging.debug("::: EXECUTING OPENAI API REQUEST :::")
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
        text_format=BookInfo,
    )
    return response.output_parsed


@memory.cache()
def _ollama_bookinfo_request(prompt_text: str, images: list[str], model: str) -> BookInfo:
    logging.debug("::: EXECUTING OLLAMA BOOKINFO REQUEST :::")
    schema_text = json.dumps(
        {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "author_confidence": {"type": "number"},
                "title": {"type": "string"},
                "title_confidence": {"type": "number"},
                "publisher": {"type": ["string", "null"]},
                "publisher_confidence": {"type": "number"},
                "subject": {"type": ["string", "null"]},
                "keywords": {"type": ["array", "null"], "items": {"type": "string"}},
                "isbn_identifiers": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "isbn_confidence": {"type": "number"},
                "model_ocr": {"type": ["string", "null"]},
            },
            "required": [
                "author",
                "author_confidence",
                "title",
                "title_confidence",
                "publisher_confidence",
                "isbn_confidence",
            ],
        }
    )
    prompt = (
        f"{prompt_text}\n\nSchema:\n{schema_text}\n"
        "Respond ONLY with JSON matching this schema."
    )
    message: dict = {"role": "user", "content": prompt}
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
        raise RuntimeError("Ollama response did not include any content")
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama response not valid JSON") from exc
    return BookInfo(**parsed)


def _content_blocks_to_text_and_images(content_blocks: list[dict[str, str]]):
    texts: list[str] = []
    images: list[str] = []
    for block in content_blocks:
        if block["type"] == "input_text":
            texts.append(block["text"])
        elif block["type"] == "input_image":
            images.append(block["image_url"])
    return "\n\n".join(texts), images


def extract_bookinfo_via_model(
    pdf_result: PdfProcessingResult,
    client: OpenAI | None,
    model: str = "gpt-4.1-mini",
    context_path: str | None = None,
    provider: str = "openai",
) -> BookInfo:
    """
    Build an OpenAI request from the rendered pages + OCR output of a PDF.
    """
    content_blocks = construct_content_blocks(pdf_result, context_path=context_path)
    if provider == "ollama":
        prompt_text, images = _content_blocks_to_text_and_images(content_blocks)
        return _ollama_bookinfo_request(prompt_text, images, model)
    if client is None:
        raise ValueError("OpenAI client is required when provider='openai'")
    return _openai_bookinfo_request(content_blocks, model, client)


def bookinfo_to_google_books_query(book: BookInfo) -> GoogleBooksQuery:
    general_parts = [book.title, book.author, book.subject]
    generalquery = " ".join(part for part in general_parts if part).strip() or "book"
    isbn_value = None
    if book.isbn_identifiers:
        for ident in book.isbn_identifiers:
            if not ident:
                continue
            cleaned = "".join(ch for ch in ident.upper() if ch.isdigit() or ch == "X")
            if len(cleaned) in (10, 13):
                isbn_value = cleaned
                break

    return GoogleBooksQuery(
        generalquery=generalquery,
        inauthor=book.author or "",
        author_confidence=book.author_confidence or 0.0,
        intitle=book.title or None,
        title_confidence=book.title_confidence or 0.0,
        inpublisher=book.publisher,
        subject=book.subject,
        isbn=isbn_value,
        isbn_confidence=book.isbn_confidence if isbn_value else 0.0,
        tags=book.keywords,
    )


def extract_google_books_query_from_pdf_result(
    pdf_result: PdfProcessingResult,
    client: OpenAI | None,
    model: str = "gpt-4.1-mini",
    context_path: str | None = None,
    provider: str = "openai",
) -> GoogleBooksQuery:
    """
    Convenience wrapper: run the BookInfo extraction and convert to GoogleBooksQuery.
    """
    book = extract_bookinfo_via_model(
        pdf_result=pdf_result,
        client=client,
        model=model,
        context_path=context_path,
        provider=provider,
    )
    return bookinfo_to_google_books_query(book)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    sample_pdf = Path("test_pdfs/bladesinthedark_v8_2.pdf")
    with open("./secrets.json", "r") as f:
        data = json.load(f)
        OPENAI_API_KEY = data["OPENAI_API_KEY"]
        OPENAI_PROJECT_ID = data["OPENAI_PROJECT_ID"]

    client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)

    with tempfile.TemporaryDirectory(prefix="pdf_pages_") as tmpdir:
        pdf_result = process_pdf_for_openai_inputs(
            pdf_path=sample_pdf,
            output_dir=tmpdir,
            max_long_edge=1200,
        )

        content_blocks = construct_content_blocks(pdf_result)

        logging.debug(f"Constructed {len(content_blocks)} content blocks")
        for idx, block in enumerate(content_blocks, start=1):
            block_type = block.get("type")
            if block_type == "input_image":
                image_url = block.get("image_url", "")
                logging.debug(
                    f"[{idx}] input_image - data length={len(image_url)} "
                    f"(prefix {image_url[:32]!r})"
                )
            elif block_type == "input_text":
                text = block.get("text", "")
                logging.debug(f"[{idx}] input_text - preview: {text[:120]!r}")
            else:
                logging.debug(f"[{idx}] block: {block}")

        result = extract_bookinfo_via_model(pdf_result, client)
        logging.debug(result)
