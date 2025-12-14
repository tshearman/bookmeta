from typing import Callable, cast

import fitz
import ollama
from openai import OpenAI
import pytesseract
from datamodel.img_ocr_result import OcrResult
from llm import (
    cached_ollama_chat,
    cached_openapi_response_text,
    cached_pytesseract_image_to_string,
)
from ocr import OCR_LLM_PROMPT
from .rendering import img_to_b64, img_to_url, page_to_image

OcrMethod = Callable[[fitz.Page], OcrResult | None]


def native_ocr_method(page: fitz.Page) -> OcrResult | None:
    native_text = cast(str, page.get_text("text") or "").strip()
    if native_text:
        return OcrResult(method="native_ocr", text=native_text)


def tesseract_ocr_method(page: fitz.Page) -> OcrResult | None:
    lang: str = "eng"
    img = page_to_image(page, grayscale=True)
    ocr_text = (cached_pytesseract_image_to_string(img, lang) or "").strip()
    if ocr_text:
        return OcrResult(method="tesseract_ocr", text=ocr_text)


def ollama_ocr_method(client: ollama.Client, model) -> OcrMethod:

    def run(page: fitz.Page) -> OcrResult | None:
        img = page_to_image(page, grayscale=True, max_long_edge=1200)

        messages = [
            ollama.Message(
                role="user",
                content=OCR_LLM_PROMPT,
                images=[ollama.Image(value=img_to_b64(img))],
            )
        ]

        response = cached_ollama_chat(model, messages, client)
        content = response["message"]["content"]
        return OcrResult(method=f"ollama:{model}", text=content)

    return run


def openai_ocr_method(client: OpenAI, model) -> OcrMethod:

    def run(page: fitz.Page) -> OcrResult | None:
        img = page_to_image(page, grayscale=True, max_long_edge=1200)
        img_block = {"type": "input_image", "image_url": img_to_url(img)}
        prompt_block = {"type": "input_text", "text": OCR_LLM_PROMPT}
        context = [{"role": "user", "content": [prompt_block, img_block]}]
        content = cached_openapi_response_text(model, context, client)
        return OcrResult(method=f"openai:{model}", text=content)

    return run
