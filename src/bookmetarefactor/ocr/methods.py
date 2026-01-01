import ollama
from openai import OpenAI
import pytesseract
from bookmetarefactor.config import TTRPG_OCR_LLM_PROMPT, MAX_LONG_EDGE_IMG
from bookmetarefactor.types import Page
from bookmetarefactor.types.ocr import *
from bookmetarefactor.utils import sanitize
from bookmetarefactor.utils.img import img_to_b64, img_to_url
from bookmetarefactor.utils.page import page_to_image


def native_ocr_method(page: Page) -> OcrOutput:
    return page.get_textpage_ocr().extractText()


def tesseract_ocr_method(page: Page) -> OcrOutput:
    img = page_to_image(page, grayscale=True)
    ocr_text: str = pytesseract.image_to_string(img, lang="eng")
    return ocr_text


def ollama_ocr_method(client: ollama.Client, model) -> OcrMethod:

    def process(page: Page) -> OcrOutput:
        img = page_to_image(page, grayscale=True, max_long_edge=MAX_LONG_EDGE_IMG)
        b64_img = img_to_b64(img)
        messages = [
            ollama.Message(
                role="user",
                content=TTRPG_OCR_LLM_PROMPT,
                images=[ollama.Image(value=b64_img)],
            )
        ]

        response = client.chat(model=model, messages=messages)
        return response.message.content

    return OcrMethod(f"ollama:{model}", process)


def openai_ocr_method(client: OpenAI, model) -> OcrMethod:

    def process(page: Page) -> OcrOutput:
        img = page_to_image(page, grayscale=True, max_long_edge=MAX_LONG_EDGE_IMG)
        img_url = img_to_url(img)
        img_block = {"type": "input_image", "image_url": img_url}
        prompt_block = {"type": "input_text", "text": TTRPG_OCR_LLM_PROMPT}
        input = sanitize([{"role": "user", "content": [prompt_block, img_block]}])
        return client.responses.parse(model=model, input=input).output_text

    return OcrMethod(f"openai:{model}", process)


NATIVE_OCR_METHOD = OcrMethod(name="native", process=native_ocr_method)
TESSERACT_OCR_METHOD = OcrMethod(name="tesseract", process=tesseract_ocr_method)
DEFAULT_OCR_METHODS: tuple[OcrMethod, ...] = (NATIVE_OCR_METHOD, TESSERACT_OCR_METHOD)
