import warnings
from threading import Lock

import pytesseract
from joblib import Memory

from bookmeta.config import CACHE_ROOT
from bookmeta.types import Page
from bookmeta.types.ocr import *
from bookmeta.utils.page import page_to_image

_NATIVE_OCR_LOCK = Lock()


def native_ocr_method(page: Page) -> OcrOutput:
    with _NATIVE_OCR_LOCK:
        return page.get_textpage_ocr().extractText()


OCR_CACHE_DIR = CACHE_ROOT / "ocr_calls"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message=r"Persisting input arguments took .*s to run",
    category=UserWarning,
)
OCR_MEMORY = Memory(OCR_CACHE_DIR, verbose=0)


@OCR_MEMORY.cache()
def cached_image_to_string(img) -> str:
    return pytesseract.image_to_string(img, lang="eng")


def tesseract_ocr_method(page: Page) -> OcrOutput:
    img = page_to_image(page, grayscale=True)
    return cached_image_to_string(img)


NATIVE_OCR_METHOD = OcrMethod(name="native", process=native_ocr_method)
TESSERACT_OCR_METHOD = OcrMethod(name="tesseract", process=tesseract_ocr_method)
DEFAULT_OCR_METHODS: tuple[OcrMethod, ...] = (NATIVE_OCR_METHOD, TESSERACT_OCR_METHOD)
