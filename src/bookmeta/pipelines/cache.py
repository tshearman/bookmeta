import json
from pathlib import Path

from bookmeta.config import CACHE_ROOT
from bookmeta.types import Pdf
from bookmeta.types.bookinfo import BookInfoResponse, BookInfoResult

LLM_CACHE_DIR = CACHE_ROOT / "llm_results"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _llm_cache_path(pdf: Pdf) -> Path:
    return LLM_CACHE_DIR / f"{pdf.hash}.json"


def load_cached_bookinfo(pdf: Pdf) -> BookInfoResult | None:
    path = _llm_cache_path(pdf)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        response = BookInfoResponse.model_validate(data)
        return BookInfoResult(pdf, response)
    except Exception:
        return None


def save_cached_bookinfo(result: BookInfoResult) -> None:
    path = _llm_cache_path(result.pdf)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.bookinfo.model_dump()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
