import hashlib
import json
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Iterable, Literal, Mapping

from bookmetarefactor.ocr import OcrConfig
from bookmetarefactor.pipelines import (
    produce_pdfs,
    start_extraction_pipeline,
    start_llm_batch_pipeline,
    start_llm_pipeline,
    start_ocr_pipeline,
    start_persist_batch_pipeline,
)
from bookmetarefactor.types.bookinfo import BookInfoResult
from bookmetarefactor.types.extraction import (
    ContextLimits,
    ExtractionConfig,
    PersistedBatch,
    ProviderConfig,
)
from bookmetarefactor.types.ocr import OcrMethod


@dataclass
class PdfProcessingConfig:
    prompt: str
    provider_config: ProviderConfig
    ocr_config: OcrConfig = field(default_factory=OcrConfig)
    context_limits: ContextLimits = field(default_factory=ContextLimits)


@dataclass
class PipelineRuntimeConfig:
    roots: Iterable[Path | str]
    queue_size: int | None = None
    pdf_queue_size: int | None = None
    extraction_queue_size: int | None = None
    llm_queue_size: int | None = None
    bookinfo_queue_size: int | None = None
    batch_queue_size: int | None = None
    persist_queue_size: int | None = None
    dedupe: bool = True
    limit: int | None = None
    ocr_workers: int = 32
    extraction_workers: int = 6
    llm_workers: int = 6
    stage_timeout: float | None = None
    mode: Literal["llm", "batch"] = "llm"
    batch_output_dir: Path = Path("batches")


@dataclass
class PipelineConfig:
    pdf: PdfProcessingConfig
    runtime: PipelineRuntimeConfig

    @staticmethod
    def _methods_from_names(method_names: Iterable[str]) -> tuple[OcrMethod, ...]:
        from openai import OpenAI

        import ollama
        from bookmetarefactor.ocr.methods import (
            NATIVE_OCR_METHOD,
            TESSERACT_OCR_METHOD,
            ollama_ocr_method,
            openai_ocr_method,
        )

        resolved: list[OcrMethod] = []
        for name in method_names:
            if name == "native":
                resolved.append(NATIVE_OCR_METHOD)
            elif name == "tesseract":
                resolved.append(TESSERACT_OCR_METHOD)
            elif name.startswith("openai:"):
                model = name.split(":", 1)[1]
                resolved.append(openai_ocr_method(OpenAI(), model))
            elif name.startswith("ollama:"):
                model = name.split(":", 1)[1]
                resolved.append(ollama_ocr_method(ollama.Client(), model))
            else:
                raise ValueError(
                    "Unsupported OCR method "
                    f"'{name}'. Use native, tesseract, openai:<model>, or ollama:<model>."
                )
        return tuple(resolved)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        pdf_section = data.get("pdf") or data.get("pdf_config") or {}
        runtime_section = data.get("runtime") or data.get("runtime_config") or {}

        fallback = data  # allow flat configs for backward compatibility

        prompt = pdf_section.get("prompt", fallback.get("prompt"))
        provider_raw = pdf_section.get(
            "provider_config", fallback.get("provider_config")
        )
        if prompt is None or provider_raw is None:
            raise ValueError("prompt and provider_config are required for PDF config")

        provider_config = ProviderConfig(
            provider=provider_raw["provider"],
            model=provider_raw["model"],
            client_config=provider_raw.get("client_config", {}),
        )

        default_context_limits = ContextLimits()
        context_limits_raw = pdf_section.get(
            "context_limits", fallback.get("context_limits", {})
        ) or {}
        context_limits = ContextLimits(
            num_first_images=context_limits_raw.get(
                "num_first_images", default_context_limits.num_first_images
            ),
            num_last_images=context_limits_raw.get(
                "num_last_images", default_context_limits.num_last_images
            ),
            num_first_ocr_pages=context_limits_raw.get(
                "num_first_ocr_pages", default_context_limits.num_first_ocr_pages
            ),
            num_last_ocr_pages=context_limits_raw.get(
                "num_last_ocr_pages", default_context_limits.num_last_ocr_pages
            ),
        )

        default_ocr_config = OcrConfig()
        ocr_raw = pdf_section.get("ocr_config", fallback.get("ocr_config", {})) or {}
        ocr_method_names = ocr_raw.get("methods")
        if ocr_method_names is None:
            ocr_method_names = [method.name for method in default_ocr_config.methods]
        ocr_methods = cls._methods_from_names(ocr_method_names)
        ocr_config = OcrConfig(
            num_first_pages=ocr_raw.get(
                "num_first_pages", default_ocr_config.num_first_pages
            ),
            num_last_pages=ocr_raw.get(
                "num_last_pages", default_ocr_config.num_last_pages
            ),
            methods=ocr_methods,
        )

        pdf_config = PdfProcessingConfig(
            prompt=prompt,
            provider_config=provider_config,
            ocr_config=ocr_config,
            context_limits=context_limits,
        )

        roots_raw = runtime_section.get("roots", fallback.get("roots"))
        if roots_raw is None:
            raise ValueError("roots is required for runtime config")

        runtime_config = PipelineRuntimeConfig(
            roots=tuple(Path(root) for root in roots_raw),
            queue_size=runtime_section.get("queue_size", fallback.get("queue_size")),
            pdf_queue_size=runtime_section.get(
                "pdf_queue_size", fallback.get("pdf_queue_size")
            ),
            extraction_queue_size=runtime_section.get(
                "extraction_queue_size", fallback.get("extraction_queue_size")
            ),
            llm_queue_size=runtime_section.get(
                "llm_queue_size", fallback.get("llm_queue_size")
            ),
            bookinfo_queue_size=runtime_section.get(
                "bookinfo_queue_size", fallback.get("bookinfo_queue_size")
            ),
            batch_queue_size=runtime_section.get(
                "batch_queue_size", fallback.get("batch_queue_size")
            ),
            persist_queue_size=runtime_section.get(
                "persist_queue_size", fallback.get("persist_queue_size")
            ),
            dedupe=runtime_section.get("dedupe", fallback.get("dedupe", True)),
            limit=runtime_section.get("limit", fallback.get("limit")),
            ocr_workers=runtime_section.get(
                "ocr_workers", fallback.get("ocr_workers", 32)
            ),
            extraction_workers=runtime_section.get(
                "extraction_workers", fallback.get("extraction_workers", 6)
            ),
            llm_workers=runtime_section.get(
                "llm_workers", fallback.get("llm_workers", 6)
            ),
            stage_timeout=runtime_section.get(
                "stage_timeout", fallback.get("stage_timeout")
            ),
            mode=runtime_section.get("mode", fallback.get("mode", "llm")),
            batch_output_dir=Path(
                runtime_section.get(
                    "batch_output_dir", fallback.get("batch_output_dir", "batches")
                )
            ),
        )

        return cls(pdf=pdf_config, runtime=runtime_config)

    @classmethod
    @classmethod
    def from_pyfile(cls, path: str | Path, config_var: str = "CONFIG") -> "PipelineConfig":
        module_globals = runpy.run_path(str(path))
        if config_var not in module_globals:
            raise ValueError(
                f"Config variable '{config_var}' not found in {path}. "
                "Define CONFIG to point to a mapping."
            )
        config_mapping = module_globals[config_var]
        if isinstance(config_mapping, PipelineConfig):
            return config_mapping
        if isinstance(config_mapping, Mapping):
            return cls.from_mapping(config_mapping)
        raise ValueError(
            f"{config_var} in {path} must be a mapping or PipelineConfig, "
            f"got {type(config_mapping)}."
        )


def load_openai_provider_config(secrets_path: Path) -> ProviderConfig:
    """Load OpenAI provider settings from a secrets.json file."""
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    api_key = secrets.get("OPENAI_API_KEY")
    project = secrets.get("OPENAI_PROJECT_ID")
    if not api_key:
        raise ValueError(f"Missing OPENAI_API_KEY in {secrets_path}")
    client_config = {"api_key": api_key}
    if project:
        client_config["project"] = project
    return ProviderConfig(
        provider="openai", model="gpt-4.1", client_config=client_config
    )


def _pipeline_hash(
    cfg: PipelineConfig, extraction_config: ExtractionConfig, ocr_config: OcrConfig
) -> str:
    payload = {
        "extraction_hash": extraction_config.hash,
        "ocr": {
            "num_first_pages": ocr_config.num_first_pages,
            "num_last_pages": ocr_config.num_last_pages,
            "methods": sorted(method.name for method in ocr_config.methods),
        },
        "roots": sorted(str(Path(root)) for root in cfg.runtime.roots),
        "dedupe": cfg.runtime.dedupe,
        "limit": cfg.runtime.limit,
        "mode": cfg.runtime.mode,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def run_pipeline(
    cfg: PipelineConfig,
) -> tuple[list[BookInfoResult], list[PersistedBatch]]:
    """
    Orchestrate the full pipeline from PDF discovery through LLM execution or batching.

    Returns (bookinfo_results, persisted_batches). Only one of the outputs will be populated
    depending on cfg.mode ("llm" for immediate execution, "batch" for offline submission).
    """

    provider_config = cfg.pdf.provider_config

    ocr_methods = tuple(cfg.pdf.ocr_config.methods)
    ocr_config = OcrConfig(
        num_first_pages=cfg.pdf.ocr_config.num_first_pages,
        num_last_pages=cfg.pdf.ocr_config.num_last_pages,
        methods=ocr_methods,
    )

    extraction_config = ExtractionConfig(
        prompt=cfg.pdf.prompt,
        provider_config=provider_config,
        context_limits=cfg.pdf.context_limits,
    )
    pipeline_hash = _pipeline_hash(cfg, extraction_config, ocr_config)

    rt_cfg = cfg.runtime

    def _queue_size(preferred: int | None) -> int:
        size = preferred if preferred is not None else rt_cfg.queue_size
        return 0 if size is None else size

    pdf_queue: Queue = Queue(maxsize=_queue_size(rt_cfg.pdf_queue_size))
    extraction_queue: Queue = Queue(maxsize=_queue_size(rt_cfg.extraction_queue_size))
    llm_queue: Queue = Queue(maxsize=_queue_size(rt_cfg.llm_queue_size))
    bookinfo_queue: Queue | None = None
    batch_queue: Queue | None = None
    persisted_queue: Queue | None = None

    producer_done = Event()
    ocr_done = Event()
    extraction_done = Event()
    batch_thread: Thread | None = None
    persist_thread: Thread | None = None
    batch_done = Event()
    llm_threads: list[Thread] = []

    def _produce() -> None:
        produce_pdfs(
            rt_cfg.roots, pdf_queue, dedupe=rt_cfg.dedupe, limit=rt_cfg.limit
        )
        producer_done.set()

    producer_thread = Thread(target=_produce, daemon=True)
    producer_thread.start()

    ocr_threads = start_ocr_pipeline(
        pdf_queue,
        extraction_queue,
        ocr_config,
        extraction_config,
        workers=rt_cfg.ocr_workers,
        upstream_done=producer_done,
        timeout=rt_cfg.stage_timeout,
    )

    extraction_threads = start_extraction_pipeline(
        extraction_queue,
        llm_queue,
        workers=rt_cfg.extraction_workers,
        upstream_done=ocr_done,
        timeout=rt_cfg.stage_timeout,
    )

    if rt_cfg.mode == "llm":
        bookinfo_queue = Queue(maxsize=_queue_size(rt_cfg.bookinfo_queue_size))
        llm_threads = start_llm_pipeline(
            llm_queue,
            bookinfo_queue,
            workers=rt_cfg.llm_workers,
            upstream_done=extraction_done,
            timeout=rt_cfg.stage_timeout,
        )
    elif rt_cfg.mode == "batch":
        batch_queue = Queue(maxsize=_queue_size(rt_cfg.batch_queue_size))
        batch_thread = start_llm_batch_pipeline(
            llm_queue,
            batch_queue,
            upstream_done=extraction_done,
            timeout=rt_cfg.stage_timeout,
        )
        persisted_queue = Queue(maxsize=_queue_size(rt_cfg.persist_queue_size))
        persist_thread = start_persist_batch_pipeline(
            batch_queue,
            persisted_queue,
            rt_cfg.batch_output_dir,
            pipeline_hash,
            upstream_done=batch_done,
            timeout=rt_cfg.stage_timeout,
        )
    else:
        raise ValueError(f"Unsupported pipeline mode: {rt_cfg.mode}")

    # Producer Stage
    producer_thread.join()

    # OCR Stages
    pdf_queue.join()
    for thread in ocr_threads:
        thread.join()
    ocr_done.set()

    # Extraction Stage
    extraction_queue.join()
    for thread in extraction_threads:
        thread.join()
    extraction_done.set()

    # LLM / Batching Stage
    llm_queue.join()
    if rt_cfg.mode == "llm":
        for thread in llm_threads:
            thread.join()
    else:
        assert batch_thread is not None
        batch_thread.join()
        batch_done.set()
        assert batch_queue is not None
        batch_queue.join()
        assert persist_thread is not None
        persist_thread.join()

    # Collect outputs
    bookinfo_results: list[BookInfoResult] = []
    persisted_batches: list[PersistedBatch] = []

    if rt_cfg.mode == "llm":
        assert bookinfo_queue is not None
        while not bookinfo_queue.empty():
            item = bookinfo_queue.get()
            if item is not None:
                bookinfo_results.append(item)
            bookinfo_queue.task_done()
        bookinfo_queue.join()
    else:
        assert persisted_queue is not None
        while not persisted_queue.empty():
            batch = persisted_queue.get()
            if batch:
                persisted_batches.append(batch)
            persisted_queue.task_done()
        persisted_queue.join()

    return bookinfo_results, persisted_batches
