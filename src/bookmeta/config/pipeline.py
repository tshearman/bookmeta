import json
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from bookmeta.ocr import OcrConfig
from bookmeta.types.extraction import (
    ClientConfig,
    ContextLimits,
    ExtractionConfig,
    ProviderConfig,
)
from bookmeta.types.ocr import OcrMethod


@dataclass
class PdfProcessingConfig:
    prompt: str
    provider_config: ProviderConfig
    ocr_config: OcrConfig = field(default_factory=OcrConfig)
    context_limits: ContextLimits = field(default_factory=ContextLimits)

    @property
    def extraction_config(self):
        return ExtractionConfig(self.prompt, self.provider_config, self.context_limits)


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
    stage_timeout: float = 0.5
    batch_output_dir: Path = Path("batches")
    monitor_queues: bool = True
    results_db: Path = Path("resources/bookmeta.db")
    resume: bool = False


@dataclass
class PipelineConfig:
    pdf: PdfProcessingConfig
    runtime: PipelineRuntimeConfig

    @staticmethod
    def _methods_from_names(method_names: Iterable[str]) -> tuple[OcrMethod, ...]:
        from bookmeta.ocr.methods import NATIVE_OCR_METHOD, TESSERACT_OCR_METHOD

        resolved: list[OcrMethod] = []
        for name in method_names:
            if name == "native":
                resolved.append(NATIVE_OCR_METHOD)
            elif name == "tesseract":
                resolved.append(TESSERACT_OCR_METHOD)
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
        context_limits_raw = (
            pdf_section.get("context_limits", fallback.get("context_limits", {})) or {}
        )
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
                "stage_timeout", fallback.get("stage_timeout", 0.5)
            ),
            batch_output_dir=Path(
                runtime_section.get(
                    "batch_output_dir", fallback.get("batch_output_dir", "batches")
                )
            ),
            monitor_queues=runtime_section.get(
                "monitor_queues", fallback.get("monitor_queues", True)
            ),
            results_db=Path(
                runtime_section.get(
                    "results_db",
                    fallback.get("results_db", "resources/bookmeta.db"),
                )
            ),
            resume=runtime_section.get("resume", fallback.get("resume", False)),
        )

        return cls(pdf=pdf_config, runtime=runtime_config)

    @classmethod
    def from_pyfile(
        cls, path: str | Path, config_var: str = "CONFIG"
    ) -> "PipelineConfig":
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


def openai_client_config(secrets_path: Path) -> ClientConfig:
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    api_key = secrets.get("OPENAI_API_KEY")
    project = secrets.get("OPENAI_PROJECT_ID")
    client_config = {"api_key": api_key}
    if project:
        client_config["project"] = project
    return client_config


def load_openai_provider_config(secrets_path: Path) -> ProviderConfig:
    """Load OpenAI provider settings from a secrets.json file."""
    return ProviderConfig(
        provider="openai",
        model="gpt-5-mini",
        client_config=openai_client_config(secrets_path),
    )


__all__ = [
    "PdfProcessingConfig",
    "PipelineRuntimeConfig",
    "PipelineConfig",
    "load_openai_provider_config",
]
