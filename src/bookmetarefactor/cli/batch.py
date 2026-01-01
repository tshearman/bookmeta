from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bookmetarefactor.pipelines.batch import PipelineConfig, load_openai_provider_config, run_pipeline


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix == ".py":
        return PipelineConfig.from_pyfile(config_path)
    raise ValueError(f"Unsupported config type for {config_path}. Use a .py config file.")


def summarize_results(cfg: PipelineConfig, bookinfo_results, persisted_batches) -> str:
    rt_cfg = cfg.runtime
    if rt_cfg.mode == "llm":
        return f"Completed LLM mode with {len(bookinfo_results)} bookinfo results."
    batch_paths = [batch.path for batch in persisted_batches]
    lines = [
        f"Completed batch mode with {len(persisted_batches)} persisted batches:",
        *(f"- {path}" for path in batch_paths),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bookmetarefactor batch/LLM pipeline."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("resources/batch_config.py"),
        help="Path to pipeline config (.py or .yaml). Default: resources/batch_config.py",
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "batch"],
        help="Override pipeline mode (llm|batch). Overrides config.runtime.mode.",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        help="Path to secrets.json for provider configuration (e.g., OpenAI keys).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = load_config(args.config)

    if args.mode:
        cfg.runtime.mode = args.mode  # type: ignore[assignment]

    if args.secrets:
        cfg.pdf.provider_config = load_openai_provider_config(args.secrets)

    bookinfo_results, persisted_batches = run_pipeline(cfg)
    print(summarize_results(cfg, bookinfo_results, persisted_batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
