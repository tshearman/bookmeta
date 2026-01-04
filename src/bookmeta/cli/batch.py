from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bookmeta.config.pipeline import load_openai_provider_config
from bookmeta.pipelines.batch import PipelineConfig, run_pipeline


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix == ".py":
        return PipelineConfig.from_pyfile(config_path)
    raise ValueError(
        f"Unsupported config type for {config_path}. Use a .py config file."
    )


def summarize_results(cfg: PipelineConfig, persisted_batches) -> str:
    batch_paths = [batch.path for batch in persisted_batches]
    lines = [
        f"Completed batch mode with {len(persisted_batches)} persisted batches:",
        *(f"- {path}" for path in batch_paths),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bookmeta batch pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("resources/batch_config.py"),
        help="Path to pipeline config (.py or .yaml). Default: resources/batch_config.py",
    )
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        help="Override runtime roots with one or more source directories.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Override runtime batch output directory.",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        help="Path to secrets.json for provider configuration (e.g., OpenAI keys).",
    )
    parser.add_argument(
        "--no-queue-monitor",
        dest="monitor_queues",
        action="store_false",
        default=True,
        help="Disable live queue monitoring.",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=None,
        help="Path to the SQLite DB used to record batch runs (default: resources/bookmeta.db).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip PDFs already recorded in previous batch runs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for the pipeline.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    cfg = load_config(args.config)

    if args.source:
        cfg.runtime.roots = tuple(args.source)
    if args.dest:
        cfg.runtime.batch_output_dir = args.dest
    cfg.runtime.monitor_queues = args.monitor_queues

    if args.secrets:
        cfg.pdf.provider_config = load_openai_provider_config(args.secrets)

    if args.results_db:
        cfg.runtime.results_db = args.results_db

    if args.resume:
        cfg.runtime.resume = True

    persisted_batches = run_pipeline(cfg)
    print(summarize_results(cfg, persisted_batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
