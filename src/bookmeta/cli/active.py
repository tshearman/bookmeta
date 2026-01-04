from __future__ import annotations

import argparse
import logging
import runpy
import sys
from pathlib import Path

from bookmeta.config.pipeline import load_openai_provider_config
from bookmeta.pipelines.active import PipelineConfig, run_pipeline


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() != ".py":
        raise ValueError(f"Unsupported config type for {config_path}. Use a .py file.")
    module_globals = runpy.run_path(str(config_path))
    config_obj = module_globals.get("CONFIG")
    if isinstance(config_obj, PipelineConfig):
        return config_obj
    raise ValueError(f"{config_path} must define CONFIG as a PipelineConfig instance.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bookmeta active pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("tmp/bookmeta_active_tmp.py"),
        help="Path to pipeline config (.py) defining CONFIG = PipelineConfig.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        help="Override roots with one or more source directories.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Override results database path.",
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
        "--collect-results",
        dest="collect_results",
        action="store_true",
        default=False,
        help="Collect BookInfoResult objects in memory (off by default to save RAM).",
    )
    parser.add_argument(
        "--writer-bin",
        type=Path,
        help="Path to pdf-metadata writer binary. If provided, writes augmented PDFs.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        help="Destination directory for augmented PDFs (required if --writer-bin is set).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume using cached LLM results to skip repeat calls.",
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
        cfg.roots = tuple(args.source)
    if args.db:
        cfg.results_db = args.db
    cfg.monitor_queues = args.monitor_queues
    cfg.collect_results = args.collect_results
    if args.writer_bin:
        cfg.writer_bin = args.writer_bin
        if not args.dest_dir:
            raise SystemExit("--dest-dir is required when --writer-bin is provided")
        cfg.output_dir = args.dest_dir
    cfg.resume = args.resume

    if args.secrets:
        cfg.provider_config = load_openai_provider_config(args.secrets)

    results = run_pipeline(cfg)
    print(
        f"Completed active pipeline for {cfg.roots}; results stored at {cfg.results_db}"
    )
    if cfg.collect_results:
        print(f"Collected {len(results)} BookInfoDetailedResult objects.")
        for item in results:
            print(f"\nPDF: {item.pdf.path}")
            print(item.detailed.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
