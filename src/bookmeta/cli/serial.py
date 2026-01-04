from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bookmeta.cli.active import load_config
from bookmeta.config.pipeline import (
    load_openai_provider_config,
)
from bookmeta.pipelines.serial import run_serial_pipeline


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the serial BookMeta pipeline for a single PDF."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF to process.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("tmp/bookmeta_active_tmp.py"),
        help="Path to pipeline config (.py) defining CONFIG = PipelineConfig.",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        required=True,
        help="Path to secrets.json for provider configuration (e.g., OpenAI keys).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Override results database path.",
    )
    parser.add_argument(
        "--writer-bin",
        type=Path,
        help="Path to pdf-metadata writer binary. If provided, an augmented PDF will be written.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Destination PDF path for augmented output (required if --writer-bin is set).",
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
    if args.db:
        cfg.results_db = args.db
    cfg.provider_config = load_openai_provider_config(args.secrets)
    if args.writer_bin and not args.dest:
        raise SystemExit("--dest is required when --writer-bin is provided")
    result = run_serial_pipeline(
        args.pdf,
        ocr_config=cfg.ocr_config,
        extraction_config=cfg.extraction_config,
        writer_bin=args.writer_bin,
        destination_pdf=args.dest,
    )
    if result is None:
        print("No result produced.")
        return 1
    print(result.detailed.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
