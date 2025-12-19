import logging
from pathlib import Path
from typing import Dict

import fitz

LOGGER = logging.getLogger("ocr.metadata")


def load_pdf_metadata(pdf_path: str | Path):
    """Return sanitized metadata for the given PDF."""
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        meta = doc.metadata or {}
        LOGGER.debug(f"Raw PDF metadata for {path}: {meta}")
    return meta


def main() -> None:
    import argparse
    import json

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="Inspect metadata embedded in a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to inspect.")
    args = parser.parse_args()

    metadata = load_pdf_metadata(args.pdf_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
