from pathlib import Path
from typing import Iterable


def discover_pdfs(root: Path) -> Iterable[Path]:
    """Yield PDF files beneath the given root directory."""
    if not root.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"PDF directory is not a directory: {root}")
    yield from (path for path in root.rglob("*.pdf") if path.is_file())
