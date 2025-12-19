import shutil
from pathlib import Path

import pytest

from bookmeta.cli.batch import _count_pdfs_with_ripgrep

RG_MISSING = shutil.which("rg") is None


@pytest.mark.skipif(RG_MISSING, reason="ripgrep required")
def test_counts_pdfs(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").touch()

    assert _count_pdfs_with_ripgrep(tmp_path) == 2


def test_raises_for_missing_dir(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    with pytest.raises(RuntimeError):
        _count_pdfs_with_ripgrep(missing_dir)
