import fitz
from bookmetarefactor.types import Pdf


def extract_metadata(pdf: Pdf):
    """Return sanitized metadata for the given PDF."""
    with fitz.open(pdf.path) as doc:
        meta = doc.metadata or {}
    return meta
