from pathlib import Path

import fitz

from book_info_extractor import BookInfo


def _metadata_from_bookinfo(book: BookInfo) -> dict[str, str]:
    return {
        "title": book.title or "",
        "author": book.author or "",
        "subject": book.subject or "",
        "keywords": ", ".join(book.keywords or []),
    }


def augment_pdf_with_book_info(
    pdf_path: str | Path,
    book: BookInfo,
    output_path: str | Path | None = None,
) -> Path:
    """
    Update a PDF's metadata using information from a GoogleBooksVolume.
    If ``output_path`` is None, the original file is overwritten.
    """
    pdf_path = Path(pdf_path)
    if output_path is None:
        output_path = pdf_path
        incremental = True
    else:
        output_path = Path(output_path)
        incremental = output_path == pdf_path

    metadata_updates = _metadata_from_bookinfo(book)
    if metadata_updates:
        with fitz.open(pdf_path) as doc:
            new_metadata = dict(doc.metadata or {})
            new_metadata.update({k: v for k, v in metadata_updates.items() if v})
            doc.set_metadata(new_metadata)
            if incremental:
                doc.save(
                    output_path,
                    incremental=True,
                    encryption=fitz.PDF_ENCRYPT_KEEP,
                )
            else:
                doc.save(
                    output_path,
                    garbage=3,
                    deflate=True,
                    linear=True,
                )

    return output_path


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_pdf = Path("test_pdfs/bladesinthedark_v8_2.pdf")
    book = BookInfo(
        author="John Harper",
        author_confidence=1.0,
        title="Blades in the Dark",
        title_confidence=1.0,
        publisher="Evil Hat Productions, LLC",
        publisher_confidence=1.0,
        subject="Role Playing Games",
        keywords=[
            "Role Playing Game",
            "Tabletop RPG",
            "Game Design",
            "Fantasy",
            "Fiction",
            "Game Manual",
            "Blades in the Dark",
            "Evil Hat Productions",
            "John Harper",
            "One Seven Design",
            "Digital Release",
            "2017",
        ],
        isbn_identifiers=["9781234567897"],
        isbn_confidence=1.0,
        model_ocr="Sample transcription of key text visible on the cover.",
    )

    output_pdf = sample_pdf  # .with_stem(f"{sample_pdf.stem}_augmented")

    logging.info("Augmenting %s -> %s", sample_pdf, output_pdf)
    augment_pdf_with_book_info(sample_pdf, book, output_pdf)
    with fitz.open(output_pdf) as doc:
        logging.info("Updated metadata: %s", doc.metadata)
    logging.info("Done")
