# PDF Metadata Extractor CLI

A minimal command-line tool that reuses the same logic as BookLore's `PdfMetadataExtractor`. Use it to inspect a PDF's embedded metadata (and optionally export the cover) without running the full application.

## Building & Running

From the repository root:

```bash
cd tools/java
./gradlew :pdf-metadata-extractor-cli:run --args="/path/to/book.pdf"
```

To save the rendered cover image alongside the metadata output:

```bash
./gradlew :pdf-metadata-extractor-cli:run --args="/path/to/book.pdf --cover /tmp/cover.jpg"
```

The tool prints the extracted metadata as pretty-printed JSON. When `--cover` is supplied it also writes a JPEG of the first page rendered at 300 DPI.
