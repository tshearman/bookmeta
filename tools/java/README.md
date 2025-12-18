# Java CLI Tools

All JVM utilities live under this multi-module Gradle build:

- `pdf-metadata-core` – shared PdfBox/Jackson helpers.
- `pdf-metadata-extractor-cli` – reads PDF metadata and optional cover renders.
- `pdf-metadata-writer-cli` – embeds normalized metadata back into a PDF.

From this directory you can run any CLI:

```bash
./gradlew :pdf-metadata-extractor-cli:run --args="/path/to/book.pdf"
./gradlew :pdf-metadata-writer-cli:run --args="/path/to/book.pdf /path/to/output/dir /path/to/metadata.json"
```

See each subproject's README for details.
