# PDF Metadata Writer CLI

A companion CLI that mirrors BookLore's `PdfMetadataWriter`. It takes an input PDF and a JSON metadata file, embeds the metadata into the PDF, and writes the augmented file to a destination path (defaults to `augmented/<filename>`).

## Usage

```bash
cd tools/java
./gradlew :pdf-metadata-writer-cli:run --args="/path/original.pdf /path/metadata.json"
```

To control the output location, supply a destination between the input PDF and metadata JSON:

```bash
# Write to a directory (filename preserved)
./gradlew :pdf-metadata-writer-cli:run --args="/path/original.pdf /path/output/dir /path/metadata.json"

# Write to an explicit file
./gradlew :pdf-metadata-writer-cli:run --args="/path/original.pdf /path/output/augmented.pdf /path/metadata.json"
```

When the destination is omitted, the tool writes to `augmented/<original-filename>` relative to the current directory. The metadata JSON should match the structure produced by the extractor tool (e.g., the JSON emitted from `pdf-metadata-extractor-cli`).
