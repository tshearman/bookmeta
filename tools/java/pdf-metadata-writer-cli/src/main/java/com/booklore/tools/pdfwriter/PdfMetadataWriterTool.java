package com.booklore.tools.pdfwriter;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class PdfMetadataWriterTool {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper().registerModule(new JavaTimeModule());

    private PdfMetadataWriterTool() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3) {
            printUsage();
            System.exit(1);
        }

        Path sourcePdf = Paths.get(args[0]).toAbsolutePath();
        Path metadataJson;
        Path destination;

        if (args.length == 2) {
            metadataJson = Paths.get(args[1]).toAbsolutePath();
            destination = Paths.get("augmented").resolve(sourcePdf.getFileName());
        } else {
            Path candidate = Paths.get(args[1]).toAbsolutePath();
            metadataJson = Paths.get(args[2]).toAbsolutePath();
            destination = resolveDestination(candidate, sourcePdf.getFileName().toString());
        }

        if (!Files.exists(sourcePdf)) {
            System.err.println("Source PDF not found: " + sourcePdf);
            System.exit(1);
        }
        if (!Files.exists(metadataJson)) {
            System.err.println("Metadata JSON not found: " + metadataJson);
            System.exit(1);
        }

        BookMetadataInput metadata = OBJECT_MAPPER.readValue(metadataJson.toFile(), BookMetadataInput.class);
        PdfMetadataAugmenter augmenter = new PdfMetadataAugmenter();
        Path normalizedDest = destination.toAbsolutePath();
        Path parent = normalizedDest.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        augmenter.writeMetadata(sourcePdf.toFile(), normalizedDest.toFile(), metadata);
        System.out.println("Augmented PDF written to " + normalizedDest);
    }

    private static Path resolveDestination(Path candidate, String originalFileName) {
        if (Files.exists(candidate) && Files.isDirectory(candidate)) {
            return candidate.resolve(originalFileName);
        }
        String name = candidate.getFileName() != null ? candidate.getFileName().toString().toLowerCase() : "";
        if (name.endsWith(".pdf") || name.contains(".")) {
            return candidate;
        }
        return candidate.resolve(originalFileName);
    }

    private static void printUsage() {
        System.out.println("Usage: pdf-metadata-writer <pdf> [destination] <metadata-json>");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  ./gradlew run --args=\"/tmp/book.pdf /tmp/book-metadata.json\"");
        System.out.println("  ./gradlew run --args=\"/tmp/book.pdf /tmp/output-dir /tmp/book-metadata.json\"");
    }
}
