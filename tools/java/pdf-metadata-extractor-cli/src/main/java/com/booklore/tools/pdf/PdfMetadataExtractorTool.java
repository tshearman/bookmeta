package com.booklore.tools.pdf;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public final class PdfMetadataExtractorTool {

    private PdfMetadataExtractorTool() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsage();
            System.exit(1);
        }

        Path coverOutput = null;
        List<String> positional = new ArrayList<>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--cover".equals(arg)) {
                if (i + 1 >= args.length) {
                    System.err.println("Missing value for --cover option");
                    System.exit(1);
                }
                coverOutput = Paths.get(args[++i]);
            } else if ("--help".equals(arg) || "-h".equals(arg)) {
                printUsage();
                return;
            } else {
                positional.add(arg);
            }
        }

        if (positional.isEmpty()) {
            System.err.println("You must supply a PDF file path.");
            printUsage();
            System.exit(1);
        }

        Path pdfPath = Paths.get(positional.getFirst());
        File pdfFile = pdfPath.toFile();
        if (!pdfFile.exists()) {
            System.err.println("PDF file not found: " + pdfPath);
            System.exit(1);
        }

        PdfMetadataExtractor extractor = new PdfMetadataExtractor();
        BookMetadata metadata = extractor.extractMetadata(pdfFile);
        System.out.println(metadata.toPrettyJson());

        if (coverOutput != null) {
            byte[] cover = extractor.extractCover(pdfFile);
            if (cover == null) {
                System.err.println("Cover extraction failed; no file written.");
            } else {
                Path parent = coverOutput.toAbsolutePath().getParent();
                if (parent != null) {
                    Files.createDirectories(parent);
                }
                Files.write(coverOutput, cover);
                System.out.println("Cover image written to " + coverOutput.toAbsolutePath());
            }
        }
    }

    private static void printUsage() {
        System.out.println("Usage: pdf-metadata-extractor <pdf-file> [--cover <jpg-output>]");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  ./gradlew run --args=\"/tmp/book.pdf\"");
        System.out.println("  ./gradlew run --args=\"/tmp/book.pdf --cover /tmp/cover.jpg\"");
    }
}
