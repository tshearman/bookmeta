package com.booklore.tools.pdf;

import java.io.File;

public interface FileMetadataExtractor {
    byte[] extractCover(File file);

    BookMetadata extractMetadata(File file);
}
