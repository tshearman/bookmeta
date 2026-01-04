from typing import Iterable

from bookmeta.config import MAX_LONG_EDGE_IMG
from bookmeta.types import Pdf
from bookmeta.types.blocks import (
    BlockWithPageInfo,
    ExtractionBlocks,
    ImageBlock,
    ImageBlocks,
    MetadataBlock,
    OcrBlock,
    OcrBlocks,
    PathBlock,
    PromptBlock,
)
from bookmeta.types.extraction import ContextLimits, ExtractionConfig
from bookmeta.types.ocr import OcrResults
from bookmeta.utils.doc import extract_metadata
from bookmeta.utils.page import page_to_image


def prompt_block(config: ExtractionConfig) -> PromptBlock:
    return PromptBlock(config.prompt)


def context_path_block(pdf: Pdf, config: ExtractionConfig) -> PathBlock:
    context = config.context_path if config.context_path else pdf.path.parent
    return PathBlock(context / pdf.path.stem)


def metadata_block(pdf: Pdf) -> MetadataBlock:
    """Read PDF metadata directly from the source file."""
    return MetadataBlock(extract_metadata(pdf))


def ocr_blocks(ocr_results: OcrResults) -> OcrBlocks:
    """Build labeled OCR text blocks honoring configured page limits."""
    return [
        OcrBlock(
            ocr.task.page, ocr.task.page_number, ocr.output or "", ocr.task.method.name
        )
        for ocr in ocr_results
        if ocr.output
    ]


def image_blocks(ocr_results: OcrResults) -> ImageBlocks:
    images: ImageBlocks = []
    for ocr in ocr_results:
        page = ocr.task.page
        page_number = ocr.task.page_number
        img = page_to_image(page, grayscale=False, max_long_edge=MAX_LONG_EDGE_IMG)
        images.append(ImageBlock(page, page_number, img))
    return images


def blocks_from_ocr(
    pdf: Pdf, ocr_results: OcrResults, config: ExtractionConfig
) -> ExtractionBlocks:
    """Convert OCR output into structured blocks (prompt, path, metadata, ocr, images)."""
    prompt = prompt_block(config)
    path = context_path_block(pdf, config)
    metadata = metadata_block(pdf)
    ocr = ocr_blocks(ocr_results)
    images = image_blocks(ocr_results)

    return ExtractionBlocks(
        prompt=prompt, path=path, metadata=metadata, ocr=ocr, images=images
    )


def filter_blocks(blocks: ExtractionBlocks, limits: ContextLimits) -> ExtractionBlocks:
    """Apply context limits to OCR and image blocks."""

    def _select_blocks_by_page(
        blocks: Iterable[BlockWithPageInfo],
        num_first: int | None,
        num_last: int | None,
    ) -> list[BlockWithPageInfo]:
        """Select first/last windows by page number while preserving order and de-duping."""

        ordered = sorted(blocks, key=lambda b: b.page_number)

        if num_first is None and num_last is None:
            return ordered

        first = ordered[: num_first or 0]
        last = ordered[-(num_last or 0) :] if num_last else []

        seen: set[int] = set()
        selected: list[BlockWithPageInfo] = []
        for block in first + last:
            if block.page_number in seen:
                continue
            selected.append(block)
            seen.add(block.page_number)
        return selected

    ocr_filtered = (
        _select_blocks_by_page(
            blocks.ocr,
            limits.num_first_ocr_pages,
            limits.num_last_ocr_pages,
        )
        if blocks.ocr
        else None
    )
    images_filtered = (
        _select_blocks_by_page(
            blocks.images,
            limits.num_first_images,
            limits.num_last_images,
        )
        if blocks.images
        else None
    )

    return ExtractionBlocks(
        prompt=blocks.prompt,
        path=blocks.path,
        metadata=blocks.metadata,
        ocr=ocr_filtered,
        images=images_filtered,
    )
