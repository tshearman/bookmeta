import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, Union

from PIL import Image as PILImage

from bookmeta.types import Page, PageNumber
from bookmeta.utils.img import img_to_url


class _HasPageInfo(Protocol):
    page: Page
    page_number: PageNumber


BlockWithPageInfo = TypeVar("BlockWithPageInfo", bound=_HasPageInfo)


@dataclass(frozen=True)
class ContextBlock:
    type: str
    content: dict[str, Any]

    @classmethod
    def from_text(cls, text: str) -> "ContextBlock":
        return cls(type="input_text", content={"text": text})

    @classmethod
    def from_image(cls, image: PILImage.Image) -> "ContextBlock":
        return cls(
            type="input_image",
            content={"image_url": img_to_url(image)},
        )

    def as_payload(self) -> dict[str, Any]:
        return {"type": self.type, **self.content}


class PromptBlock(str):

    @property
    def as_context_block(self) -> ContextBlock:
        return ContextBlock.from_text(str(self))

    @property
    def hash(self) -> str:
        return hashlib.sha256(str(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PathBlock:
    path: Path

    @property
    def as_context_block(self) -> ContextBlock:
        formatted = (
            "PDF FILE:\n" f"Relative Path: {self.path}\n" f"File Name: {self.path.stem}"
        )
        return ContextBlock.from_text(formatted)

    @property
    def hash(self) -> str:
        return hashlib.sha256(str(self.path).encode("utf-8")).hexdigest()


class MetadataBlock(dict[str, Any]):

    @property
    def as_context_block(self) -> ContextBlock | None:
        lines = [f"{key.title()}: {value}" for key, value in self.items() if value]
        if not lines:
            return None
        return ContextBlock.from_text("PDF METADATA\n" + "\n".join(lines))

    @property
    def hash(self) -> str:
        serialized = json.dumps(dict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OcrBlock(_HasPageInfo):
    page: Page
    page_number: PageNumber
    text: str
    method_name: str

    @property
    def as_context_block(self) -> ContextBlock:
        formatted = (
            f"PAGE: {self.page_number} OCR USING METHOD: {self.method_name}\n"
            f"{self.text or ''}"
        )
        return ContextBlock.from_text(formatted)

    @property
    def hash(self) -> str:
        payload = {
            "page_number": self.page_number,
            "text": self.text,
            "method_name": self.method_name,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImageBlock(_HasPageInfo):
    page: Page
    page_number: PageNumber
    image: PILImage.Image

    @property
    def as_context_block(self) -> ContextBlock:
        return ContextBlock.from_image(self.image)

    @property
    def hash(self) -> str:
        payload = {
            "page_number": self.page_number,
            "mode": self.image.mode,
            "size": self.image.size,
        }
        serialized = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8"))
        digest.update(self.image.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class ExtractionBlocks:
    prompt: PromptBlock
    path: PathBlock
    metadata: MetadataBlock | None
    ocr: "OcrBlocks | None"
    images: "ImageBlocks | None"

    @property
    def hash(self) -> str:
        hashes = [
            self.prompt.hash,
            self.path.hash,
        ]
        if self.metadata:
            hashes.append(self.metadata.hash)
        if self.ocr:
            for ocr_block in self.ocr:
                hashes.append(ocr_block.hash)
        if self.images:
            for image_block in self.images:
                hashes.append(image_block.hash)
        digest = hashlib.sha256()
        for h in sorted(hashes):
            digest.update(h.encode("utf-8"))
        return digest.hexdigest()

    @property
    def as_context_blocks(self) -> "ContextBlocks":
        ctx: ContextBlocks = []
        ctx.append(self.prompt.as_context_block)
        ctx.append(self.path.as_context_block)

        if self.metadata:
            meta_ctx = self.metadata.as_context_block
            if meta_ctx:
                ctx.append(meta_ctx)

        if self.ocr:
            ctx.extend(
                [
                    b.as_context_block
                    for b in sorted(
                        self.ocr, key=lambda o: (o.method_name, o.page_number)
                    )
                ]
            )

        if self.images:
            ctx.extend(
                [
                    b.as_context_block
                    for b in sorted(self.images, key=lambda o: o.page_number)
                ]
            )

        return ctx


type OcrBlocks = list[OcrBlock]
type TextBlocks = list[Union[PromptBlock, PathBlock, MetadataBlock, OcrBlock]]
type ImageBlocks = list[ImageBlock]
type ContextBlocks = list[ContextBlock]
