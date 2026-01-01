from typing import Union
from bookmetarefactor.types import Page, PageNumber
from bookmetarefactor.utils.img import img_to_url
from dataclasses import asdict, dataclass
from pathlib import Path
from PIL import Image as PILImage
from typing import Any, Protocol, TypeVar, Union
import hashlib
import json


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
        serialized = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
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
        digest = hashlib.sha256()
        digest.update(self.prompt.hash.encode("utf-8"))
        digest.update(self.path.hash.encode("utf-8"))
        if self.metadata:
            digest.update(self.metadata.hash.encode("utf-8"))
        if self.ocr:
            for ocr_block in self.ocr:
                digest.update(ocr_block.hash.encode("utf-8"))
        if self.images:
            for image_block in self.images:
                digest.update(image_block.hash.encode("utf-8"))
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
            ctx.extend([b.as_context_block for b in self.ocr])

        if self.images:
            ctx.extend([b.as_context_block for b in self.images])

        return ctx

    @property
    def text_blocks(self) -> "TextBlocks":
        blocks: TextBlocks = [self.prompt, self.path]
        if self.metadata:
            meta_ctx = self.metadata
            if meta_ctx:
                blocks.append(meta_ctx)
        if self.ocr:
            blocks.extend(self.ocr)
        return blocks

    @property
    def img_blocks(self) -> "ImageBlocks":
        return self.images if self.images is not None else []


type OcrBlocks = list[OcrBlock]
type TextBlocks = list[Union[PromptBlock, PathBlock, MetadataBlock, OcrBlock]]
type ImageBlocks = list[ImageBlock]
type ContextBlocks = list[ContextBlock]
