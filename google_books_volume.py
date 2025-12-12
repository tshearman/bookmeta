

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class IndustryIdentifier:
    type: Optional[str] = None
    identifier: Optional[str] = None


@dataclass
class VolumeInfo:
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    description: Optional[str] = None
    industry_identifiers: List[IndustryIdentifier] = field(default_factory=list)
    reading_modes: dict[str, bool] = field(default_factory=dict)
    page_count: Optional[int] = None
    print_type: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    maturity_rating: Optional[str] = None
    allow_anon_logging: Optional[bool] = None
    content_version: Optional[str] = None
    language: Optional[str] = None
    preview_link: Optional[str] = None
    info_link: Optional[str] = None
    canonical_volume_link: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VolumeInfo":
        identifiers = [
            IndustryIdentifier(
                type=item.get("type"),
                identifier=item.get("identifier"),
            )
            for item in data.get("industryIdentifiers", []) or []
        ]
        return cls(
            title=data.get("title"),
            authors=data.get("authors", []) or [],
            published_date=data.get("publishedDate"),
            description=data.get("description"),
            industry_identifiers=identifiers,
            reading_modes=data.get("readingModes", {}) or {},
            page_count=data.get("pageCount"),
            print_type=data.get("printType"),
            categories=data.get("categories", []) or [],
            maturity_rating=data.get("maturityRating"),
            allow_anon_logging=data.get("allowAnonLogging"),
            content_version=data.get("contentVersion"),
            language=data.get("language"),
            preview_link=data.get("previewLink"),
            info_link=data.get("infoLink"),
            canonical_volume_link=data.get("canonicalVolumeLink"),
        )


@dataclass
class SaleInfo:
    country: Optional[str] = None
    saleability: Optional[str] = None
    is_ebook: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SaleInfo":
        return cls(
            country=data.get("country"),
            saleability=data.get("saleability"),
            is_ebook=data.get("isEbook"),
        )


@dataclass
class AccessInfo:
    country: Optional[str] = None
    viewability: Optional[str] = None
    embeddable: Optional[bool] = None
    public_domain: Optional[bool] = None
    text_to_speech_permission: Optional[str] = None
    epub_available: Optional[bool] = None
    pdf_available: Optional[bool] = None
    web_reader_link: Optional[str] = None
    access_view_status: Optional[str] = None
    quote_sharing_allowed: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessInfo":
        epub = data.get("epub") or {}
        pdf = data.get("pdf") or {}
        return cls(
            country=data.get("country"),
            viewability=data.get("viewability"),
            embeddable=data.get("embeddable"),
            public_domain=data.get("publicDomain"),
            text_to_speech_permission=data.get("textToSpeechPermission"),
            epub_available=epub.get("isAvailable"),
            pdf_available=pdf.get("isAvailable"),
            web_reader_link=data.get("webReaderLink"),
            access_view_status=data.get("accessViewStatus"),
            quote_sharing_allowed=data.get("quoteSharingAllowed"),
        )


@dataclass
class SearchInfo:
    text_snippet: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchInfo":
        if not data:
            return cls()
        return cls(text_snippet=data.get("textSnippet"))


@dataclass
class GoogleBooksVolume:
    kind: Optional[str] = None
    volume_id: Optional[str] = None
    etag: Optional[str] = None
    self_link: Optional[str] = None
    volume_info: Optional[VolumeInfo] = None
    sale_info: Optional[SaleInfo] = None
    access_info: Optional[AccessInfo] = None
    search_info: Optional[SearchInfo] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleBooksVolume":
        return cls(
            kind=data.get("kind"),
            volume_id=data.get("id"),
            etag=data.get("etag"),
            self_link=data.get("selfLink"),
            volume_info=VolumeInfo.from_dict(data.get("volumeInfo", {})),
            sale_info=SaleInfo.from_dict(data.get("saleInfo", {})),
            access_info=AccessInfo.from_dict(data.get("accessInfo", {})),
            search_info=SearchInfo.from_dict(data.get("searchInfo", {})),
            raw=data,
        )

