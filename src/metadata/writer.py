#!/usr/bin/env python3
"""
Python equivalent of PdfMetadataWriter.

This script copies a source PDF into a target directory and injects metadata
using the same fields the Java PdfMetadataWriter handles. The metadata JSON
must conform to the BookMetadata shape produced by Booklore. Optional clear
flags can be provided via a MetadataClearFlags JSON file.

Dependencies:
    pip install PyPDF2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

try:
    from PyPDF2 import PdfReader, PdfWriter
    from PyPDF2.generic import DecodedStreamObject, NameObject
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "PyPDF2 is required. Install it via 'pip install PyPDF2'."
    ) from exc


XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpidq": "http://ns.adobe.com/xmp/Identifier/qual/1.0/",
    "calibre": "http://calibre-ebook.com/xmp-namespace",
    "calibreSI": "http://calibre-ebook.com/xmp-namespace-series-index",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def q(prefix: str, local: str) -> ET.QName:
    return ET.QName(NS[prefix], local)


class MetadataHelper:
    """Utility functions to apply clear flags and normalize metadata values."""

    def __init__(
        self, metadata: Dict[str, Any], clear_flags: Optional[Dict[str, Any]] = None
    ):
        self.metadata = metadata or {}
        self.clear_flags = clear_flags or {}

    def _cleared(self, key: str) -> bool:
        return bool(self.clear_flags.get(key))

    def text(self, key: str) -> Optional[str]:
        if self._cleared(key):
            return None
        value = self.metadata.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value)

    def string_list(self, key: str) -> List[str]:
        if self._cleared(key):
            return []
        value = self.metadata.get(key)
        if value is None:
            return []
        iterable: Iterable[Any]
        if isinstance(value, str):
            iterable = [value]
        elif isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            return []

        results: List[str] = []
        for item in iterable:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    results.append(trimmed)
        return results

    def number(self, key: str) -> Optional[float]:
        if self._cleared(key):
            return None
        value = self.metadata.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def published_date(self) -> Optional[datetime]:
        if self._cleared("publishedDate"):
            return None
        value = self.metadata.get("publishedDate")
        if value is None:
            return None
        try:
            if isinstance(value, str):
                dt = datetime.fromisoformat(value)
            elif isinstance(value, list) and len(value) == 3:
                year, month, day = value
                dt = datetime(int(year), int(month), int(day))
            else:
                return None
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject metadata into a PDF (Python port of PdfMetadataWriter)."
    )
    parser.add_argument("--pdf", required=True, help="Path to the source PDF")
    parser.add_argument(
        "--out", required=True, help="Directory where the updated PDF should be written"
    )
    parser.add_argument("--metadata", required=True, help="Path to BookMetadata JSON")
    parser.add_argument("--clear", help="Path to MetadataClearFlags JSON")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_copy(source_pdf: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source_pdf.name
    if source_pdf.resolve() != target.resolve():
        shutil.copy2(source_pdf, target)
    return target


def build_docinfo(helper: MetadataHelper) -> Dict[str, str]:
    def join(items: Iterable[str]) -> str:
        filtered = [item for item in items if item]
        return ", ".join(filtered)

    publisher_values = helper.string_list("publisher")
    return {
        "/Title": helper.text("title") or "",
        "/Producer": publisher_values[0] if publisher_values else "",
        "/Author": join(helper.string_list("authors")),
        "/Keywords": join(helper.string_list("categories")),
    }


def _add_alt_text(parent: ET.Element, tag: ET.QName, text: Optional[str]) -> None:
    if not text:
        return
    container = ET.SubElement(parent, tag)
    alt = ET.SubElement(container, q("rdf", "Alt"))
    li = ET.SubElement(alt, q("rdf", "li"), {ET.QName(XML_NS, "lang"): "x-default"})
    li.text = text


def _add_collection(
    parent: ET.Element, tag: ET.QName, items: Iterable[str], collection: str
) -> None:
    values = [item for item in items if item]
    if not values:
        return
    elem = ET.SubElement(parent, tag)
    coll = ET.SubElement(elem, q("rdf", collection))
    for value in values:
        li = ET.SubElement(coll, q("rdf", "li"))
        li.text = value


def _add_simple(parent: ET.Element, tag: ET.QName, text: Optional[str]) -> None:
    if not text:
        return
    elem = ET.SubElement(parent, tag)
    elem.text = text


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_xmp(helper: MetadataHelper) -> bytes:
    xmpmeta = ET.Element(q("x", "xmpmeta"))
    rdf = ET.SubElement(xmpmeta, q("rdf", "RDF"))

    description = ET.SubElement(rdf, q("rdf", "Description"), {q("rdf", "about"): ""})

    _add_alt_text(description, q("dc", "title"), helper.text("title"))
    _add_alt_text(description, q("dc", "description"), helper.text("description"))
    _add_collection(
        description, q("dc", "publisher"), helper.string_list("publisher"), "Bag"
    )
    _add_collection(
        description, q("dc", "language"), helper.string_list("language"), "Bag"
    )
    _add_collection(
        description, q("dc", "creator"), helper.string_list("authors"), "Seq"
    )
    _add_collection(
        description, q("dc", "subject"), helper.string_list("categories"), "Bag"
    )

    published = helper.published_date()
    if published:
        elem = ET.SubElement(description, q("dc", "date"))
        seq = ET.SubElement(elem, q("rdf", "Seq"))
        li = ET.SubElement(seq, q("rdf", "li"))
        li.text = published.isoformat()

    identifier_entries: List[Tuple[str, Optional[str]]] = [
        ("google", helper.text("googleId")),
        ("goodreads", helper.text("goodreadsId")),
        ("comicvine", helper.text("comicvineId")),
        ("hardcover", helper.text("hardcoverId")),
        ("amazon", helper.text("asin")),
        ("isbn", helper.text("isbn13") or helper.text("isbn10")),
    ]
    identifier_entries = [
        (scheme, value) for scheme, value in identifier_entries if value
    ]
    if identifier_entries:
        identifier = ET.SubElement(description, q("xmp", "Identifier"))
        bag = ET.SubElement(identifier, q("rdf", "Bag"))
        for scheme, value in identifier_entries:
            li = ET.SubElement(
                bag,
                q("rdf", "li"),
                {q("rdf", "parseType"): "Resource"},
            )
            scheme_elem = ET.SubElement(li, q("xmpidq", "Scheme"))
            scheme_elem.text = scheme
            value_elem = ET.SubElement(li, q("rdf", "value"))
            value_elem.text = value

    now_iso = _iso_now()
    _add_simple(description, q("xmp", "MetadataDate"), now_iso)
    create_date = published.isoformat() if published else now_iso
    _add_simple(description, q("xmp", "CreateDate"), create_date)
    _add_simple(description, q("xmp", "CreatorTool"), "Booklore Python MetadataWriter")
    _add_simple(description, q("xmp", "ModifyDate"), now_iso)

    series_name = helper.text("seriesName")
    series_number = helper.number("seriesNumber")
    if series_name or series_number is not None:
        calibre_desc = ET.SubElement(
            rdf, q("rdf", "Description"), {q("rdf", "about"): ""}
        )
        series_elem = ET.SubElement(
            calibre_desc, q("calibre", "series"), {q("rdf", "parseType"): "Resource"}
        )
        value_elem = ET.SubElement(series_elem, q("rdf", "value"))
        value_elem.text = series_name or ""
        index_elem = ET.SubElement(series_elem, q("calibreSI", "series_index"))
        index_elem.text = (
            f"{series_number:.2f}" if series_number is not None else "0.00"
        )

    return ET.tostring(xmpmeta, encoding="utf-8", xml_declaration=False)


def embed_metadata(target_pdf: Path, helper: MetadataHelper) -> None:
    reader = PdfReader(str(target_pdf))
    writer = PdfWriter()
    append = getattr(writer, "append_pages_from_reader", None)
    if callable(append):
        append(reader)
    else:
        for page in reader.pages:
            writer.add_page(page)

    writer.add_metadata(build_docinfo(helper))

    xmp_bytes = build_xmp(helper)
    metadata_stream = DecodedStreamObject()
    setter = getattr(metadata_stream, "set_data", None)
    if callable(setter):
        setter(xmp_bytes)
    else:
        metadata_stream.setData(xmp_bytes)
    metadata_stream.update(
        {
            NameObject("/Type"): NameObject("/Metadata"),
            NameObject("/Subtype"): NameObject("/XML"),
        }
    )
    writer._root_object.update(
        {NameObject("/Metadata"): writer._add_object(metadata_stream)}
    )

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="pdfmeta-out-", suffix=".pdf")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as handle:
            writer.write(handle)
        shutil.move(tmp_path, target_pdf)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> None:
    args = parse_args()
    source_pdf = Path(args.pdf).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    metadata = load_json(Path(args.metadata).expanduser().resolve())
    clear_flags = (
        load_json(Path(args.clear).expanduser().resolve()) if args.clear else {}
    )

    if not source_pdf.exists():
        raise SystemExit(f"PDF file does not exist: {source_pdf}")

    target_pdf = ensure_copy(source_pdf, output_dir)
    helper = MetadataHelper(metadata, clear_flags)

    backup_fd, backup_path = tempfile.mkstemp(prefix="pdfmeta-backup-", suffix=".pdf")
    os.close(backup_fd)
    shutil.copy2(target_pdf, backup_path)

    try:
        embed_metadata(target_pdf, helper)
        print(f"Metadata embedded into {target_pdf}")
    except Exception as exc:
        shutil.copy2(backup_path, target_pdf)
        raise SystemExit(f"Failed to embed metadata: {exc}") from exc
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)


if __name__ == "__main__":
    main()
